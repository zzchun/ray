"""Drain of underutilized worker nodes for autoscaler v2.

Autoscaler v2 only terminates fully idle nodes by default. This module adds an
opt-in policy that drains worker nodes that have stayed underutilized for a
while, provided that every actor and running task on the node is expected to
fit on the remaining nodes. Drained actors are restarted by Ray (node
preemption restarts don't consume `max_restarts`), and tasks are retried
(retries due to node preemption don't consume `max_retries`).

Only existing GCS / raylet RPCs are used. The pieces are:

    - `UnderutilizedNodeDrainConfig`: the parsed `underutilized_node_drain`
      autoscaling config.
    - `build_node_workload`: turns the raylet's per-worker stats and the GCS
      actor table into a `NodeWorkload`, validating that the stats are complete.
    - `UnderutilizationTracker`: tracks how long each node has been
      underutilized.
    - `UnderutilizedDrainRegistry`: in-memory records of the nodes being
      drained, shared by the scheduler, the reconciler and the RayStopper.
    - `NodeWorkloadFetcher`: fetches the actor table and the raylet stats.
    - `UnderutilizedNodeDrainer`: ties the above together for each
      reconciliation round.

The node selection and the feasibility simulation themselves live in
`ResourceDemandScheduler._enforce_underutilized_drain`.
"""

import asyncio
import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from ray._common.utils import binary_to_hex, hex_to_binary
from ray.autoscaler.v2.utils import ResourceRequestUtil
from ray.core.generated.autoscaler_pb2 import (
    ClusterResourceState,
    NodeState,
    NodeStatus,
    ResourceRequest,
)
from ray.core.generated.common_pb2 import LabelSelectorOperator, WorkerType
from ray.core.generated.instance_manager_pb2 import (
    Instance,
    NodeKind,
    TerminationRequest,
)

logger = logging.getLogger(__name__)

# Termination requests of underutilized node drains are identified by this
# prefix of `TerminationRequest.details` (with cause UNKNOWN), so that no
# protobuf change is needed.
UNDERUTILIZED_DRAIN_DETAILS_PREFIX = "underutilized node drain"

# Nodes with this label set to "true" are never drained for underutilization.
DRAIN_PROTECTED_LABEL = "ray.io/drain-protected"

# The label key used by label selectors to pin a task/actor to a node.
NODE_ID_LABEL = "ray.io/node-id"

# Dynamic label prefix of the placement groups on a node.
PLACEMENT_GROUP_LABEL_PREFIX = "_PG_"

# Resources that are not counted towards the utilization of a node, nor
# reconciled against the per-worker resource usage.
_NON_WORKLOAD_RESOURCES = {"object_store_memory"}
_NON_WORKLOAD_RESOURCE_PREFIXES = ("node:", "accelerator_type:")

# Resource names that don't make an actor "specify resources", see
# `python/ray/actor.py` (ActorClass._remote).
_ACTOR_DEFAULT_RESOURCE_KEYS = {"memory", "object_store_memory"}

# Length of a binary ActorID. A nil ActorID is all 0xff bytes.
_ACTOR_ID_SIZE = 16


def is_underutilized_drain_request(request: Optional[TerminationRequest]) -> bool:
    """Returns True if the termination request is an underutilized node drain."""
    return (
        request is not None
        and request.cause == TerminationRequest.Cause.UNKNOWN
        and request.details.startswith(UNDERUTILIZED_DRAIN_DETAILS_PREFIX)
    )


def is_workload_resource(resource_name: str) -> bool:
    """Returns True if the resource is used by tasks/actors' logical resources."""
    return resource_name not in _NON_WORKLOAD_RESOURCES and not any(
        resource_name.startswith(p) for p in _NON_WORKLOAD_RESOURCE_PREFIXES
    )


def dominant_utilization(
    total_resources: Dict[str, float], available_resources: Dict[str, float]
) -> float:
    """The max utilization across the workload resources of a node."""
    utilization = 0.0
    for resource_name, total in total_resources.items():
        if total <= 0 or not is_workload_resource(resource_name):
            continue
        used = total - available_resources.get(resource_name, 0.0)
        utilization = max(utilization, used / total)
    return min(max(utilization, 0.0), 1.0)


def placement_resources(required_resources: Dict[str, float]) -> Dict[str, float]:
    """Resources needed to schedule (i.e. restart) an actor.

    An actor that doesn't specify any resources holds 0 CPU for its lifetime,
    but needs 1 CPU to be scheduled (see `python/ray/actor.py`).
    `ActorTableData.required_resources` holds the lifetime resources, with zero
    values dropped.
    """
    resources = {k: v for k, v in required_resources.items() if v > 0}
    if not set(resources) - _ACTOR_DEFAULT_RESOURCE_KEYS:
        resources["CPU"] = resources.get("CPU", 0.0) + 1.0
    return resources


def parse_label_selector(
    label_selector: Dict[str, str]
) -> Optional[List[Tuple[str, int, List[str]]]]:
    """Parses a label selector map, e.g. {"region": "!in(us-west1, us-east1)"}.

    Supported values are `v`, `!v`, `in(v1, v2)`, `!in(v1, v2)`, `(v1, v2)`
    and `!(v1, v2)`, following `ray._private.label_utils.LABEL_SELECTOR_REGEX`.

    Args:
        label_selector: The label selector map of an actor.

    Returns:
        A list of (label_key, operator, label_values) constraints as accepted
        by `ResourceRequestUtil.make`, or None if any value can't be parsed.
    """
    constraints = []
    for key, raw_value in label_selector.items():
        value = (raw_value or "").strip()
        if not key or not value:
            return None
        negated = value.startswith("!")
        if negated:
            value = value[1:].strip()
        if value.endswith(")"):
            if value.startswith("in("):
                value = value[len("in(") :]
            elif value.startswith("("):
                value = value[1:]
            else:
                return None
            values = [v.strip() for v in value[:-1].split(",")]
        else:
            values = [value]
        if not values or any(not v or "(" in v or ")" in v for v in values):
            return None
        operator = (
            LabelSelectorOperator.LABEL_OPERATOR_NOT_IN
            if negated
            else LabelSelectorOperator.LABEL_OPERATOR_IN
        )
        constraints.append((key, operator, values))
    return constraints


def sum_resource_allocations(used_resources) -> Dict[str, float]:
    """Sums `CoreWorkerStats.used_resources` (map<string, ResourceAllocations>)."""
    resources = {}
    for resource_name, allocations in used_resources.items():
        amount = sum(slot.allocation for slot in allocations.resource_slots)
        if amount > 0:
            resources[resource_name] = amount
    return resources


def _is_nil_actor_id(actor_id: bytes) -> bool:
    # Non-actor workers report ActorID::Nil(), which is all 0xff bytes, not an
    # empty string.
    return len(actor_id) == _ACTOR_ID_SIZE and all(b == 0xFF for b in actor_id)


@dataclass(frozen=True)
class UnderutilizedNodeDrainConfig:
    """The `underutilized_node_drain` section of the autoscaling config."""

    enabled: bool = False
    # Only compute and log the decisions, without draining any node.
    dry_run: bool = False
    # How often to evaluate the candidates (fetching the actor table and the
    # raylet stats).
    evaluation_interval_s: float = 60
    # The max number of underutilized nodes to fetch raylet stats for in each
    # evaluation.
    max_candidates_to_inspect: int = 5
    # A node is underutilized if its dominant resource utilization is below it.
    utilization_threshold: float = 0.3
    # How long a node has to stay underutilized to be drained.
    underutilized_duration_s: float = 600
    # The max number of nodes to start draining in one round.
    max_nodes_per_round: int = 1
    # The max number of nodes being drained concurrently, either an int or a
    # percentage of the worker nodes, e.g. "10%".
    max_concurrent_draining: Any = 1
    # The drain deadline after the drain is issued.
    drain_grace_period_s: float = 300
    # Extra time to wait after the drain deadline before terminating the
    # instance, to tolerate clock skew between nodes.
    termination_buffer_s: float = 30
    # Draining instances without a known deadline (e.g. after an autoscaler
    # restart) are terminated after staying in RAY_STOPPING this long.
    ray_stopping_timeout_s: float = 3600
    # Nodes with more running non-actor tasks are not drained.
    max_running_tasks_per_node: int = 10
    # Nodes with more actors are not drained.
    max_actors_per_node: int = 20
    # Nodes using more object store memory are not drained.
    max_object_store_used_bytes: float = 1024**3
    # Skip nodes with actors that own objects, which would be lost (raising
    # OwnerDiedError to their borrowers) when the actors restart.
    skip_if_actor_owns_objects: bool = True
    # Placing the drained workloads must not push any resource of a remaining
    # node above this utilization.
    post_drain_max_utilization: float = 0.9
    # Don't drain any node if the cluster scaled up recently.
    cooldown_after_scale_up_s: float = 600
    # Don't drain nodes with actors in these namespaces / jobs (hex job ids).
    excluded_namespaces: FrozenSet[str] = frozenset()
    excluded_job_ids: FrozenSet[str] = frozenset()
    # Node types that are never drained for underutilization.
    disabled_node_types: FrozenSet[str] = frozenset()
    # Timeout of each GCS / raylet RPC.
    rpc_timeout_s: float = 10
    # Tolerance when reconciling the node's used resources against the
    # resources reported by its workers.
    resource_reconcile_tolerance: float = 0.01

    _NUMERIC_FIELDS = (
        "evaluation_interval_s",
        "utilization_threshold",
        "underutilized_duration_s",
        "drain_grace_period_s",
        "termination_buffer_s",
        "ray_stopping_timeout_s",
        "max_object_store_used_bytes",
        "post_drain_max_utilization",
        "cooldown_after_scale_up_s",
        "rpc_timeout_s",
        "resource_reconcile_tolerance",
    )
    _INT_FIELDS = (
        "max_candidates_to_inspect",
        "max_nodes_per_round",
        "max_running_tasks_per_node",
        "max_actors_per_node",
    )
    _BOOL_FIELDS = ("enabled", "dry_run", "skip_if_actor_owns_objects")
    _LIST_FIELDS = ("excluded_namespaces", "excluded_job_ids")

    @classmethod
    def from_dict(
        cls,
        config: Optional[Dict[str, Any]],
        disabled_node_types: Optional[Set[str]] = None,
    ) -> "UnderutilizedNodeDrainConfig":
        """Parses and validates the config.

        Args:
            config: The raw `underutilized_node_drain` config, None if absent.
            disabled_node_types: The node types that opted out.

        Returns:
            The parsed config.

        Raises:
            ValueError: If the config is invalid.
        """
        config = dict(config or {})
        known = set(
            cls._NUMERIC_FIELDS
            + cls._INT_FIELDS
            + cls._BOOL_FIELDS
            + cls._LIST_FIELDS
            + ("max_concurrent_draining",)
        )
        unknown = set(config) - known
        if unknown:
            raise ValueError(
                f"Unknown underutilized_node_drain config keys: {sorted(unknown)}"
            )

        kwargs: Dict[str, Any] = {}
        for name in cls._BOOL_FIELDS:
            if name in config:
                if not isinstance(config[name], bool):
                    raise ValueError(f"underutilized_node_drain.{name} must be a bool")
                kwargs[name] = config[name]
        for name in cls._INT_FIELDS:
            if name in config:
                value = config[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"underutilized_node_drain.{name} must be a non-negative int"
                    )
                kwargs[name] = value
        for name in cls._NUMERIC_FIELDS:
            if name in config:
                value = config[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                ):
                    raise ValueError(
                        f"underutilized_node_drain.{name} must be a non-negative number"
                    )
                kwargs[name] = float(value)
        for name in cls._LIST_FIELDS:
            if name in config:
                value = config[name]
                if not isinstance(value, list) or not all(
                    isinstance(v, str) for v in value
                ):
                    raise ValueError(
                        f"underutilized_node_drain.{name} must be a list of strings"
                    )
                kwargs[name] = frozenset(value)
        if "max_concurrent_draining" in config:
            value = config["max_concurrent_draining"]
            cls._parse_max_concurrent_draining(value)
            kwargs["max_concurrent_draining"] = value
        for name in ("utilization_threshold", "post_drain_max_utilization"):
            if name in kwargs and kwargs[name] > 1:
                raise ValueError(f"underutilized_node_drain.{name} must be <= 1")

        return cls(disabled_node_types=frozenset(disabled_node_types or ()), **kwargs)

    @staticmethod
    def _parse_max_concurrent_draining(value: Any) -> Tuple[bool, float]:
        """Returns (is_percentage, number)."""
        if isinstance(value, bool):
            raise ValueError("max_concurrent_draining must be an int or 'N%'")
        if isinstance(value, int):
            if value < 0:
                raise ValueError("max_concurrent_draining must be non-negative")
            return False, float(value)
        if isinstance(value, str) and value.strip().endswith("%"):
            try:
                percentage = float(value.strip()[:-1])
            except ValueError:
                raise ValueError(
                    f"Invalid max_concurrent_draining percentage: {value!r}"
                ) from None
            if not 0 <= percentage <= 100:
                raise ValueError(
                    f"max_concurrent_draining percentage must be in [0, 100]: {value!r}"
                )
            return True, percentage
        raise ValueError(f"max_concurrent_draining must be an int or 'N%': {value!r}")

    def get_max_concurrent_draining(self, num_worker_nodes: int) -> int:
        """Resolves `max_concurrent_draining` for the current cluster size.

        A percentage resolves to max(1, floor(num_worker_nodes * N / 100)).
        """
        is_percentage, number = self._parse_max_concurrent_draining(
            self.max_concurrent_draining
        )
        if not is_percentage:
            return int(number)
        if number == 0:
            return 0
        return max(1, math.floor(num_worker_nodes * number / 100))


@dataclass
class ActorOnNode:
    """An actor running on a node, combining the GCS and raylet views."""

    actor_id: str
    job_id: str
    namespace: str
    # Resources held by the actor for its lifetime.
    required_resources: Dict[str, float]
    # Resources needed to restart the actor.
    placement_resources: Dict[str, float]
    # Parsed label selector, None if it can't be parsed.
    label_selector: Optional[List[Tuple[str, int, List[str]]]]
    in_placement_group: bool
    max_restarts: int
    is_detached: bool
    # The number of actor methods being executed.
    num_running_methods: int = 0
    # The number of actors / objects owned by the actor.
    num_owned_actors: int = 0
    num_owned_objects: int = 0


@dataclass
class RunningTask:
    """A worker running (or leased for) non-actor tasks."""

    worker_id: str
    num_running_tasks: int
    # Resources held by the worker's lease.
    resources: Dict[str, float]


@dataclass
class NodeWorkload:
    """The workload of a node that needs to be moved if the node is drained."""

    ray_node_id: str
    actors: List[ActorOnNode] = field(default_factory=list)
    tasks: List[RunningTask] = field(default_factory=list)
    has_driver: bool = False
    # Object store memory used on the node, which may be lost when draining it.
    object_store_used_bytes: float = 0.0
    # Whether the workload passed the completeness validation. Only complete
    # workloads are used to select nodes to drain.
    complete: bool = False
    incomplete_reason: str = ""
    # When the workload was fetched (seconds since epoch).
    fetched_at_s: float = 0.0

    @property
    def num_running_tasks(self) -> int:
        return sum(t.num_running_tasks for t in self.tasks)

    @property
    def num_running_actor_methods(self) -> int:
        return sum(a.num_running_methods for a in self.actors)

    def summary(self) -> str:
        return (
            f"running_tasks={self.num_running_tasks}, actors={len(self.actors)}, "
            f"running_actor_methods={self.num_running_actor_methods}"
        )


def _incomplete(ray_node_id: str, reason: str, fetched_at_s: float) -> NodeWorkload:
    return NodeWorkload(
        ray_node_id=ray_node_id,
        complete=False,
        incomplete_reason=reason,
        fetched_at_s=fetched_at_s,
    )


def build_node_workload(
    ray_node_id: str,
    core_worker_stats: Optional[List[Any]],
    alive_actors_on_node: Dict[str, Any],
    node_used_resources: Dict[str, float],
    resource_reconcile_tolerance: float,
    fetched_at_s: float,
) -> NodeWorkload:
    """Builds the workload of a node and validates its completeness.

    The raylet's GetNodeStats ignores the failure of individual worker RPCs and
    merges an empty `CoreWorkerStats` for them while still returning OK, so the
    stats must be validated before they are trusted (fail closed).

    Args:
        ray_node_id: The hex ray node id.
        core_worker_stats: `GetNodeStatsReply.core_workers_stats` of the node,
            None if the RPC failed.
        alive_actors_on_node: ALIVE `ActorTableData` on the node from GCS, keyed
            by hex actor id.
        node_used_resources: total - available resources of the node from the
            cluster resource state.
        resource_reconcile_tolerance: Tolerance of the resource reconciliation.
        fetched_at_s: When the stats were fetched.

    Returns:
        The node workload. `complete` is False if the stats can't be trusted.
    """
    if core_worker_stats is None:
        return _incomplete(ray_node_id, "GetNodeStats failed", fetched_at_s)

    workload = NodeWorkload(
        ray_node_id=ray_node_id,
        fetched_at_s=fetched_at_s,
        object_store_used_bytes=node_used_resources.get("object_store_memory", 0.0),
    )
    seen_actor_ids = set()
    reported_resources: Dict[str, float] = defaultdict(float)

    for stats in core_worker_stats:
        if not stats.worker_id:
            # A worker whose GetCoreWorkerStats RPC failed.
            return _incomplete(ray_node_id, "missing stats of a worker", fetched_at_s)
        if stats.worker_type == WorkerType.DRIVER:
            workload.has_driver = True
            continue
        if stats.worker_type != WorkerType.WORKER:
            # IO workers (spill/restore) don't run user workloads.
            continue

        used = sum_resource_allocations(stats.used_resources)
        for resource_name, amount in used.items():
            reported_resources[resource_name] += amount

        if _is_nil_actor_id(stats.actor_id):
            if stats.num_running_tasks > 0 or used:
                workload.tasks.append(
                    RunningTask(
                        worker_id=binary_to_hex(stats.worker_id),
                        num_running_tasks=stats.num_running_tasks,
                        resources=used,
                    )
                )
            continue

        actor_id = binary_to_hex(stats.actor_id)
        actor_data = alive_actors_on_node.get(actor_id)
        if actor_data is None:
            # E.g. the actor is being created or restarted.
            return _incomplete(
                ray_node_id,
                f"actor {actor_id} is not ALIVE on the node in GCS",
                fetched_at_s,
            )
        seen_actor_ids.add(actor_id)
        required = {k: v for k, v in actor_data.required_resources.items() if v > 0}
        workload.actors.append(
            ActorOnNode(
                actor_id=actor_id,
                job_id=binary_to_hex(actor_data.job_id),
                namespace=actor_data.ray_namespace,
                required_resources=required,
                placement_resources=placement_resources(required),
                label_selector=parse_label_selector(dict(actor_data.label_selector)),
                in_placement_group=bool(
                    actor_data.HasField("placement_group_id")
                    and actor_data.placement_group_id
                ),
                max_restarts=actor_data.max_restarts,
                is_detached=actor_data.is_detached,
                num_running_methods=stats.num_running_tasks,
                num_owned_actors=stats.num_owned_actors,
                num_owned_objects=stats.num_owned_objects,
            )
        )

    missing_actor_ids = set(alive_actors_on_node) - seen_actor_ids
    if missing_actor_ids:
        return _incomplete(
            ray_node_id,
            f"{len(missing_actor_ids)} ALIVE actors in GCS are missing from the "
            "node's stats",
            fetched_at_s,
        )

    for resource_name, used in node_used_resources.items():
        if not is_workload_resource(resource_name):
            continue
        unaccounted = used - reported_resources.get(resource_name, 0.0)
        if unaccounted > resource_reconcile_tolerance:
            return _incomplete(
                ray_node_id,
                f"{unaccounted} {resource_name} used by the node are not reported "
                "by its workers",
                fetched_at_s,
            )

    workload.complete = True
    return workload


def workload_to_resource_requests(
    workload: NodeWorkload, include_tasks: bool = True
) -> List[ResourceRequest]:
    """Resource requests needed to reschedule the workload of a node.

    Each actor is one request with its placement resources and label selector.
    Each running task worker is one request with its leased resources (tasks'
    scheduling constraints are unknown). Requests without any resource are
    dropped since they fit anywhere.
    """
    requests = []
    for actor in workload.actors:
        if not actor.placement_resources:
            continue
        requests.append(
            ResourceRequestUtil.make(
                actor.placement_resources,
                label_selectors=[actor.label_selector]
                if actor.label_selector
                else None,
            )
        )
    if include_tasks:
        for task in workload.tasks:
            if task.resources:
                requests.append(ResourceRequestUtil.make(task.resources))
    return requests


def get_actor_skip_reason(
    actor: ActorOnNode, ray_node_id: str, config: UnderutilizedNodeDrainConfig
) -> Optional[str]:
    """Returns why the actor prevents draining its node, None if it doesn't."""
    if actor.max_restarts == 0:
        return f"actor {actor.actor_id} is not restartable (max_restarts=0)"
    if actor.in_placement_group:
        return f"actor {actor.actor_id} is in a placement group"
    if actor.label_selector is None:
        return f"actor {actor.actor_id} has an unsupported label selector"
    for key, operator, values in actor.label_selector:
        if (
            key == NODE_ID_LABEL
            and operator == LabelSelectorOperator.LABEL_OPERATOR_IN
            and ray_node_id in values
        ):
            return f"actor {actor.actor_id} is pinned to the node"
    if actor.num_owned_actors > 0:
        return f"actor {actor.actor_id} owns other actors"
    if config.skip_if_actor_owns_objects and actor.num_owned_objects > 0:
        return f"actor {actor.actor_id} owns objects"
    if actor.namespace in config.excluded_namespaces:
        return f"actor {actor.actor_id} is in excluded namespace {actor.namespace}"
    if actor.job_id in config.excluded_job_ids:
        return f"actor {actor.actor_id} is in excluded job {actor.job_id}"
    return None


def get_workload_skip_reason(
    workload: NodeWorkload, config: UnderutilizedNodeDrainConfig
) -> Optional[str]:
    """Returns why the workload prevents draining its node, None if it doesn't."""
    if not workload.complete:
        return f"stats_incomplete: {workload.incomplete_reason}"
    if workload.has_driver:
        return "driver: a driver is running on the node"
    if workload.num_running_tasks > config.max_running_tasks_per_node:
        return (
            f"too_many_tasks: {workload.num_running_tasks} > "
            f"{config.max_running_tasks_per_node}"
        )
    if len(workload.actors) > config.max_actors_per_node:
        return f"too_many_actors: {len(workload.actors)} > {config.max_actors_per_node}"
    for actor in workload.actors:
        reason = get_actor_skip_reason(actor, workload.ray_node_id, config)
        if reason:
            return f"not_migratable: {reason}"
    return None


@dataclass
class UnderutilizedDrainInput:
    """The underutilized node drain inputs of one scheduling round."""

    config: UnderutilizedNodeDrainConfig
    now_s: float
    # Workloads of the candidate nodes, keyed by hex ray node id. Only set in
    # evaluation rounds.
    candidate_workloads: Dict[str, NodeWorkload] = field(default_factory=dict)
    # Dominant utilization and underutilized duration of the candidates.
    node_utilization: Dict[str, float] = field(default_factory=dict)
    underutilized_duration_ms: Dict[str, int] = field(default_factory=dict)
    # Resource requests reserving capacity for the workloads of the nodes
    # being drained.
    reservation_requests: List[ResourceRequest] = field(default_factory=list)
    # IM instance ids of the nodes being drained.
    draining_instance_ids: Set[str] = field(default_factory=set)
    # The last time an instance was queued to launch (seconds since epoch).
    last_scale_up_s: Optional[float] = None


class UnderutilizationTracker:
    """Tracks since when each ray node has been continuously underutilized."""

    def __init__(self):
        self._since_s: Dict[str, float] = {}

    def update(
        self, utilization: Dict[str, float], threshold: float, now_s: float
    ) -> Dict[str, int]:
        """Updates the tracker with the current utilization of all the nodes.

        Args:
            utilization: Dominant utilization of the tracked nodes. Nodes not
                in it are forgotten.
            threshold: The underutilization threshold.
            now_s: The current time.

        Returns:
            The underutilized duration in ms of each underutilized node.
        """
        durations = {}
        for ray_node_id in list(self._since_s):
            if utilization.get(ray_node_id, 1.0) >= threshold:
                del self._since_s[ray_node_id]
        for ray_node_id, util in utilization.items():
            if util >= threshold:
                continue
            since_s = self._since_s.setdefault(ray_node_id, now_s)
            durations[ray_node_id] = int((now_s - since_s) * 1000)
        return durations


@dataclass
class DrainRecord:
    """A node being drained because it's underutilized."""

    instance_id: str
    ray_node_id: str
    node_type: str
    # The latest workload of the node, used to reserve capacity for it.
    workload: NodeWorkload
    # Unix timestamp in ms after which the instance may be force terminated.
    # None until the drain is issued.
    drain_deadline_ms: Optional[int] = None


class UnderutilizedDrainRegistry:
    """Thread-safe in-memory records of the nodes being drained.

    Like the instance manager's storage, it's lost on autoscaler restarts. The
    reconciler then falls back to `ray_stopping_timeout_s`.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._records: Dict[str, DrainRecord] = {}

    def add(self, record: DrainRecord) -> None:
        with self._lock:
            self._records[record.instance_id] = record

    def get(self, instance_id: str) -> Optional[DrainRecord]:
        with self._lock:
            return self._records.get(instance_id)

    def remove(self, instance_id: str) -> None:
        with self._lock:
            self._records.pop(instance_id, None)

    def set_deadline(self, instance_id: str, deadline_ms: int) -> None:
        with self._lock:
            if instance_id in self._records:
                self._records[instance_id].drain_deadline_ms = deadline_ms

    def update_workload(self, instance_id: str, workload: NodeWorkload) -> None:
        with self._lock:
            if instance_id in self._records:
                self._records[instance_id].workload = workload

    def records(self) -> List[DrainRecord]:
        with self._lock:
            return list(self._records.values())


class NodeWorkloadFetcher:
    """Fetches the actor table from GCS and the per-worker stats from raylets.

    Thread-safe: used by the autoscaler loop and the RayStopper thread.
    """

    _MAX_CACHED_STUBS = 64

    def __init__(self, gcs_client, rpc_timeout_s: float = 10):
        self._gcs_client = gcs_client
        self._rpc_timeout_s = rpc_timeout_s
        self._lock = threading.Lock()
        self._stubs: Dict[str, Any] = {}

    def fetch_alive_actors_by_node(self) -> Dict[str, Dict[str, Any]]:
        """Returns the ALIVE actors keyed by hex node id and hex actor id."""
        gcs_client = self._gcs_client
        timeout_s = self._rpc_timeout_s

        # `async_get_all_actor_info` returns an asyncio.Future created by
        # `asyncio.wrap_future`, which must be called (and awaited) inside a
        # running event loop.
        async def _fetch():
            return await gcs_client.async_get_all_actor_info(
                actor_state_name="ALIVE", timeout=timeout_s
            )

        actors = asyncio.run(_fetch())
        actors_by_node: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for actor_data in actors.values():
            node_id = (
                actor_data.node_id
                if actor_data.HasField("node_id")
                else actor_data.address.node_id
            )
            if not node_id:
                continue
            actors_by_node[binary_to_hex(node_id)][
                binary_to_hex(actor_data.actor_id)
            ] = actor_data
        return actors_by_node

    def fetch_core_worker_stats(
        self, ray_node_ids: List[str]
    ) -> Dict[str, Optional[List[Any]]]:
        """Returns the per-worker stats of the nodes, None for failed nodes."""
        from ray.core.generated.gcs_service_pb2 import GetAllNodeInfoRequest

        results: Dict[str, Optional[List[Any]]] = {i: None for i in ray_node_ids}
        if not ray_node_ids:
            return results
        try:
            node_infos = self._gcs_client.get_all_node_info(
                timeout=self._rpc_timeout_s,
                node_selectors=[
                    GetAllNodeInfoRequest.NodeSelector(node_id=hex_to_binary(i))
                    for i in ray_node_ids
                ],
            )
        except Exception:
            logger.exception("Failed to get node info for %s", ray_node_ids)
            return results

        for node_info in node_infos.values():
            ray_node_id = binary_to_hex(node_info.node_id)
            if ray_node_id not in results:
                continue
            try:
                results[ray_node_id] = self._get_node_stats(
                    node_info.node_manager_address, node_info.node_manager_port
                )
            except Exception as e:
                logger.warning(f"Failed to get node stats of {ray_node_id}: {e}")
        return results

    def _get_node_stats(self, address: str, port: int) -> List[Any]:
        from ray._private.grpc_utils import init_grpc_channel
        from ray._raylet import build_address
        from ray.core.generated import node_manager_pb2, node_manager_pb2_grpc

        raylet_address = build_address(address, port)
        with self._lock:
            stub = self._stubs.get(raylet_address)
            if stub is None:
                if len(self._stubs) >= self._MAX_CACHED_STUBS:
                    # Raylets come and go: don't keep stubs of dead ones forever.
                    self._stubs.clear()
                # The channel carries the auth interceptors when token
                # authentication is enabled.
                channel = init_grpc_channel(
                    raylet_address,
                    options=[
                        ("grpc.max_send_message_length", 512 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
                    ],
                )
                stub = node_manager_pb2_grpc.NodeManagerServiceStub(channel)
                self._stubs[raylet_address] = stub
        reply = stub.GetNodeStats(
            node_manager_pb2.GetNodeStatsRequest(include_memory_info=False),
            timeout=self._rpc_timeout_s,
        )
        return list(reply.core_workers_stats)

    def fetch_node_state(self, ray_node_id: str) -> Optional[NodeState]:
        """Returns the node's state in the cluster resource state, if any."""
        from ray.autoscaler.v2.sdk import get_cluster_resource_state

        state = get_cluster_resource_state(self._gcs_client)
        for node_state in state.node_states:
            if binary_to_hex(node_state.node_id) == ray_node_id:
                return node_state
        return None


def _node_used_resources(node_state: NodeState) -> Dict[str, float]:
    return {
        resource_name: max(
            0.0, total - node_state.available_resources.get(resource_name, 0.0)
        )
        for resource_name, total in node_state.total_resources.items()
    }


def _node_dominant_utilization(node_state: NodeState) -> float:
    return dominant_utilization(
        dict(node_state.total_resources), dict(node_state.available_resources)
    )


class UnderutilizedNodeDrainer:
    """Coordinates the underutilized node drain across reconciliation rounds.

    Each round, the reconciler calls `prepare` to build the scheduler inputs
    and `on_scheduled` with the scheduler's termination requests. The
    RayStopper calls `recheck` and `on_drain_issued` when issuing the drain,
    and the reconciler calls `get_instances_to_terminate` to enforce the drain
    deadlines.
    """

    # Active statuses of an instance being drained.
    _DRAINING_STATUSES = {Instance.RAY_STOP_REQUESTED, Instance.RAY_STOPPING}

    def __init__(
        self,
        fetcher: Optional[NodeWorkloadFetcher],
        clock: Callable[[], float] = time.time,
    ):
        self._fetcher = fetcher
        self._clock = clock
        self._tracker = UnderutilizationTracker()
        self._registry = UnderutilizedDrainRegistry()
        self._config = UnderutilizedNodeDrainConfig()
        self._last_evaluation_s: Optional[float] = None
        # Workloads of the candidates evaluated in the current round, used to
        # create the drain records.
        self._candidate_workloads: Dict[str, NodeWorkload] = {}

    @property
    def registry(self) -> UnderutilizedDrainRegistry:
        return self._registry

    @property
    def config(self) -> UnderutilizedNodeDrainConfig:
        return self._config

    def prepare(
        self,
        config: UnderutilizedNodeDrainConfig,
        ray_state: ClusterResourceState,
        im_instances: List[Instance],
    ) -> UnderutilizedDrainInput:
        """Builds the scheduler inputs of the round.

        Args:
            config: The current config.
            ray_state: The cluster resource state.
            im_instances: The current instance manager instances.

        Returns:
            The inputs for `ResourceDemandScheduler`.
        """
        self._config = config
        now_s = self._clock()
        ray_nodes = {binary_to_hex(n.node_id): n for n in ray_state.node_states}
        instances_by_id = {i.instance_id: i for i in im_instances}

        self._forget_finished_drains(instances_by_id)
        records = self._registry.records()
        drain_input = UnderutilizedDrainInput(
            config=config,
            now_s=now_s,
            draining_instance_ids={r.instance_id for r in records},
            last_scale_up_s=self._get_last_scale_up_s(im_instances),
        )

        candidates: List[str] = []
        if config.enabled:
            candidates = self._get_coarse_candidates(
                config, ray_nodes, im_instances, drain_input, now_s
            )
        else:
            self._tracker = UnderutilizationTracker()

        evaluate = self._last_evaluation_s is None or (
            now_s - self._last_evaluation_s >= config.evaluation_interval_s
        )
        alive_records = [
            r
            for r in records
            if r.ray_node_id in ray_nodes
            and ray_nodes[r.ray_node_id].status != NodeStatus.DEAD
        ]
        self._candidate_workloads = {}
        if evaluate and (candidates or alive_records) and self._fetcher is not None:
            self._last_evaluation_s = now_s
            workloads = self._fetch_workloads(
                candidates + [r.ray_node_id for r in alive_records], ray_nodes, config
            )
            for ray_node_id in candidates:
                self._candidate_workloads[ray_node_id] = workloads[ray_node_id]
            for record in alive_records:
                # Keep the previous workload if the refresh is incomplete, to
                # keep reserving capacity for it.
                workload = workloads.get(record.ray_node_id)
                if workload is not None and workload.complete:
                    self._registry.update_workload(record.instance_id, workload)

        drain_input.candidate_workloads = dict(self._candidate_workloads)
        drain_input.reservation_requests = self._get_reservation_requests(
            alive_records, config, now_s
        )
        return drain_input

    def on_scheduled(self, to_terminate: List[TerminationRequest]) -> None:
        """Records the nodes the scheduler decided to drain.

        It must be called before the instances are transitioned to
        RAY_STOP_REQUESTED, since the RayStopper looks up the records.
        """
        for request in to_terminate:
            if not is_underutilized_drain_request(request):
                continue
            workload = self._candidate_workloads.get(request.ray_node_id)
            if workload is None:
                logger.warning(
                    f"No workload of the node {request.ray_node_id} to drain."
                )
                workload = NodeWorkload(ray_node_id=request.ray_node_id)
            self._registry.add(
                DrainRecord(
                    instance_id=request.instance_id,
                    ray_node_id=request.ray_node_id,
                    node_type=request.instance_type,
                    workload=workload,
                )
            )

    def recheck(self, instance_id: str) -> Tuple[bool, str]:
        """Rechecks a node right before issuing its drain.

        Args:
            instance_id: The IM instance id of the node to drain.

        Returns:
            (ok, reason). ok is False if the node shouldn't be drained anymore.
        """
        record = self._registry.get(instance_id)
        if record is None:
            return False, "no drain record"
        if self._fetcher is None:
            return False, "no workload fetcher"
        config = self._config
        try:
            node_state = self._fetcher.fetch_node_state(record.ray_node_id)
            actors_by_node = self._fetcher.fetch_alive_actors_by_node()
            stats = self._fetcher.fetch_core_worker_stats([record.ray_node_id])
        except Exception as e:
            return False, f"failed to fetch the node workload: {e}"
        if node_state is None or node_state.status != NodeStatus.RUNNING:
            return False, "the node is no longer running"
        utilization = _node_dominant_utilization(node_state)
        if utilization >= config.utilization_threshold:
            return False, f"the node is no longer underutilized: {utilization:.2f}"
        workload = build_node_workload(
            record.ray_node_id,
            stats.get(record.ray_node_id),
            actors_by_node.get(record.ray_node_id, {}),
            _node_used_resources(node_state),
            config.resource_reconcile_tolerance,
            self._clock(),
        )
        reason = get_workload_skip_reason(workload, config)
        if reason:
            return False, reason
        self._registry.update_workload(instance_id, workload)
        return True, ""

    def on_drain_issued(self, instance_id: str, deadline_ms: int) -> None:
        self._registry.set_deadline(instance_id, deadline_ms)

    def forget(self, instance_id: str) -> None:
        self._registry.remove(instance_id)

    def get_drain_deadline_ms(self) -> int:
        return int((self._clock() + self._config.drain_grace_period_s) * 1000)

    def get_instances_to_terminate(
        self, ray_stopping_instances: List[Instance], ray_stopping_since_s: Callable
    ) -> Dict[str, str]:
        """Returns the RAY_STOPPING instances to terminate, with the reasons.

        Args:
            ray_stopping_instances: Instances in RAY_STOPPING.
            ray_stopping_since_s: Returns when an instance entered RAY_STOPPING.

        Returns:
            The reasons to terminate, keyed by the instance ids to terminate.
        """
        now_s = self._clock()
        config = self._config
        to_terminate = {}
        for instance in ray_stopping_instances:
            record = self._registry.get(instance.instance_id)
            if record is not None and record.drain_deadline_ms is not None:
                terminate_at_s = (
                    record.drain_deadline_ms / 1000 + config.termination_buffer_s
                )
                if now_s > terminate_at_s:
                    to_terminate[
                        instance.instance_id
                    ] = "underutilized node drain deadline passed"
            elif record is not None:
                # The drain was accepted but the deadline isn't recorded yet.
                since_s = ray_stopping_since_s(instance)
                terminate_at_s = (
                    since_s + config.drain_grace_period_s + config.termination_buffer_s
                )
                if now_s > terminate_at_s:
                    to_terminate[
                        instance.instance_id
                    ] = "underutilized node drain grace period passed"
            elif config.enabled:
                since_s = ray_stopping_since_s(instance)
                if now_s - since_s > config.ray_stopping_timeout_s:
                    to_terminate[instance.instance_id] = (
                        f"stuck in RAY_STOPPING for more than "
                        f"{config.ray_stopping_timeout_s}s"
                    )
        return to_terminate

    def _forget_finished_drains(self, instances_by_id: Dict[str, Instance]) -> None:
        for record in self._registry.records():
            instance = instances_by_id.get(record.instance_id)
            if instance is None or instance.status not in self._DRAINING_STATUSES:
                # The drain finished, failed (back to RAY_RUNNING), or the
                # instance is gone.
                self._registry.remove(record.instance_id)

    @staticmethod
    def _get_last_scale_up_s(im_instances: List[Instance]) -> Optional[float]:
        last_ns = None
        for instance in im_instances:
            for history in instance.status_history:
                if history.instance_status == Instance.QUEUED:
                    if last_ns is None or history.timestamp_ns > last_ns:
                        last_ns = history.timestamp_ns
        return last_ns / 1e9 if last_ns is not None else None

    def _get_coarse_candidates(
        self,
        config: UnderutilizedNodeDrainConfig,
        ray_nodes: Dict[str, NodeState],
        im_instances: List[Instance],
        drain_input: UnderutilizedDrainInput,
        now_s: float,
    ) -> List[str]:
        """Selects the candidates from the cluster resource state only."""
        utilization = {}
        for ray_node_id, node_state in ray_nodes.items():
            if node_state.status in (NodeStatus.RUNNING, NodeStatus.IDLE):
                utilization[ray_node_id] = _node_dominant_utilization(node_state)
        durations = self._tracker.update(
            utilization, config.utilization_threshold, now_s
        )

        candidates = []
        min_duration_ms = config.underutilized_duration_s * 1000
        for instance in im_instances:
            ray_node_id = instance.node_id
            node_state = ray_nodes.get(ray_node_id)
            if (
                instance.status != Instance.RAY_RUNNING
                or instance.node_kind == NodeKind.HEAD
                or node_state is None
                # Idle nodes are handled by the idle termination.
                or node_state.status != NodeStatus.RUNNING
                or instance.instance_type in config.disabled_node_types
                or durations.get(ray_node_id, -1) < min_duration_ms
            ):
                continue
            if node_state.labels.get(DRAIN_PROTECTED_LABEL, "").lower() == "true":
                continue
            if any(
                label.startswith(PLACEMENT_GROUP_LABEL_PREFIX)
                for label in node_state.dynamic_labels
            ):
                continue
            object_store_used = node_state.total_resources.get(
                "object_store_memory", 0.0
            ) - node_state.available_resources.get("object_store_memory", 0.0)
            if object_store_used > config.max_object_store_used_bytes:
                continue
            candidates.append(ray_node_id)

        candidates.sort(key=lambda i: (utilization[i], -durations[i]))
        candidates = candidates[: config.max_candidates_to_inspect]
        drain_input.node_utilization = {i: utilization[i] for i in candidates}
        drain_input.underutilized_duration_ms = {i: durations[i] for i in candidates}
        return candidates

    def _fetch_workloads(
        self,
        ray_node_ids: List[str],
        ray_nodes: Dict[str, NodeState],
        config: UnderutilizedNodeDrainConfig,
    ) -> Dict[str, NodeWorkload]:
        now_s = self._clock()
        try:
            actors_by_node = self._fetcher.fetch_alive_actors_by_node()
            stats = self._fetcher.fetch_core_worker_stats(ray_node_ids)
        except Exception as e:
            logger.warning(f"Failed to fetch the node workloads: {e}")
            return {
                i: _incomplete(i, f"failed to fetch: {e}", now_s) for i in ray_node_ids
            }
        return {
            ray_node_id: build_node_workload(
                ray_node_id,
                stats.get(ray_node_id),
                actors_by_node.get(ray_node_id, {}),
                _node_used_resources(ray_nodes[ray_node_id]),
                config.resource_reconcile_tolerance,
                now_s,
            )
            for ray_node_id in ray_node_ids
        }

    def _get_reservation_requests(
        self,
        alive_records: List[DrainRecord],
        config: UnderutilizedNodeDrainConfig,
        now_s: float,
    ) -> List[ResourceRequest]:
        """Requests reserving capacity for the workloads of draining nodes.

        If a workload hasn't been refreshed for 2 evaluation intervals, only
        its actors are reserved: they always need to be restarted, while the
        tasks have likely finished.
        """
        requests = []
        for record in alive_records:
            stale = (
                now_s - record.workload.fetched_at_s > 2 * config.evaluation_interval_s
            )
            requests.extend(
                workload_to_resource_requests(record.workload, include_tasks=not stale)
            )
        return requests
