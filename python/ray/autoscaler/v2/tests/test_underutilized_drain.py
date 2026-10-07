import logging
import os
import sys
from queue import Queue
from typing import Dict, List, Optional
from unittest import mock

import pytest
import yaml

from ray._common.test_utils import wait_for_condition
from ray._private.test_utils import get_test_config_path
from ray.autoscaler.v2.event_logger import AutoscalerEventLogger
from ray.autoscaler.v2.instance_manager.config import (
    AutoscalingConfig,
    InstanceReconcileConfig,
    NodeTypeConfig,
)
from ray.autoscaler.v2.instance_manager.instance_manager import InstanceManager
from ray.autoscaler.v2.instance_manager.instance_storage import InstanceStorage
from ray.autoscaler.v2.instance_manager.reconciler import Reconciler
from ray.autoscaler.v2.instance_manager.storage import InMemoryStorage
from ray.autoscaler.v2.instance_manager.subscribers.ray_stopper import (
    RayStopError,
    RayStopper,
)
from ray.autoscaler.v2.scheduler import (
    ResourceDemandScheduler,
    ResourceRequestSource,
    SchedulingNode,
    SchedulingNodeStatus,
    SchedulingRequest,
)
from ray.autoscaler.v2.tests.util import (
    MockEventLogger,
    create_instance,
    make_autoscaler_instance,
)
from ray.autoscaler.v2.underutilized_drain import (
    RAY_DATA_MAP_WORKER_CLASS_NAME_PREFIX,
    RAY_DATA_MAP_WORKER_MODULE,
    UNDERUTILIZED_DRAIN_DETAILS_PREFIX,
    ActorOnNode,
    DrainRecord,
    NodeWorkload,
    NodeWorkloadFetcher,
    RunningTask,
    UnderutilizationTracker,
    UnderutilizedDrainInput,
    UnderutilizedNodeDrainConfig,
    UnderutilizedNodeDrainer,
    build_node_workload,
    dominant_utilization,
    get_workload_growth_reason,
    get_workload_skip_reason,
    is_ray_data_map_worker,
    is_underutilized_drain_request,
    parse_label_selector,
    placement_resources,
    workload_to_resource_requests,
)
from ray.autoscaler.v2.utils import ResourceRequestUtil
from ray.core.generated.autoscaler_pb2 import (
    ClusterResourceState,
    DrainNodeReason,
    NodeState,
    NodeStatus,
)
from ray.core.generated.common_pb2 import (
    CoreWorkerStats,
    LabelSelectorOperator,
    WorkerType,
)
from ray.core.generated.gcs_pb2 import ActorTableData
from ray.core.generated.instance_manager_pb2 import (
    Instance,
    InstanceUpdateEvent,
    NodeKind,
    TerminationRequest,
)

IN = LabelSelectorOperator.LABEL_OPERATOR_IN
NOT_IN = LabelSelectorOperator.LABEL_OPERATOR_NOT_IN
NIL_ACTOR_ID = b"\xff" * 16

event_logger = AutoscalerEventLogger(MockEventLogger(logging.getLogger(__name__)))


def _actor_id(i: int) -> bytes:
    return bytes([i]) * 16


def _worker_stats(
    worker_id: bytes,
    actor_id: bytes = NIL_ACTOR_ID,
    worker_type: int = WorkerType.WORKER,
    num_running_tasks: int = 0,
    used: Optional[Dict[str, float]] = None,
    num_owned_actors: int = 0,
    num_owned_objects: int = 0,
) -> CoreWorkerStats:
    stats = CoreWorkerStats(
        worker_id=worker_id,
        actor_id=actor_id,
        worker_type=worker_type,
        num_running_tasks=num_running_tasks,
        num_owned_actors=num_owned_actors,
        num_owned_objects=num_owned_objects,
    )
    for resource_name, amount in (used or {}).items():
        stats.used_resources[resource_name].resource_slots.add(
            slot=0, allocation=amount
        )
    return stats


def _actor_data(
    actor_id: bytes,
    node_id: bytes = b"n",
    required: Optional[Dict[str, float]] = None,
    max_restarts: int = -1,
    label_selector: Optional[Dict[str, str]] = None,
    placement_group_id: Optional[bytes] = None,
    module_name: str = RAY_DATA_MAP_WORKER_MODULE,
    class_name: str = "MapWorker(Map(fn))",
) -> ActorTableData:
    data = ActorTableData(
        actor_id=actor_id,
        job_id=b"\x01\x00\x00\x00",
        state=ActorTableData.ALIVE,
        max_restarts=max_restarts,
        required_resources=required or {},
        label_selector=label_selector or {},
        ray_namespace="ns",
        node_id=node_id,
    )
    if placement_group_id is not None:
        data.placement_group_id = placement_group_id
    data.class_name = class_name
    data.function_descriptor.python_function_descriptor.module_name = module_name
    data.function_descriptor.python_function_descriptor.class_name = class_name
    return data


def _actor(
    actor_id: str = "a1",
    resources: Optional[Dict[str, float]] = None,
    label_selector=(),
    **kwargs,
) -> ActorOnNode:
    resources = {"CPU": 1} if resources is None else resources
    fields = dict(
        actor_id=actor_id,
        job_id="01000000",
        namespace="ns",
        required_resources=resources,
        placement_resources=placement_resources(resources),
        label_selector=list(label_selector),
        in_placement_group=False,
        max_restarts=-1,
        is_detached=False,
        class_name="MapWorker(Map(fn))",
        is_ray_data_map_worker=True,
    )
    fields.update(kwargs)
    return ActorOnNode(**fields)


def _workload(
    ray_node_id: str,
    actors: Optional[List[ActorOnNode]] = None,
    tasks: Optional[List[RunningTask]] = None,
    **kwargs,
) -> NodeWorkload:
    return NodeWorkload(
        ray_node_id=ray_node_id,
        actors=actors or [],
        tasks=tasks or [],
        complete=True,
        **kwargs,
    )


#########################################################################
# Helpers
#########################################################################


@pytest.mark.parametrize(
    "required,expected",
    [
        # Default actors hold 0 CPU, but need 1 CPU to be scheduled.
        ({}, {"CPU": 1}),
        ({"memory": 100}, {"memory": 100, "CPU": 1}),
        # Actors that specify resources need exactly them.
        ({"CPU": 2}, {"CPU": 2}),
        ({"GPU": 1}, {"GPU": 1}),
        ({"CPU": 0, "GPU": 1}, {"GPU": 1}),
    ],
)
def test_placement_resources(required, expected):
    assert placement_resources(required) == expected


@pytest.mark.parametrize(
    "selector,expected",
    [
        ({"region": "us"}, [("region", IN, ["us"])]),
        ({"region": "!us"}, [("region", NOT_IN, ["us"])]),
        ({"region": "in(us, eu)"}, [("region", IN, ["us", "eu"])]),
        ({"region": "!in(us,eu)"}, [("region", NOT_IN, ["us", "eu"])]),
        ({"region": "(us)"}, [("region", IN, ["us"])]),
        ({}, []),
        ({"region": ""}, None),
        ({"region": "foo(us)"}, None),
        ({"region": "in()"}, None),
    ],
)
def test_parse_label_selector(selector, expected):
    assert parse_label_selector(selector) == expected


def test_dominant_utilization():
    # Object store memory and implicit resources are ignored.
    assert dominant_utilization(
        {"CPU": 4, "GPU": 2, "object_store_memory": 10, "node:1.2.3.4": 1},
        {"CPU": 3, "GPU": 0, "object_store_memory": 0, "node:1.2.3.4": 0},
    ) == pytest.approx(1.0)
    assert dominant_utilization({"CPU": 4}, {"CPU": 3}) == pytest.approx(0.25)
    assert dominant_utilization({}, {}) == 0.0


@pytest.mark.parametrize(
    "module_name,class_name,expected",
    [
        (RAY_DATA_MAP_WORKER_MODULE, "MapWorker(Map(fn))", True),
        (RAY_DATA_MAP_WORKER_MODULE, "_MapWorker", True),
        (RAY_DATA_MAP_WORKER_MODULE, "_ActorPool", False),
        ("ray.serve._private.replica", "MapWorker(x)", False),
        ("__main__", "MapWorker(Map(fn))", False),
    ],
)
def test_is_ray_data_map_worker(module_name, class_name, expected):
    assert is_ray_data_map_worker(module_name, class_name) == expected


def test_ray_data_map_worker_matches_ray_data():
    # The identification and the restart assumptions rely on Ray Data's actor
    # pool implementation.
    from ray.data._internal.execution.operators import actor_pool_map_operator
    from ray.data.context import DataContext

    worker_cls = type(
        actor_pool_map_operator.get_map_worker_cls_name("Map(fn)"),
        (actor_pool_map_operator._MapWorker,),
        {"__module__": actor_pool_map_operator.__name__},
    )
    assert actor_pool_map_operator.__name__ == RAY_DATA_MAP_WORKER_MODULE
    assert worker_cls.__name__.startswith(RAY_DATA_MAP_WORKER_CLASS_NAME_PREFIX)
    assert is_ray_data_map_worker(worker_cls.__module__, worker_cls.__name__)

    remote_args = (
        actor_pool_map_operator.ActorPoolMapOperator._apply_default_remote_args(
            {}, DataContext.get_current()
        )
    )
    assert remote_args["max_restarts"] == -1
    assert remote_args["max_task_retries"] == -1


def test_config_parsing():
    config = UnderutilizedNodeDrainConfig.from_dict(
        {
            "enabled": True,
            "max_concurrent_draining": "10%",
            "excluded_namespaces": ["serve"],
        },
        disabled_node_types={"gpu"},
    )
    assert config.enabled
    assert config.excluded_namespaces == frozenset({"serve"})
    assert config.disabled_node_types == frozenset({"gpu"})
    assert UnderutilizedNodeDrainConfig.from_dict(None) == (
        UnderutilizedNodeDrainConfig()
    )


@pytest.mark.parametrize(
    "value,num_workers,expected",
    [
        (3, 100, 3),
        ("10%", 25, 2),
        ("10%", 5, 1),  # At least 1.
        ("0%", 100, 0),
        ("12.5%", 16, 2),
    ],
)
def test_max_concurrent_draining(value, num_workers, expected):
    config = UnderutilizedNodeDrainConfig.from_dict({"max_concurrent_draining": value})
    assert config.get_max_concurrent_draining(num_workers) == expected


@pytest.mark.parametrize(
    "bad_config",
    [
        {"unknown_key": 1},
        {"enabled": "yes"},
        {"max_concurrent_draining": "abc"},
        {"max_concurrent_draining": "120%"},
        {"max_concurrent_draining": -1},
        {"utilization_threshold": 2},
        {"max_running_tasks_per_node": 1.5},
        {"excluded_namespaces": "serve"},
    ],
)
def test_config_validation(bad_config):
    with pytest.raises(ValueError):
        UnderutilizedNodeDrainConfig.from_dict(bad_config)


def _autoscaling_config(extra: Dict, node_type_extra: Optional[Dict] = None):
    with open(get_test_config_path("test_multi_node.yaml")) as f:
        raw = yaml.safe_load(f)
    raw.update(extra)
    node_type = next(iter(raw["available_node_types"]))
    if node_type_extra:
        raw["available_node_types"][node_type].update(node_type_extra)
    return AutoscalingConfig(raw, skip_content_hash=True), node_type


def test_autoscaling_config():
    config, node_type = _autoscaling_config(
        {
            "underutilized_node_drain": {
                "enabled": True,
                "max_concurrent_draining": "5%",
            }
        },
        {"underutilized_node_drain": {"enabled": False}},
    )
    drain_config = config.get_underutilized_node_drain_config()
    assert drain_config.enabled
    assert drain_config.max_concurrent_draining == "5%"
    assert drain_config.disabled_node_types == frozenset({node_type})

    # Not configured: disabled.
    config, _ = _autoscaling_config({})
    assert not config.get_underutilized_node_drain_config().enabled

    # Invalid configs fail when loading the config.
    with pytest.raises(Exception):
        _autoscaling_config({"underutilized_node_drain": {"max_nodes_per_round": -1}})


def test_tracker():
    tracker = UnderutilizationTracker()
    assert tracker.update({"a": 0.1, "b": 0.5}, 0.3, now_s=100) == {"a": 0}
    assert tracker.update({"a": 0.1, "b": 0.1}, 0.3, now_s=110) == {
        "a": 10000,
        "b": 0,
    }
    # "a" is busy for a moment: its timer restarts.
    assert tracker.update({"a": 0.9, "b": 0.1}, 0.3, now_s=120) == {"b": 10000}
    assert tracker.update({"a": 0.1, "b": 0.1}, 0.3, now_s=130) == {
        "a": 0,
        "b": 20000,
    }
    # Removed nodes are forgotten.
    assert tracker.update({"b": 0.1}, 0.3, now_s=140) == {"b": 30000}


#########################################################################
# Node workload and its completeness validation
#########################################################################


def _complete_node_stats():
    stats = [
        # A task worker running 2 tasks with 1 CPU.
        _worker_stats(b"w1", num_running_tasks=2, used={"CPU": 1}),
        # An idle pooled worker.
        _worker_stats(b"w2"),
        # An actor with 2 CPUs executing 1 method.
        _worker_stats(
            b"w3", actor_id=_actor_id(1), num_running_tasks=1, used={"CPU": 2}
        ),
        # An IO worker.
        _worker_stats(b"w4", worker_type=WorkerType.SPILL_WORKER),
    ]
    actors = {
        _actor_id(1).hex(): _actor_data(
            _actor_id(1), required={"CPU": 2}, label_selector={"zone": "in(a,b)"}
        )
    }
    return stats, actors


def test_build_node_workload():
    stats, actors = _complete_node_stats()
    workload = build_node_workload(
        "n",
        stats,
        actors,
        {"CPU": 3, "object_store_memory": 100, "node:1.2.3.4": 1},
        resource_reconcile_tolerance=0.01,
        fetched_at_s=5,
    )
    assert workload.complete, workload.incomplete_reason
    assert not workload.has_driver
    assert workload.num_running_tasks == 2
    assert workload.num_running_actor_methods == 1
    assert workload.object_store_used_bytes == 100
    assert [t.resources for t in workload.tasks] == [{"CPU": 1}]
    (actor,) = workload.actors
    assert actor.actor_id == _actor_id(1).hex()
    assert actor.placement_resources == {"CPU": 2}
    assert actor.label_selector == [("zone", IN, ["a", "b"])]
    assert actor.max_restarts == -1
    assert actor.is_ray_data_map_worker
    assert actor.class_name == "MapWorker(Map(fn))"

    stats.append(_worker_stats(b"w5", worker_type=WorkerType.DRIVER))
    workload = build_node_workload("n", stats, actors, {"CPU": 3}, 0.01, 5)
    assert workload.complete and workload.has_driver


@pytest.mark.parametrize(
    "case",
    [
        "rpc_failed",
        "worker_rpc_failed",
        "gcs_actor_missing_from_stats",
        "stats_actor_not_alive_in_gcs",
        "unreported_resources",
    ],
)
def test_build_node_workload_incomplete(case):
    stats, actors = _complete_node_stats()
    node_used = {"CPU": 3}
    if case == "rpc_failed":
        stats = None
    elif case == "worker_rpc_failed":
        # GetNodeStats merges an empty CoreWorkerStats for a failed worker RPC.
        stats.append(CoreWorkerStats())
    elif case == "gcs_actor_missing_from_stats":
        actors[_actor_id(2).hex()] = _actor_data(_actor_id(2))
    elif case == "stats_actor_not_alive_in_gcs":
        stats.append(_worker_stats(b"w9", actor_id=_actor_id(3)))
    elif case == "unreported_resources":
        # E.g. a lease is granted but its worker isn't reported yet.
        node_used = {"CPU": 4}

    workload = build_node_workload("n", stats, actors, node_used, 0.01, 5)
    assert not workload.complete
    assert workload.incomplete_reason
    assert get_workload_skip_reason(
        workload, UnderutilizedNodeDrainConfig()
    ).startswith("stats_incomplete")


def test_nil_actor_id_is_not_an_actor():
    # A non-actor worker reports ActorID::Nil(), not an empty actor id.
    workload = build_node_workload(
        "n",
        [_worker_stats(b"w1", num_running_tasks=1, used={"CPU": 1})],
        {},
        {"CPU": 1},
        0.01,
        5,
    )
    assert workload.complete
    assert workload.actors == []
    assert workload.num_running_tasks == 1


@pytest.mark.parametrize(
    "actor_kwargs,config_kwargs,expected_prefix",
    [
        ({}, {}, None),
        ({"is_ray_data_map_worker": False}, {}, "not_migratable"),
        ({"max_restarts": 0}, {}, "not_migratable"),
        ({"in_placement_group": True}, {}, "not_migratable"),
        ({"label_selector": None}, {}, "not_migratable"),
        ({"label_selector": [("ray.io/node-id", IN, ["n"])]}, {}, "not_migratable"),
        ({"label_selector": [("ray.io/node-id", NOT_IN, ["n"])]}, {}, None),
        ({"num_owned_actors": 1}, {}, "not_migratable"),
        ({"num_owned_objects": 1}, {}, "not_migratable"),
        (
            {"num_owned_objects": 1},
            {"skip_if_actor_owns_objects": False},
            None,
        ),
        ({"namespace": "serve"}, {"excluded_namespaces": ["serve"]}, "not_migratable"),
        ({}, {"max_actors_per_node": 0}, "too_many_actors"),
    ],
)
def test_workload_skip_reason(actor_kwargs, config_kwargs, expected_prefix):
    actor_kwargs = dict(actor_kwargs)
    label_selector = actor_kwargs.pop("label_selector", [])
    actor = _actor(**actor_kwargs)
    actor.label_selector = label_selector
    config = UnderutilizedNodeDrainConfig.from_dict(config_kwargs)
    reason = get_workload_skip_reason(_workload("n", actors=[actor]), config)
    if expected_prefix is None:
        assert reason is None
    else:
        assert reason.startswith(expected_prefix)


def test_too_many_tasks_and_driver():
    config = UnderutilizedNodeDrainConfig.from_dict({"max_running_tasks_per_node": 1})
    tasks = [RunningTask("w1", 1, {"CPU": 1}), RunningTask("w2", 1, {"CPU": 1})]
    assert get_workload_skip_reason(_workload("n", tasks=tasks), config).startswith(
        "too_many_tasks"
    )
    assert get_workload_skip_reason(_workload("n", has_driver=True), config).startswith(
        "driver"
    )


def test_workload_to_resource_requests():
    workload = _workload(
        "n",
        actors=[
            _actor("a1", {}),
            _actor("a2", {"GPU": 1}, label_selector=[("zone", IN, ["a"])]),
        ],
        tasks=[RunningTask("w1", 1, {"CPU": 0.5}), RunningTask("w2", 1, {})],
    )
    requests = workload_to_resource_requests(workload)
    assert [dict(r.resources_bundle) for r in requests] == [
        {"CPU": 1},
        {"GPU": 1},
        {"CPU": 0.5},
    ]
    assert requests[1].label_selectors[0].label_constraints[0].label_key == "zone"
    assert [
        dict(r.resources_bundle)
        for r in workload_to_resource_requests(workload, include_tasks=False)
    ] == [{"CPU": 1}, {"GPU": 1}]


#########################################################################
# Scheduler: node selection and feasibility simulation
#########################################################################

NODE_TYPE_CONFIGS = {
    "cpu": NodeTypeConfig(
        name="cpu", resources={"CPU": 4}, min_worker_nodes=0, max_worker_nodes=10
    ),
    "gpu": NodeTypeConfig(
        name="gpu", resources={"GPU": 1}, min_worker_nodes=0, max_worker_nodes=10
    ),
    "head": NodeTypeConfig(
        name="head", resources={}, min_worker_nodes=0, max_worker_nodes=1
    ),
}


def _instance(
    ray_node_id: str,
    available_cpu: float,
    total_cpu: float = 4,
    status: int = Instance.RAY_RUNNING,
    node_type: str = "cpu",
    node_kind: int = NodeKind.WORKER,
    labels: Optional[Dict[str, str]] = None,
    idle_duration_ms: int = 0,
):
    ray_status = NodeStatus.IDLE if idle_duration_ms else NodeStatus.RUNNING
    return make_autoscaler_instance(
        im_instance=Instance(
            instance_id=f"i-{ray_node_id}",
            instance_type=node_type,
            status=status,
            node_id=ray_node_id,
            node_kind=node_kind,
        ),
        ray_node=NodeState(
            node_id=ray_node_id.encode(),
            ray_node_type_name=node_type,
            total_resources={"CPU": total_cpu},
            available_resources={"CPU": available_cpu},
            status=ray_status,
            idle_duration_ms=idle_duration_ms,
            labels=labels or {},
        ),
        cloud_instance_id=f"c-{ray_node_id}",
    )


def _head():
    return _instance("head", 0, total_cpu=0, node_type="head", node_kind=NodeKind.HEAD)


def _drain_input(
    workloads: Dict[str, NodeWorkload],
    utilization: Optional[Dict[str, float]] = None,
    **config_kwargs,
) -> UnderutilizedDrainInput:
    config = dict(enabled=True)
    config.update(config_kwargs)
    return UnderutilizedDrainInput(
        config=UnderutilizedNodeDrainConfig.from_dict(config),
        now_s=10000,
        candidate_workloads=workloads,
        node_utilization=utilization or {i: 0.25 for i in workloads},
        underutilized_duration_ms={i: 600000 for i in workloads},
    )


def _schedule(instances, drain_input, resource_requests=None, idle_timeout_s=None):
    return ResourceDemandScheduler(event_logger).schedule(
        SchedulingRequest(
            node_type_configs=NODE_TYPE_CONFIGS,
            disable_launch_config_check=True,
            current_instances=instances,
            resource_requests=ResourceRequestUtil.group_by_count(
                resource_requests or []
            ),
            idle_timeout_s=idle_timeout_s,
            underutilized_drain=drain_input,
        )
    )


def _drained(reply) -> List[str]:
    return [
        r.ray_node_id for r in reply.to_terminate if is_underutilized_drain_request(r)
    ]


def test_drain_selects_node_with_least_running_work():
    instances = [
        _head(),
        _instance("r-1", available_cpu=3),
        _instance("r-2", available_cpu=3),
        _instance("r-3", available_cpu=2),
    ]
    workloads = {
        # 1 actor, no running method.
        "r-1": _workload("r-1", actors=[_actor("a1", {"CPU": 1})]),
        # 2 running tasks.
        "r-2": _workload(
            "r-2",
            tasks=[
                RunningTask("w1", 1, {"CPU": 0.5}),
                RunningTask("w2", 1, {"CPU": 0.5}),
            ],
        ),
    }
    reply = _schedule(instances, _drain_input(workloads))
    assert _drained(reply) == ["r-1"]
    (request,) = reply.to_terminate
    assert request.cause == TerminationRequest.Cause.UNKNOWN
    assert request.details.startswith(UNDERUTILIZED_DRAIN_DETAILS_PREFIX)
    assert request.instance_id == "i-r-1"
    assert request.instance_status == Instance.RAY_RUNNING


def test_drain_infeasible():
    instances = [
        _head(),
        _instance("r-1", available_cpu=2),
        # Only 0.6 CPU left below the 0.9 utilization cap.
        _instance("r-2", available_cpu=1),
    ]
    workloads = {"r-1": _workload("r-1", actors=[_actor("a1", {"CPU": 1})])}
    reply = _schedule(instances, _drain_input(workloads))
    assert _drained(reply) == []


@pytest.mark.parametrize("cap,drained", [(0.9, False), (1.0, True)])
def test_drain_utilization_cap(cap, drained):
    instances = [
        _head(),
        _instance("r-1", available_cpu=2),
        _instance("r-2", available_cpu=2.2),
    ]
    workloads = {"r-1": _workload("r-1", actors=[_actor("a1", {"CPU": 2})])}
    reply = _schedule(
        instances, _drain_input(workloads, post_drain_max_utilization=cap)
    )
    assert _drained(reply) == (["r-1"] if drained else [])


def test_over_cap_host_does_not_veto_drain():
    instances = [
        _head(),
        _instance("r-1", available_cpu=3),
        # Fully used, above the cap, and doesn't receive anything.
        _instance("r-2", available_cpu=0),
        _instance("r-3", available_cpu=4),
    ]
    workloads = {"r-1": _workload("r-1", actors=[_actor("a1", {"CPU": 1})])}
    reply = _schedule(instances, _drain_input(workloads))
    assert _drained(reply) == ["r-1"]


def test_default_actor_needs_one_cpu_to_restart():
    # The actor holds 0 CPU, but needs 1 CPU to be restarted.
    instances = [
        _head(),
        _instance("r-1", available_cpu=4),
        _instance("r-2", available_cpu=0.5, total_cpu=1),
    ]
    workloads = {"r-1": _workload("r-1", actors=[_actor("a1", {})])}
    reply = _schedule(
        instances, _drain_input(workloads, post_drain_max_utilization=1.0)
    )
    assert _drained(reply) == []


def test_no_drain_while_scaling_up():
    instances = [
        _head(),
        _instance("r-1", available_cpu=3),
        _instance("r-2", available_cpu=4),
    ]
    workloads = {"r-1": _workload("r-1", actors=[_actor("a1", {"CPU": 1})])}
    reply = _schedule(
        instances,
        _drain_input(workloads),
        resource_requests=[ResourceRequestUtil.make({"GPU": 1})],
    )
    assert reply.to_launch
    assert _drained(reply) == []


def test_no_drain_during_cooldown():
    instances = [
        _head(),
        _instance("r-1", available_cpu=3),
        _instance("r-2", available_cpu=4),
    ]
    workloads = {"r-1": _workload("r-1", actors=[_actor("a1", {"CPU": 1})])}
    drain_input = _drain_input(workloads, cooldown_after_scale_up_s=600)
    drain_input.last_scale_up_s = drain_input.now_s - 10
    assert _drained(_schedule(instances, drain_input)) == []
    drain_input.last_scale_up_s = drain_input.now_s - 700
    assert _drained(_schedule(instances, drain_input)) == ["r-1"]


def test_dry_run():
    instances = [
        _head(),
        _instance("r-1", available_cpu=3),
        _instance("r-2", available_cpu=4),
    ]
    workloads = {"r-1": _workload("r-1", actors=[_actor("a1", {"CPU": 1})])}
    reply = _schedule(instances, _drain_input(workloads, dry_run=True))
    assert reply.to_terminate == []


def test_respect_min_worker_nodes():
    node_type_configs = dict(NODE_TYPE_CONFIGS)
    node_type_configs["cpu"] = NodeTypeConfig(
        name="cpu", resources={"CPU": 4}, min_worker_nodes=2, max_worker_nodes=10
    )
    instances = [
        _head(),
        _instance("r-1", available_cpu=3),
        _instance("r-2", available_cpu=4),
    ]
    workloads = {"r-1": _workload("r-1", actors=[_actor("a1", {"CPU": 1})])}
    with mock.patch.dict(NODE_TYPE_CONFIGS, node_type_configs):
        reply = _schedule(instances, _drain_input(workloads))
    assert _drained(reply) == []


def test_skip_node_needed_by_demand():
    node = SchedulingNode(
        node_type="cpu",
        total_resources={"CPU": 4},
        available_resources={"CPU": 3},
        labels={},
        status=SchedulingNodeStatus.SCHEDULABLE,
        im_instance_status=Instance.RAY_RUNNING,
        ray_node_id="r-1",
    )
    drain_input = _drain_input({"r-1": _workload("r-1")})
    assert (
        ResourceDemandScheduler._get_underutilized_drain_skip_reason(
            node, drain_input.candidate_workloads["r-1"], drain_input
        )
        is None
    )
    node.add_sched_request(
        ResourceRequestUtil.make({"CPU": 1}), ResourceRequestSource.PENDING_DEMAND
    )
    assert ResourceDemandScheduler._get_underutilized_drain_skip_reason(
        node, drain_input.candidate_workloads["r-1"], drain_input
    ).startswith("needed_by_demand")


def test_capacity_is_accumulated_across_candidates():
    # Both candidates' actors can only run on r-3, which fits only one of them.
    selector = [("pool", IN, ["big"])]
    instances = [
        _head(),
        _instance("r-1", available_cpu=2),
        _instance("r-2", available_cpu=2),
        _instance("r-3", available_cpu=4, labels={"pool": "big"}),
    ]
    workloads = {
        "r-1": _workload(
            "r-1", actors=[_actor("a1", {"CPU": 2}, label_selector=selector)]
        ),
        "r-2": _workload(
            "r-2", actors=[_actor("a2", {"CPU": 2}, label_selector=selector)]
        ),
    }
    reply = _schedule(
        instances,
        _drain_input(
            workloads,
            utilization={"r-1": 0.1, "r-2": 0.2},
            max_nodes_per_round=2,
            max_concurrent_draining=2,
        ),
    )
    assert _drained(reply) == ["r-1"]


def test_receiving_node_is_not_drained_in_same_round():
    instances = [
        _head(),
        _instance("r-1", available_cpu=2),
        _instance("r-2", available_cpu=2),
    ]
    workloads = {
        "r-1": _workload("r-1", actors=[_actor("a1", {"CPU": 1})]),
        "r-2": _workload("r-2", actors=[_actor("a2", {"CPU": 1})]),
    }
    reply = _schedule(
        instances,
        _drain_input(workloads, max_nodes_per_round=2, max_concurrent_draining=2),
    )
    assert len(_drained(reply)) == 1


@pytest.mark.parametrize(
    "max_concurrent,num_draining,expected",
    # The percentage applies to the 4 running workers plus the draining ones.
    [(1, 0, 1), (1, 1, 0), ("50%", 2, 1), ("50%", 3, 0)],
)
def test_max_concurrent_draining_quota(max_concurrent, num_draining, expected):
    instances = [_head()] + [
        _instance(f"r-{i}", available_cpu=4 if i else 3) for i in range(4)
    ]
    workloads = {"r-0": _workload("r-0", actors=[_actor("a1", {"CPU": 1})])}
    drain_input = _drain_input(workloads, max_concurrent_draining=max_concurrent)
    # Draining nodes are counted from the instance manager, including those
    # drained for other reasons or before an autoscaler restart.
    drain_input.num_draining_nodes = num_draining
    drain_input.num_worker_nodes = 4 + num_draining
    assert len(_drained(_schedule(instances, drain_input))) == expected


def test_reservations_of_draining_nodes():
    instances = [
        _head(),
        # Idle, would be terminated without the reservation.
        _instance("r-1", available_cpu=4, idle_duration_ms=1000000),
        # Being drained: must not be modeled as an empty node to host the
        # reservation.
        _instance("r-9", available_cpu=3, status=Instance.RAY_STOP_REQUESTED),
    ]
    reservation = ResourceRequestUtil.make({"CPU": 1})
    infeasible_reservation = ResourceRequestUtil.make({"CPU": 100})
    drain_input = _drain_input({})
    drain_input.draining_instance_ids = {"i-r-9"}
    drain_input.reservation_requests = [reservation, infeasible_reservation]
    reply = _schedule(instances, drain_input, idle_timeout_s=1)
    assert reply.to_terminate == []
    # Reservations are not reported as infeasible demands.
    assert reply.infeasible_resource_requests == []

    # Without the reservation, the idle node is terminated.
    drain_input.reservation_requests = []
    reply = _schedule(instances, drain_input, idle_timeout_s=1)
    assert [r.ray_node_id for r in reply.to_terminate] == ["r-1"]


def test_draining_nodes_count_towards_max_workers():
    node_type_configs = dict(NODE_TYPE_CONFIGS)
    node_type_configs["cpu"] = NodeTypeConfig(
        name="cpu", resources={"CPU": 4}, min_worker_nodes=0, max_worker_nodes=2
    )
    instances = [
        _head(),
        _instance("r-1", available_cpu=0),
        _instance("r-9", available_cpu=3, status=Instance.RAY_STOPPING),
    ]
    drain_input = _drain_input({})
    drain_input.draining_instance_ids = {"i-r-9"}
    with mock.patch.dict(NODE_TYPE_CONFIGS, node_type_configs):
        reply = _schedule(
            instances,
            drain_input,
            resource_requests=[ResourceRequestUtil.make({"CPU": 4})],
        )
        # The draining node still counts: no room to launch another node.
        assert reply.to_launch == []
        assert reply.to_terminate == []

        drain_input.draining_instance_ids = set()
        reply = _schedule(
            [_head(), _instance("r-1", available_cpu=0)],
            drain_input,
            resource_requests=[ResourceRequestUtil.make({"CPU": 4})],
        )
        assert [(r.instance_type, r.count) for r in reply.to_launch] == [("cpu", 1)]


#########################################################################
# UnderutilizedNodeDrainer
#########################################################################


class FakeClock:
    def __init__(self, now_s: float = 1000):
        self.now_s = now_s

    def __call__(self) -> float:
        return self.now_s


class FakeFetcher:
    def __init__(self):
        self.actors_by_node: Dict[str, Dict] = {}
        self.stats: Dict[str, Optional[List[CoreWorkerStats]]] = {}
        self.node_states: Dict[str, NodeState] = {}
        self.draining_deadlines: Dict[str, int] = {}
        self.fetched_node_ids: List[List[str]] = []
        self.rpc_timeout_s = None

    def set_rpc_timeout(self, rpc_timeout_s):
        self.rpc_timeout_s = rpc_timeout_s

    def fetch_draining_deadlines(self):
        return self.draining_deadlines

    def fetch_alive_actors_by_node(self):
        return self.actors_by_node

    def fetch_core_worker_stats(self, ray_node_ids):
        self.fetched_node_ids.append(list(ray_node_ids))
        return {i: self.stats.get(i) for i in ray_node_ids}

    def fetch_node_state(self, ray_node_id):
        return self.node_states.get(ray_node_id)


def _node_state(
    ray_node_id: str,
    available_cpu: float,
    status=NodeStatus.RUNNING,
    labels=None,
    dynamic_labels=None,
) -> NodeState:
    return NodeState(
        node_id=bytes.fromhex(ray_node_id),
        total_resources={"CPU": 4},
        available_resources={"CPU": available_cpu},
        status=status,
        labels=labels or {},
        dynamic_labels=dynamic_labels or {},
    )


R1, R2, R3, R4, R5 = ("aa" * 28, "bb" * 28, "cc" * 28, "dd" * 28, "ee" * 28)


def _drainer_setup():
    clock = FakeClock(1000)
    fetcher = FakeFetcher()
    drainer = UnderutilizedNodeDrainer(fetcher, clock=clock)
    config = UnderutilizedNodeDrainConfig.from_dict(
        {
            "enabled": True,
            "underutilized_duration_s": 100,
            "evaluation_interval_s": 60,
        }
    )
    ray_state = ClusterResourceState(
        node_states=[
            _node_state(R1, 3),  # Underutilized.
            _node_state(R2, 0.5),  # Busy.
            _node_state(R3, 4, status=NodeStatus.IDLE),  # Idle termination's job.
            _node_state(R4, 3, dynamic_labels={"_PG_abc": ""}),  # Has a PG.
            _node_state(R5, 3, labels={"ray.io/drain-protected": "true"}),
        ]
    )
    im_instances = [
        create_instance(
            f"i-{i}",
            status=Instance.RAY_RUNNING,
            ray_node_id=node_id,
            instance_type="cpu",
            status_times=[(Instance.QUEUED, 0), (Instance.RAY_RUNNING, 1)],
        )
        for i, node_id in enumerate([R1, R2, R3, R4, R5], start=1)
    ]
    fetcher.stats[R1] = [_worker_stats(b"w1", num_running_tasks=1, used={"CPU": 1})]
    return clock, fetcher, drainer, config, ray_state, im_instances


def test_drainer_prepare_candidates():
    clock, fetcher, drainer, config, ray_state, im_instances = _drainer_setup()

    # Not underutilized for long enough yet.
    drain_input = drainer.prepare(config, ray_state, im_instances)
    assert drain_input.candidate_workloads == {}
    assert fetcher.fetched_node_ids == []

    clock.now_s += 200
    drain_input = drainer.prepare(config, ray_state, im_instances)
    assert list(drain_input.candidate_workloads) == [R1]
    assert drain_input.candidate_workloads[R1].complete
    assert drain_input.underutilized_duration_ms == {R1: 200000}
    assert drain_input.node_utilization == {R1: pytest.approx(0.25)}
    assert fetcher.fetched_node_ids == [[R1]]

    # Not evaluated again before the evaluation interval.
    clock.now_s += 30
    drain_input = drainer.prepare(config, ray_state, im_instances)
    assert drain_input.candidate_workloads == {}
    assert len(fetcher.fetched_node_ids) == 1

    # The config is applied to the fetcher.
    assert fetcher.rpc_timeout_s == config.rpc_timeout_s

    # Disabled without any drain in progress: nothing is computed.
    clock.now_s += 100
    disabled = UnderutilizedNodeDrainConfig.from_dict(
        {"enabled": False, "rpc_timeout_s": 3}
    )
    drain_input = drainer.prepare(disabled, ray_state, im_instances)
    assert drain_input.candidate_workloads == {}
    assert drain_input.last_scale_up_s is None
    assert fetcher.rpc_timeout_s == 3


def test_drainer_counts_draining_nodes():
    clock, fetcher, drainer, config, ray_state, im_instances = _drainer_setup()
    # A node drained for another reason (or by this feature before an
    # autoscaler restart) has no record, but still counts.
    im_instances[1].status = Instance.RAY_STOPPING
    im_instances[2].status = Instance.RAY_STOP_REQUESTED
    im_instances.append(
        create_instance("head", status=Instance.RAY_RUNNING, node_kind=NodeKind.HEAD)
    )
    drain_input = drainer.prepare(config, ray_state, im_instances)
    assert drain_input.num_draining_nodes == 2
    assert drain_input.num_worker_nodes == 5
    assert drain_input.draining_instance_ids == set()


def _drain_request(instance_id: str, ray_node_id: str) -> TerminationRequest:
    return TerminationRequest(
        id="t",
        instance_id=instance_id,
        ray_node_id=ray_node_id,
        cause=TerminationRequest.Cause.UNKNOWN,
        instance_type="cpu",
        details=f"{UNDERUTILIZED_DRAIN_DETAILS_PREFIX}: test",
    )


def test_drainer_records_and_reservations():
    clock, fetcher, drainer, config, ray_state, im_instances = _drainer_setup()
    drainer.prepare(config, ray_state, im_instances)
    clock.now_s += 200
    drainer.prepare(config, ray_state, im_instances)
    drainer.on_scheduled(
        [
            _drain_request("i-1", R1),
            # Not an underutilized drain.
            TerminationRequest(instance_id="i-2", cause=TerminationRequest.Cause.IDLE),
        ]
    )
    assert [r.instance_id for r in drainer.registry.records()] == ["i-1"]

    # The instance is being drained: reserve capacity for its running task.
    im_instances[0].status = Instance.RAY_STOP_REQUESTED
    clock.now_s += 10
    drain_input = drainer.prepare(config, ray_state, im_instances)
    assert drain_input.draining_instance_ids == {"i-1"}
    assert [dict(r.resources_bundle) for r in drain_input.reservation_requests] == [
        {"CPU": 1}
    ]

    # The refresh fails: keep the previous workload, but only reserve its
    # actors (none here) once it's stale.
    fetcher.stats[R1] = None
    clock.now_s += 200
    drain_input = drainer.prepare(config, ray_state, im_instances)
    assert drain_input.reservation_requests == []

    # A successful refresh replaces the workload.
    fetcher.stats[R1] = [
        _worker_stats(b"w1", num_running_tasks=1, used={"CPU": 0.5}),
    ]
    ray_state.node_states[0].available_resources["CPU"] = 3.5
    clock.now_s += 100
    drain_input = drainer.prepare(config, ray_state, im_instances)
    assert [dict(r.resources_bundle) for r in drain_input.reservation_requests] == [
        {"CPU": 0.5}
    ]

    # The drain failed: the instance is back to RAY_RUNNING.
    im_instances[0].status = Instance.RAY_RUNNING
    drain_input = drainer.prepare(config, ray_state, im_instances)
    assert drainer.registry.records() == []
    assert drain_input.reservation_requests == []


def _select_r1_to_drain(clock, drainer, config, ray_state, im_instances):
    # Start tracking the underutilization, then evaluate R1 as a candidate.
    drainer.prepare(config, ray_state, im_instances)
    clock.now_s += 200
    drain_input = drainer.prepare(config, ray_state, im_instances)
    assert drain_input.candidate_workloads[R1].complete
    drainer.on_scheduled([_drain_request("i-1", R1)])


def test_drainer_recheck():
    clock, fetcher, drainer, config, ray_state, im_instances = _drainer_setup()
    _select_r1_to_drain(clock, drainer, config, ray_state, im_instances)
    fetcher.node_states[R1] = _node_state(R1, 3)
    assert drainer.recheck("i-1") == (True, "")

    # The node got busy.
    fetcher.node_states[R1] = _node_state(R1, 0)
    ok, reason = drainer.recheck("i-1")
    assert not ok and "no longer underutilized" in reason

    # A driver started on the node.
    fetcher.node_states[R1] = _node_state(R1, 3)
    fetcher.stats[R1].append(_worker_stats(b"d", worker_type=WorkerType.DRIVER))
    ok, reason = drainer.recheck("i-1")
    assert not ok and reason.startswith("driver")

    assert drainer.recheck("unknown") == (False, "no drain record")


@pytest.mark.parametrize("change", ["new_actor", "more_resources", "pg", "none"])
def test_drainer_recheck_workload_changed(change):
    clock, fetcher, drainer, config, ray_state, im_instances = _drainer_setup()
    _select_r1_to_drain(clock, drainer, config, ray_state, im_instances)
    fetcher.node_states[R1] = _node_state(R1, 3)
    if change == "new_actor":
        # A new map worker landed on the node after the simulation.
        fetcher.stats[R1].append(
            _worker_stats(b"w2", actor_id=_actor_id(7), used={"CPU": 0.5})
        )
        fetcher.actors_by_node[R1] = {
            _actor_id(7).hex(): _actor_data(_actor_id(7), required={"CPU": 0.5})
        }
    elif change == "more_resources":
        fetcher.stats[R1] = [
            _worker_stats(b"w1", num_running_tasks=1, used={"CPU": 1.5})
        ]
    elif change == "pg":
        fetcher.node_states[R1] = _node_state(R1, 3, dynamic_labels={"_PG_x": ""})

    ok, reason = drainer.recheck("i-1")
    if change == "none":
        assert ok, reason
    else:
        assert not ok
        assert reason.startswith("pg" if change == "pg" else "workload_changed")


def test_workload_growth_reason():
    before = _workload(
        "n", actors=[_actor("a1")], tasks=[RunningTask("w", 1, {"CPU": 1})]
    )
    # Tasks churn, but the total doesn't grow.
    after = _workload(
        "n",
        actors=[_actor("a1")],
        tasks=[RunningTask("x", 1, {"CPU": 0.5}), RunningTask("y", 1, {"CPU": 0.5})],
    )
    assert get_workload_growth_reason(before, after, 0.01) is None
    after.tasks.append(RunningTask("z", 1, {"GPU": 1}))
    assert get_workload_growth_reason(before, after, 0.01).startswith(
        "workload_changed"
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_drainer_instances_to_terminate(enabled):
    clock = FakeClock(1000)
    fetcher = FakeFetcher()
    drainer = UnderutilizedNodeDrainer(fetcher, clock=clock)
    drainer.update_config(
        UnderutilizedNodeDrainConfig.from_dict(
            {
                "enabled": enabled,
                "termination_buffer_s": 30,
                "drain_grace_period_s": 300,
            }
        )
    )
    instances = {
        # Drained by this feature, deadline recorded.
        "i-1": create_instance("i-1", status=Instance.RAY_STOPPING, ray_node_id="n1"),
        # Drained by this feature, deadline not recorded: read from GCS.
        "i-2": create_instance("i-2", status=Instance.RAY_STOPPING, ray_node_id="n2"),
        # Drained by this feature, no deadline anywhere: grace period.
        "i-3": create_instance("i-3", status=Instance.RAY_STOPPING, ray_node_id="n3"),
        # No record (other drain, or lost on restart) with a deadline.
        "i-4": create_instance("i-4", status=Instance.RAY_STOPPING, ray_node_id="n4"),
        # No record, without a deadline (e.g. an idle termination).
        "i-5": create_instance("i-5", status=Instance.RAY_STOPPING, ray_node_id="n5"),
    }
    for instance_id in ("i-1", "i-2", "i-3"):
        drainer.registry.add(
            DrainRecord(
                instance_id=instance_id,
                ray_node_id=instances[instance_id].node_id,
                node_type="cpu",
                workload=NodeWorkload(ray_node_id=instances[instance_id].node_id),
            )
        )
    drainer.on_drain_issued("i-1", deadline_ms=1100 * 1000)
    fetcher.draining_deadlines = {"n2": 1200 * 1000, "n4": 1300 * 1000, "n5": 0}

    def terminate_at(now_s):
        clock.now_s = now_s
        return set(
            drainer.get_instances_to_terminate(
                list(instances.values()), lambda instance: 900
            )
        )

    assert terminate_at(1100) == set()
    assert terminate_at(1131) == {"i-1"}
    assert terminate_at(1231) == {"i-1", "i-2", "i-3"}
    # Drains not issued by this feature are only forced when it's enabled, and
    # never before their deadline.
    others = {"i-4"} if enabled else set()
    assert terminate_at(1331) == {"i-1", "i-2", "i-3"} | others
    assert terminate_at(100000) == {"i-1", "i-2", "i-3"} | others


def test_fetch_core_worker_stats_in_parallel():
    gcs_client = mock.MagicMock()
    gcs_client.get_all_node_info.return_value = {
        node_id: mock.MagicMock(
            node_id=bytes.fromhex(node_id),
            node_manager_address="1.2.3.4",
            node_manager_port=port,
        )
        for port, node_id in enumerate([R1, R2, R3])
    }
    fetcher = NodeWorkloadFetcher(gcs_client, rpc_timeout_s=1)

    def get_node_stats(address, port):
        if port == 1:
            raise TimeoutError("timeout")
        return [_worker_stats(b"w")]

    with mock.patch.object(fetcher, "_get_node_stats", side_effect=get_node_stats):
        stats = fetcher.fetch_core_worker_stats([R1, R2, R3, R4])
    assert len(stats[R1]) == 1
    assert stats[R2] is None  # Failed.
    assert len(stats[R3]) == 1
    assert stats[R4] is None  # Unknown node.


#########################################################################
# RayStopper
#########################################################################


def _ray_stop_requested_event(termination_request: TerminationRequest):
    return InstanceUpdateEvent(
        instance_id=termination_request.instance_id,
        new_instance_status=Instance.RAY_STOP_REQUESTED,
        termination_request=termination_request,
    )


def _drainer_with_record(clock):
    fetcher = FakeFetcher()
    fetcher.node_states[R1] = _node_state(R1, 3)
    fetcher.stats[R1] = [_worker_stats(b"w1", num_running_tasks=1, used={"CPU": 1})]
    drainer = UnderutilizedNodeDrainer(fetcher, clock=clock)
    drainer.prepare(
        UnderutilizedNodeDrainConfig.from_dict(
            {"enabled": True, "drain_grace_period_s": 300}
        ),
        ClusterResourceState(),
        [],
    )
    drainer.registry.add(
        DrainRecord(
            instance_id="i-1",
            ray_node_id=R1,
            node_type="cpu",
            # The workload when the node was selected.
            workload=build_node_workload(R1, fetcher.stats[R1], {}, {}, 0.01, 1000),
        )
    )
    return fetcher, drainer


def test_ray_stopper_drains_underutilized_node():
    clock = FakeClock(1000)
    _, drainer = _drainer_with_record(clock)
    gcs_client = mock.MagicMock()
    gcs_client.drain_node.return_value = (True, "")
    error_queue = Queue()
    stopper = RayStopper(gcs_client, error_queue, underutilized_drainer=drainer)

    stopper.notify([_ray_stop_requested_event(_drain_request("i-1", R1))])

    wait_for_condition(lambda: gcs_client.drain_node.call_count == 1)
    kwargs = gcs_client.drain_node.call_args.kwargs
    assert kwargs["node_id"] == R1
    assert kwargs["reason"] == DrainNodeReason.DRAIN_NODE_REASON_PREEMPTION
    assert kwargs["deadline_timestamp_ms"] == (1000 + 300) * 1000
    wait_for_condition(
        lambda: drainer.registry.get("i-1").drain_deadline_ms == 1300 * 1000
    )
    assert gcs_client.drain_nodes.call_count == 0
    assert error_queue.empty()


@pytest.mark.parametrize("case", ["recheck_failed", "no_drainer", "drain_failed"])
def test_ray_stopper_underutilized_drain_failures(case):
    clock = FakeClock(1000)
    fetcher, drainer = _drainer_with_record(clock)
    gcs_client = mock.MagicMock()
    gcs_client.drain_node.return_value = (True, "")
    if case == "recheck_failed":
        fetcher.node_states[R1] = _node_state(R1, 0)
    elif case == "no_drainer":
        drainer = None
    elif case == "drain_failed":
        gcs_client.drain_node.side_effect = Exception("error")
    error_queue = Queue()
    stopper = RayStopper(gcs_client, error_queue, underutilized_drainer=drainer)

    stopper.notify([_ray_stop_requested_event(_drain_request("i-1", R1))])

    wait_for_condition(lambda: not error_queue.empty())
    assert error_queue.get_nowait() == RayStopError(im_instance_id="i-1")
    # Never stop ray without a preemption drain.
    assert gcs_client.drain_nodes.call_count == 0
    if case != "drain_failed":
        assert gcs_client.drain_node.call_count == 0
    if drainer is not None:
        assert drainer.registry.get("i-1") is None


#########################################################################
# Reconciler
#########################################################################


@pytest.mark.parametrize("deadline_passed", [True, False])
def test_reconciler_terminates_after_drain_deadline(deadline_passed):
    instance_storage = InstanceStorage(
        cluster_id="test_cluster_id", storage=InMemoryStorage()
    )
    instance_manager = InstanceManager(
        instance_storage=instance_storage, instance_status_update_subscribers=[]
    )
    instance_storage.upsert_instance(
        create_instance(
            "i-1",
            status=Instance.RAY_STOPPING,
            ray_node_id=R1,
            cloud_instance_id="c-1",
        )
    )

    clock = FakeClock(1000)
    drainer = UnderutilizedNodeDrainer(None, clock=clock)
    drainer.registry.add(
        DrainRecord(
            instance_id="i-1",
            ray_node_id=R1,
            node_type="cpu",
            workload=NodeWorkload(ray_node_id=R1),
            drain_deadline_ms=(900 if deadline_passed else 1200) * 1000,
        )
    )

    Reconciler._handle_stuck_instances(
        instance_manager=instance_manager,
        reconcile_config=InstanceReconcileConfig(),
        _logger=logging.getLogger(__name__),
        underutilized_drainer=drainer,
    )

    instances, _ = instance_storage.get_instances()
    expected = Instance.TERMINATING if deadline_passed else Instance.RAY_STOPPING
    assert instances["i-1"].status == expected


def test_event_logger_underutilized_cause():
    # Doesn't raise on the UNKNOWN cause of underutilized drains.
    event_logger.log_cluster_scheduling_update(
        cluster_resources={},
        terminate_requests=[_drain_request("i-1", R1)],
    )


if __name__ == "__main__":
    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
