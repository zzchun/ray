"""Resource-aware accelerator device injection for container runtime envs.

Ray sets accelerator *visibility* environment variables (``CUDA_VISIBLE_DEVICES``
and friends) inside the worker, but those only constrain libraries that honour
them. A worker that runs inside a ``runtime_env`` container can still open any
host GPU device node the container was given -- for example when the container
runs with ``--privileged`` or with ``/dev`` bind-mounted -- so a worker holding
one GPU can reach a GPU that belongs to another worker.

This module turns the accelerator allocation the raylet made for a worker into
an *exact* device list for the container runtime, so that:

  * a CPU-only worker is given no GPU device at all,
  * a worker holding one GPU can only open that GPU,
  * MIG / vGPU allocations are expressed with their own device identifiers,
  * RDMA devices are modelled explicitly instead of mounting all of ``/dev``.

The allocation is per worker, but the runtime env context (and therefore the
``podman run`` prefix) is created once per runtime env and cached by the runtime
env agent. The container plugin therefore emits a placeholder token into the
container command, and :func:`resolve_device_args_placeholder` expands it in the
per-worker ``setup_worker.py`` process, where the raylet's allocation for *this*
worker is available in the environment.

Two injection back ends are supported:

``cdi``
    The Container Device Interface. Devices are requested by name, e.g.
    ``--device nvidia.com/gpu=GPU-2b0f9c31-...``. This is the preferred mode:
    it names devices by UUID, so the list is unambiguous under MIG and does not
    depend on device enumeration order inside the container.

``nvidia``
    The NVIDIA Container Toolkit hook, which reads ``NVIDIA_VISIBLE_DEVICES``
    from the container environment and injects exactly the listed devices.

Both back ends are *fail closed*: when the worker's allocation cannot be
determined, or when device resolution fails, the container is started with
``NVIDIA_VISIBLE_DEVICES=void`` (no GPU, and no driver injection) rather than
inheriting whatever the image happens to default to.
"""

import json
import logging
import os
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

default_logger = logging.getLogger(__name__)

# Token embedded into the container command by the container runtime env
# plugins. It is expanded per worker by `resolve_device_args_placeholder`.
# It intentionally contains no shell metacharacters so that it survives the
# `bash -c` that runs the container command.
DEVICE_ARGS_PLACEHOLDER = "@RAY_CONTAINER_DEVICE_ARGS@"

# Set by the raylet on the worker startup process. JSON object mapping an
# accelerator resource name to the instance ids allocated to this worker, e.g.
# `{"GPU": ["0", "3"]}`. An empty mapping means "this worker holds no
# accelerator"; an absent variable means "unknown" and is also treated as
# holding no accelerator.
RAY_ASSIGNED_ACCELERATOR_IDS_ENV_VAR = "RAY_ASSIGNED_ACCELERATOR_IDS"

# `auto` (default), `cdi`, `nvidia` or `legacy`. `legacy` restores the
# pre-injection behaviour: Ray adds no device flags and the container sees
# whatever the image and the user's run options give it.
RAY_CONTAINER_DEVICE_MODE_ENV_VAR = "RAY_RUNTIME_ENV_CONTAINER_DEVICE_MODE"

# CDI kind used to request NVIDIA GPUs. Vendors that ship a different kind
# (e.g. `runtime.nvidia.com/gpu`) can override it.
RAY_CONTAINER_CDI_KIND_ENV_VAR = "RAY_RUNTIME_ENV_CONTAINER_CDI_KIND"
DEFAULT_NVIDIA_CDI_KIND = "nvidia.com/gpu"

# RDMA devices to expose to container workers. Unset means none. `auto`
# enumerates the InfiniBand verbs character devices. Otherwise a comma
# separated list of device node paths.
RAY_CONTAINER_RDMA_DEVICES_ENV_VAR = "RAY_RUNTIME_ENV_CONTAINER_RDMA_DEVICES"

# When set, unsafe container run options (`--privileged`, a bind mount of
# `/dev`, ...) and undetectable GPU runtimes raise instead of warning.
RAY_CONTAINER_STRICT_ISOLATION_ENV_VAR = (
    "RAY_RUNTIME_ENV_CONTAINER_STRICT_DEVICE_ISOLATION"
)

# Ray's own visibility env vars are redundant once the container holds exactly
# the assigned devices, and actively harmful: Ray would set
# `CUDA_VISIBLE_DEVICES` to *host* indices, which do not address the same
# devices inside a container that only has a subset of them.
NOSET_CUDA_VISIBLE_DEVICES_ENV_VAR = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"

NVIDIA_VISIBLE_DEVICES_ENV_VAR = "NVIDIA_VISIBLE_DEVICES"
NVIDIA_DRIVER_CAPABILITIES_ENV_VAR = "NVIDIA_DRIVER_CAPABILITIES"
DEFAULT_NVIDIA_DRIVER_CAPABILITIES = "compute,utility"

# `void` tells the NVIDIA container runtime hook to treat the container as a
# non-GPU container: no devices *and* no driver injection. `none` would still
# inject the driver, so `void` is the stricter choice for CPU-only workers.
NVIDIA_VISIBLE_DEVICES_NONE = "void"

# Directories searched for CDI specs, in the order the CDI specification lists
# them (a spec in the later directory overrides the earlier one).
CDI_SPEC_DIRS = ("/etc/cdi", "/var/run/cdi")

INFINIBAND_DEV_DIR = "/dev/infiniband"

# Run options that defeat device isolation no matter what Ray injects.
_UNSAFE_RUN_OPTION_PATTERNS = (
    ("--privileged", "runs the container with full access to all host devices"),
    ("--device=/dev ", "exposes every host device node to the container"),
    ("-v /dev:", "bind mounts the host /dev into the container"),
    ("--volume /dev:", "bind mounts the host /dev into the container"),
    ("--volume=/dev:", "bind mounts the host /dev into the container"),
)


class DeviceInjectionMode(str, Enum):
    """How Ray hands accelerator devices to a container worker."""

    AUTO = "auto"
    CDI = "cdi"
    NVIDIA = "nvidia"
    LEGACY = "legacy"


@dataclass(frozen=True)
class NvidiaDevice:
    """A single NVIDIA device that a worker is allowed to open.

    Attributes:
        ray_id: The accelerator instance id Ray allocated, as it appears in
            ``RAY_ASSIGNED_ACCELERATOR_IDS``.
        reference: The identifier handed to the container runtime. A device
            UUID (``GPU-...``), a MIG UUID (``MIG-...``) or, when NVML is not
            reachable, the raw index Ray allocated.
        kind: ``"gpu"``, ``"mig"`` or ``"vgpu"``.
    """

    ray_id: str
    reference: str
    kind: str = "gpu"


@dataclass
class ContainerDeviceConfig:
    """The container flags that expose exactly the assigned devices."""

    run_args: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)

    def to_run_options(self) -> List[str]:
        """Flatten into ``podman run`` arguments."""
        options = list(self.run_args)
        for name, value in self.env_vars.items():
            options.extend(["--env", f"{name}={value}"])
        return options

    def to_command_string(self) -> str:
        """Flatten into a shell-quoted fragment of a ``podman run`` command."""
        return " ".join(shlex.quote(option) for option in self.to_run_options())


def _strict_isolation(environ: Optional[Dict[str, str]] = None) -> bool:
    environ = os.environ if environ is None else environ
    return environ.get(RAY_CONTAINER_STRICT_ISOLATION_ENV_VAR, "0").lower() in (
        "1",
        "true",
    )


def get_device_injection_mode(
    environ: Optional[Dict[str, str]] = None,
) -> DeviceInjectionMode:
    """Read the configured injection mode, defaulting to ``auto``."""
    environ = os.environ if environ is None else environ
    raw = environ.get(RAY_CONTAINER_DEVICE_MODE_ENV_VAR, "").strip().lower()
    if not raw:
        return DeviceInjectionMode.AUTO
    try:
        return DeviceInjectionMode(raw)
    except ValueError:
        raise ValueError(
            f"Invalid {RAY_CONTAINER_DEVICE_MODE_ENV_VAR}={raw!r}. Expected one of "
            f"{[mode.value for mode in DeviceInjectionMode]}."
        )


def parse_assigned_accelerator_ids(
    environ: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, List[str]]]:
    """Parse the accelerator allocation the raylet made for this worker.

    Returns ``None`` when the raylet did not report an allocation at all, which
    is distinct from reporting an empty one: an empty mapping means "this
    worker holds no accelerator", while ``None`` means "unknown", as happens
    when the worker is started by a raylet that predates device injection.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(RAY_ASSIGNED_ACCELERATOR_IDS_ENV_VAR)
    if raw is None or raw == "":
        return None

    assigned = json.loads(raw)
    if not isinstance(assigned, dict):
        raise ValueError(
            f"{RAY_ASSIGNED_ACCELERATOR_IDS_ENV_VAR} must be a JSON object mapping "
            f"resource names to instance ids, got {raw!r}."
        )
    return {
        resource_name: [str(instance_id) for instance_id in instance_ids]
        for resource_name, instance_ids in assigned.items()
    }


def _cdi_specs_contain_kind(
    kind: str, spec_dirs: Sequence[str] = CDI_SPEC_DIRS
) -> bool:
    """Return whether any installed CDI spec declares ``kind``.

    CDI specs are small JSON/YAML documents, so scanning them is cheap. Both
    encodings write the kind on a single line, so a substring match over the
    file is enough to tell whether the runtime can resolve the kind.
    """
    for spec_dir in spec_dirs:
        try:
            spec_names = sorted(os.listdir(spec_dir))
        except OSError:
            continue
        for spec_name in spec_names:
            if not spec_name.endswith((".json", ".yaml", ".yml")):
                continue
            try:
                with open(os.path.join(spec_dir, spec_name), "r") as spec_file:
                    if kind in spec_file.read():
                        return True
            except OSError:
                continue
    return False


def _nvidia_container_runtime_available() -> bool:
    """Return whether the NVIDIA Container Toolkit hook is installed."""
    import shutil

    return any(
        shutil.which(executable) is not None
        for executable in ("nvidia-container-runtime-hook", "nvidia-container-runtime")
    )


def _resolve_auto_mode(cdi_kind: str, logger: logging.Logger) -> DeviceInjectionMode:
    if _cdi_specs_contain_kind(cdi_kind):
        return DeviceInjectionMode.CDI
    if _nvidia_container_runtime_available():
        return DeviceInjectionMode.NVIDIA
    return DeviceInjectionMode.LEGACY


def _kind_of_reference(reference: str) -> str:
    """Classify a device identifier by its UUID prefix."""
    return "mig" if reference.startswith("MIG-") else "gpu"


def _import_pynvml():
    """Import the vendored pynvml.

    Split out as its own function so that callers have a single place to handle
    a node without an NVIDIA driver, and so tests can substitute it.
    """
    import ray._private.thirdparty.pynvml as pynvml

    return pynvml


def _nvml_device_reference(
    pynvml, nvml_ref: str, logger: logging.Logger
) -> NvidiaDevice:
    """Resolve one NVML device reference to a stable device identifier.

    ``nvml_ref`` is either an index into the NVML device list or a device UUID
    (which is what operators put in ``CUDA_VISIBLE_DEVICES`` when they hand MIG
    instances to Ray). Indices are resolved to UUIDs so that the identifier
    handed to the container runtime does not depend on enumeration order.
    """
    if not nvml_ref.isdigit():
        # Already a stable UUID. Resolving it through NVML would only tell us
        # whether it is a vGPU, which does not change the identifier.
        return NvidiaDevice(
            ray_id=nvml_ref, reference=nvml_ref, kind=_kind_of_reference(nvml_ref)
        )

    handle = pynvml.nvmlDeviceGetHandleByIndex(int(nvml_ref))
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    if isinstance(uuid, bytes):
        uuid = uuid.decode("utf-8")

    kind = "gpu"
    try:
        current_mig_mode, _ = pynvml.nvmlDeviceGetMigMode(handle)
        if current_mig_mode == pynvml.NVML_DEVICE_MIG_ENABLE:
            kind = "mig-parent"
            logger.warning(
                "MIG is enabled on GPU %s but Ray allocated the parent device, so "
                "the container is given every MIG instance of that GPU. Set "
                "CUDA_VISIBLE_DEVICES to MIG UUIDs on this node so that Ray "
                "schedules MIG instances individually.",
                nvml_ref,
            )
    except pynvml.NVMLError:
        # Pre-MIG drivers and non-datacenter cards do not implement the query.
        pass

    if kind == "gpu":
        try:
            virtualization_mode = pynvml.nvmlDeviceGetVirtualizationMode(handle)
            if virtualization_mode in (
                pynvml.NVML_GPU_VIRTUALIZATION_MODE_VGPU,
                pynvml.NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU,
            ):
                kind = "vgpu"
        except pynvml.NVMLError:
            pass

    return NvidiaDevice(ray_id=nvml_ref, reference=uuid, kind=kind)


def resolve_nvidia_devices(
    ray_instance_ids: Sequence[str],
    logger: logging.Logger = default_logger,
) -> List[NvidiaDevice]:
    """Map Ray GPU instance ids onto NVIDIA device identifiers.

    Ray's instance ids index the *node's* visible device list, so an id of ``0``
    on a raylet started with ``CUDA_VISIBLE_DEVICES=2,5`` means host device 2.
    The node's visible list is read from the environment of this process, which
    the raylet passes down to the worker startup process.

    Falls back to the raw ids when NVML is unreachable: the identifiers are
    still exact, they are just indices rather than UUIDs.
    """
    if not ray_instance_ids:
        return []

    node_visible_ids = None
    try:
        from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager

        node_visible_ids = (
            NvidiaGPUAcceleratorManager.get_current_process_visible_accelerator_ids()
        )
    except Exception:
        logger.debug("Could not read the node's visible GPU ids.", exc_info=True)

    nvml_refs = []
    for ray_id in ray_instance_ids:
        if node_visible_ids is None or not ray_id.isdigit():
            nvml_refs.append(ray_id)
            continue
        index = int(ray_id)
        if index >= len(node_visible_ids):
            raise ValueError(
                f"Ray allocated GPU instance {index} to this worker but the node "
                f"only makes {node_visible_ids} visible. The raylet's "
                "CUDA_VISIBLE_DEVICES and its GPU resource count disagree."
            )
        nvml_refs.append(node_visible_ids[index])

    try:
        pynvml = _import_pynvml()
        pynvml.nvmlInit()
    except Exception:
        logger.warning(
            "NVML is not available, so GPU devices %s are passed to the container "
            "by index instead of by UUID. Indices are only stable as long as the "
            "host's device enumeration order is.",
            nvml_refs,
        )
        return [
            NvidiaDevice(
                ray_id=ray_id, reference=nvml_ref, kind=_kind_of_reference(nvml_ref)
            )
            for ray_id, nvml_ref in zip(ray_instance_ids, nvml_refs)
        ]

    try:
        devices = []
        for ray_id, nvml_ref in zip(ray_instance_ids, nvml_refs):
            device = _nvml_device_reference(pynvml, nvml_ref, logger)
            devices.append(
                NvidiaDevice(
                    ray_id=ray_id, reference=device.reference, kind=device.kind
                )
            )
        return devices
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def resolve_rdma_device_args(
    environ: Optional[Dict[str, str]] = None,
    logger: logging.Logger = default_logger,
) -> List[str]:
    """Build ``--device`` flags for the RDMA devices the operator opted into.

    RDMA is modelled separately from accelerators: an operator either exposes a
    named set of device nodes or none at all. Ray never falls back to mounting
    all of ``/dev`` or to ``--privileged``.
    """
    environ = os.environ if environ is None else environ
    configured = environ.get(RAY_CONTAINER_RDMA_DEVICES_ENV_VAR, "").strip()
    if not configured:
        return []

    if configured.lower() == "auto":
        try:
            device_names = sorted(os.listdir(INFINIBAND_DEV_DIR))
        except OSError:
            logger.warning(
                "%s=auto but %s does not exist, so no RDMA device is exposed to "
                "container workers.",
                RAY_CONTAINER_RDMA_DEVICES_ENV_VAR,
                INFINIBAND_DEV_DIR,
            )
            return []
        device_paths = [
            os.path.join(INFINIBAND_DEV_DIR, device_name)
            for device_name in device_names
        ]
    else:
        device_paths = [path.strip() for path in configured.split(",") if path.strip()]

    args = []
    for device_path in device_paths:
        if not os.path.exists(device_path):
            logger.warning(
                "Skipping RDMA device %s: it does not exist on this node.", device_path
            )
            continue
        args.extend(["--device", device_path])

    if args:
        # Registering memory regions with the HCA requires locking that memory,
        # and the default container memlock limit is far below what any real
        # transfer needs.
        args.extend(["--ulimit", "memlock=-1:-1"])
    return args


def validate_container_run_options(
    run_options: Optional[Sequence[str]],
    logger: logging.Logger = default_logger,
    environ: Optional[Dict[str, str]] = None,
) -> None:
    """Flag user run options that defeat the device isolation Ray sets up.

    Warns by default so that existing deployments keep working, and raises when
    ``RAY_RUNTIME_ENV_CONTAINER_STRICT_DEVICE_ISOLATION`` is set.
    """
    if not run_options:
        return

    # Match against a padded, normalized string so that `--device=/dev ` also
    # catches a trailing `--device=/dev` at the end of the options.
    joined = " ".join(run_options) + " "
    problems = []
    for pattern, explanation in _UNSAFE_RUN_OPTION_PATTERNS:
        if pattern in joined:
            problems.append(f"`{pattern.strip()}` {explanation}")

    if f"{NVIDIA_VISIBLE_DEVICES_ENV_VAR}=all" in joined:
        problems.append(
            f"`{NVIDIA_VISIBLE_DEVICES_ENV_VAR}=all` exposes every host GPU, "
            "overriding the device list Ray computed for this worker"
        )

    if not problems:
        return

    message = (
        "The container run options for this runtime env defeat Ray's GPU device "
        "isolation: " + "; ".join(problems) + ". The worker will be able to open "
        "devices that belong to other workers."
    )
    if _strict_isolation(environ):
        raise ValueError(message)
    logger.warning(message)


def build_container_device_config(
    assigned_accelerator_ids: Optional[Dict[str, List[str]]],
    mode: DeviceInjectionMode = DeviceInjectionMode.AUTO,
    environ: Optional[Dict[str, str]] = None,
    logger: logging.Logger = default_logger,
) -> ContainerDeviceConfig:
    """Build the container flags that expose exactly the assigned devices.

    Args:
        assigned_accelerator_ids: The allocation the raylet made for this
            worker, e.g. ``{"GPU": ["0", "3"]}``. An empty mapping means the
            worker holds no accelerator and must not see any GPU. ``None``
            means the raylet did not report one, in which case Ray keeps the
            pre-injection behaviour unless strict isolation is enabled.
        mode: Injection back end. ``AUTO`` picks CDI when CDI specs for the
            NVIDIA kind are installed, otherwise the NVIDIA Container Toolkit
            hook, otherwise no GPU flags at all.
        environ: Environment used for configuration lookups. Defaults to
            ``os.environ``.
    """
    environ = os.environ if environ is None else environ
    config = ContainerDeviceConfig()
    config.run_args.extend(resolve_rdma_device_args(environ, logger))

    if mode == DeviceInjectionMode.LEGACY:
        return config

    if assigned_accelerator_ids is None:
        message = (
            "The raylet did not report an accelerator allocation for this container "
            f"worker (no {RAY_ASSIGNED_ACCELERATOR_IDS_ENV_VAR} in the environment), "
            "so Ray cannot restrict the container to the worker's own devices. This "
            "happens when the worker is started by a raylet older than the node's "
            "Ray version."
        )
        if _strict_isolation(environ):
            raise RuntimeError(message)
        logger.warning(message)
        return config

    cdi_kind = environ.get(RAY_CONTAINER_CDI_KIND_ENV_VAR) or DEFAULT_NVIDIA_CDI_KIND
    gpu_ids = assigned_accelerator_ids.get("GPU", [])

    if mode == DeviceInjectionMode.AUTO:
        mode = _resolve_auto_mode(cdi_kind, logger)
        if mode == DeviceInjectionMode.LEGACY and gpu_ids:
            message = (
                "Ray allocated GPUs to a container worker but found neither CDI "
                f"specs for kind `{cdi_kind}` nor the NVIDIA Container Toolkit on "
                "this node, so it cannot restrict the container to those GPUs. "
                "Install the NVIDIA Container Toolkit (`nvidia-ctk cdi generate`) "
                "to get exact device injection."
            )
            if _strict_isolation(environ):
                raise RuntimeError(message)
            logger.warning(message)
            return config

    if not gpu_ids:
        # Fail closed: `void` also suppresses driver injection, so a CPU-only
        # worker has no usable NVIDIA device even if the image asks for one.
        config.env_vars[NVIDIA_VISIBLE_DEVICES_ENV_VAR] = NVIDIA_VISIBLE_DEVICES_NONE
        return config

    devices = resolve_nvidia_devices(gpu_ids, logger)
    logger.info(
        "Restricting container worker to GPU devices %s (mode=%s).",
        [device.reference for device in devices],
        mode.value,
    )

    if mode == DeviceInjectionMode.CDI:
        for device in devices:
            config.run_args.extend(["--device", f"{cdi_kind}={device.reference}"])
        # CDI has already injected the devices; make sure a toolkit hook baked
        # into the image does not inject a second, wider set on top.
        config.env_vars[NVIDIA_VISIBLE_DEVICES_ENV_VAR] = NVIDIA_VISIBLE_DEVICES_NONE
    else:
        config.env_vars[NVIDIA_VISIBLE_DEVICES_ENV_VAR] = ",".join(
            device.reference for device in devices
        )
        config.env_vars.setdefault(
            NVIDIA_DRIVER_CAPABILITIES_ENV_VAR, DEFAULT_NVIDIA_DRIVER_CAPABILITIES
        )

    # The container now holds exactly the assigned devices and renumbers them
    # from 0, so Ray must not additionally set CUDA_VISIBLE_DEVICES to the
    # host-side indices it allocated.
    config.env_vars[NOSET_CUDA_VISIBLE_DEVICES_ENV_VAR] = "1"
    return config


def resolve_device_args_placeholder(
    command: str,
    environ: Optional[Dict[str, str]] = None,
    logger: logging.Logger = default_logger,
) -> str:
    """Expand :data:`DEVICE_ARGS_PLACEHOLDER` for the worker being started.

    Called from the per-worker ``setup_worker.py`` process, which is where the
    raylet's allocation for this worker is visible. Any failure degrades to the
    fail-closed flag set (no GPU device) rather than starting a container with
    whatever devices the image defaults to.
    """
    if DEVICE_ARGS_PLACEHOLDER not in command:
        return command

    environ = os.environ if environ is None else environ
    try:
        config = build_container_device_config(
            parse_assigned_accelerator_ids(environ),
            get_device_injection_mode(environ),
            environ,
            logger,
        )
        replacement = config.to_command_string()
    except Exception:
        logger.exception(
            "Failed to compute the accelerator device list for this container "
            "worker. Starting it without any GPU device."
        )
        replacement = ContainerDeviceConfig(
            env_vars={NVIDIA_VISIBLE_DEVICES_ENV_VAR: NVIDIA_VISIBLE_DEVICES_NONE}
        ).to_command_string()

    return command.replace(DEVICE_ARGS_PLACEHOLDER, replacement)
