import json
import logging
import sys

import pytest

from ray._private.runtime_env import container_devices
from ray._private.runtime_env.container_devices import (
    ContainerDeviceConfig,
    DeviceInjectionMode,
    NvidiaDevice,
    build_container_device_config,
    get_device_injection_mode,
    parse_assigned_accelerator_ids,
    resolve_device_args_placeholder,
    resolve_nvidia_devices,
    resolve_rdma_device_args,
    validate_container_run_options,
)

CUDA = container_devices.NVIDIA_VISIBLE_DEVICES_ENV_VAR
NOSET = container_devices.NOSET_CUDA_VISIBLE_DEVICES_ENV_VAR
VOID = container_devices.NVIDIA_VISIBLE_DEVICES_NONE

logger = logging.getLogger(__name__)


class _FakeNvmlError(Exception):
    pass


class _FakeNvml:
    """A stand-in for the vendored pynvml, driven by per-test attributes."""

    NVML_DEVICE_MIG_ENABLE = 1
    NVML_GPU_VIRTUALIZATION_MODE_VGPU = 2
    NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU = 3
    NVMLError = _FakeNvmlError

    def __init__(self):
        self.handles_requested = []
        self.shutdown_calls = 0
        self.uuids_as_bytes = False
        self.mig_enabled_indices = set()
        self.vgpu_indices = set()
        self.raise_on_handle = False

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetHandleByIndex(self, index):
        if self.raise_on_handle:
            raise RuntimeError("boom")
        self.handles_requested.append(index)
        return index

    def nvmlDeviceGetUUID(self, handle):
        uuid = f"GPU-nvml-{handle}"
        return uuid.encode("utf-8") if self.uuids_as_bytes else uuid

    def nvmlDeviceGetMigMode(self, handle):
        enabled = (
            self.NVML_DEVICE_MIG_ENABLE if handle in self.mig_enabled_indices else 0
        )
        return (enabled, enabled)

    def nvmlDeviceGetVirtualizationMode(self, handle):
        if handle in self.vgpu_indices:
            return self.NVML_GPU_VIRTUALIZATION_MODE_VGPU
        return 0


@pytest.fixture
def fake_nvml(monkeypatch):
    nvml = _FakeNvml()
    monkeypatch.setattr(container_devices, "_import_pynvml", lambda: nvml)
    return nvml


@pytest.fixture
def no_nvml(monkeypatch):
    """Simulate a node where the NVIDIA driver is not reachable."""

    def _unavailable():
        raise ImportError("no driver")

    monkeypatch.setattr(container_devices, "_import_pynvml", _unavailable)


@pytest.fixture
def stub_nvidia_devices(monkeypatch):
    """Resolve Ray GPU ids to `GPU-uuid-<id>` without touching NVML."""

    def _resolve(ray_instance_ids, logger=None):
        return [
            NvidiaDevice(ray_id=ray_id, reference=f"GPU-uuid-{ray_id}")
            for ray_id in ray_instance_ids
        ]

    monkeypatch.setattr(container_devices, "resolve_nvidia_devices", _resolve)


class TestParseAssignedAcceleratorIds:
    @pytest.mark.parametrize("raw", [None, ""])
    def test_absent_is_unknown(self, raw):
        environ = {} if raw is None else {"RAY_ASSIGNED_ACCELERATOR_IDS": raw}
        assert parse_assigned_accelerator_ids(environ) is None

    def test_empty_allocation_is_not_unknown(self):
        environ = {"RAY_ASSIGNED_ACCELERATOR_IDS": "{}"}
        assert parse_assigned_accelerator_ids(environ) == {}

    def test_parses_ids(self):
        environ = {"RAY_ASSIGNED_ACCELERATOR_IDS": '{"GPU": ["0", "3"]}'}
        assert parse_assigned_accelerator_ids(environ) == {"GPU": ["0", "3"]}

    def test_coerces_ids_to_strings(self):
        environ = {"RAY_ASSIGNED_ACCELERATOR_IDS": '{"GPU": [0, 3]}'}
        assert parse_assigned_accelerator_ids(environ) == {"GPU": ["0", "3"]}

    @pytest.mark.parametrize("raw", ["[]", '"GPU"', "3"])
    def test_rejects_non_objects(self, raw):
        with pytest.raises(ValueError):
            parse_assigned_accelerator_ids({"RAY_ASSIGNED_ACCELERATOR_IDS": raw})

    def test_rejects_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            parse_assigned_accelerator_ids({"RAY_ASSIGNED_ACCELERATOR_IDS": "{oops"})


class TestInjectionMode:
    def test_defaults_to_auto(self):
        assert get_device_injection_mode({}) == DeviceInjectionMode.AUTO

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("cdi", DeviceInjectionMode.CDI),
            ("NVIDIA", DeviceInjectionMode.NVIDIA),
            (" legacy ", DeviceInjectionMode.LEGACY),
        ],
    )
    def test_parses_configured_mode(self, raw, expected):
        environ = {"RAY_RUNTIME_ENV_CONTAINER_DEVICE_MODE": raw}
        assert get_device_injection_mode(environ) == expected

    def test_rejects_unknown_mode(self):
        environ = {"RAY_RUNTIME_ENV_CONTAINER_DEVICE_MODE": "yolo"}
        with pytest.raises(ValueError, match="yolo"):
            get_device_injection_mode(environ)

    @pytest.mark.parametrize(
        "has_cdi,has_toolkit,expected",
        [
            (True, True, DeviceInjectionMode.CDI),
            (True, False, DeviceInjectionMode.CDI),
            (False, True, DeviceInjectionMode.NVIDIA),
            (False, False, DeviceInjectionMode.LEGACY),
        ],
    )
    def test_auto_prefers_cdi(self, monkeypatch, has_cdi, has_toolkit, expected):
        monkeypatch.setattr(
            container_devices, "_cdi_specs_contain_kind", lambda kind, **kw: has_cdi
        )
        monkeypatch.setattr(
            container_devices,
            "_nvidia_container_runtime_available",
            lambda: has_toolkit,
        )
        assert (
            container_devices._resolve_auto_mode("nvidia.com/gpu", logger) == expected
        )

    def test_cdi_spec_scan_matches_kind(self, tmp_path):
        (tmp_path / "not-a-spec.txt").write_text("nvidia.com/gpu")
        (tmp_path / "other.json").write_text('{"kind": "amd.com/gpu"}')
        assert not container_devices._cdi_specs_contain_kind(
            "nvidia.com/gpu", [str(tmp_path)]
        )

        (tmp_path / "nvidia.yaml").write_text("kind: nvidia.com/gpu\n")
        assert container_devices._cdi_specs_contain_kind(
            "nvidia.com/gpu", [str(tmp_path)]
        )

    def test_cdi_spec_scan_tolerates_missing_dir(self, tmp_path):
        missing = str(tmp_path / "nope")
        assert not container_devices._cdi_specs_contain_kind(
            "nvidia.com/gpu", [missing]
        )


class TestBuildContainerDeviceConfig:
    def test_legacy_mode_injects_nothing(self):
        config = build_container_device_config(
            {"GPU": ["0"]}, DeviceInjectionMode.LEGACY, {}, logger
        )
        assert config.run_args == []
        assert config.env_vars == {}

    def test_unknown_allocation_keeps_previous_behaviour(self):
        config = build_container_device_config(
            None, DeviceInjectionMode.CDI, {}, logger
        )
        assert config.run_args == []
        assert config.env_vars == {}

    def test_unknown_allocation_raises_under_strict_isolation(self):
        environ = {"RAY_RUNTIME_ENV_CONTAINER_STRICT_DEVICE_ISOLATION": "1"}
        with pytest.raises(RuntimeError, match="did not report"):
            build_container_device_config(
                None, DeviceInjectionMode.CDI, environ, logger
            )

    @pytest.mark.parametrize(
        "mode", [DeviceInjectionMode.CDI, DeviceInjectionMode.NVIDIA]
    )
    def test_cpu_only_worker_gets_no_gpu(self, mode):
        config = build_container_device_config({}, mode, {}, logger)
        assert config.run_args == []
        assert config.env_vars == {CUDA: VOID}
        assert config.to_run_options() == ["--env", f"{CUDA}={VOID}"]

    def test_cpu_only_worker_with_other_accelerators_gets_no_gpu(self):
        config = build_container_device_config(
            {"TPU": ["0"]}, DeviceInjectionMode.CDI, {}, logger
        )
        assert config.env_vars == {CUDA: VOID}

    def test_cdi_mode_names_each_assigned_device(self, stub_nvidia_devices):
        config = build_container_device_config(
            {"GPU": ["0", "3"]}, DeviceInjectionMode.CDI, {}, logger
        )
        assert config.run_args == [
            "--device",
            "nvidia.com/gpu=GPU-uuid-0",
            "--device",
            "nvidia.com/gpu=GPU-uuid-3",
        ]
        # The toolkit hook must not inject a second, wider set on top of CDI.
        assert config.env_vars[CUDA] == VOID
        assert config.env_vars[NOSET] == "1"

    def test_cdi_kind_is_configurable(self, stub_nvidia_devices):
        environ = {"RAY_RUNTIME_ENV_CONTAINER_CDI_KIND": "runtime.nvidia.com/gpu"}
        config = build_container_device_config(
            {"GPU": ["1"]}, DeviceInjectionMode.CDI, environ, logger
        )
        assert config.run_args == ["--device", "runtime.nvidia.com/gpu=GPU-uuid-1"]

    def test_nvidia_mode_lists_each_assigned_device(self, stub_nvidia_devices):
        config = build_container_device_config(
            {"GPU": ["0", "3"]}, DeviceInjectionMode.NVIDIA, {}, logger
        )
        assert config.run_args == []
        assert config.env_vars[CUDA] == "GPU-uuid-0,GPU-uuid-3"
        assert config.env_vars["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"
        assert config.env_vars[NOSET] == "1"

    def test_auto_mode_without_a_gpu_runtime_warns_and_skips(self, monkeypatch, caplog):
        monkeypatch.setattr(
            container_devices, "_cdi_specs_contain_kind", lambda kind, **kw: False
        )
        monkeypatch.setattr(
            container_devices, "_nvidia_container_runtime_available", lambda: False
        )
        with caplog.at_level(logging.WARNING):
            config = build_container_device_config(
                {"GPU": ["0"]}, DeviceInjectionMode.AUTO, {}, logger
            )
        assert config.run_args == []
        assert config.env_vars == {}
        assert "NVIDIA Container Toolkit" in caplog.text

    def test_auto_mode_without_a_gpu_runtime_raises_under_strict_isolation(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            container_devices, "_cdi_specs_contain_kind", lambda kind, **kw: False
        )
        monkeypatch.setattr(
            container_devices, "_nvidia_container_runtime_available", lambda: False
        )
        environ = {"RAY_RUNTIME_ENV_CONTAINER_STRICT_DEVICE_ISOLATION": "true"}
        with pytest.raises(RuntimeError, match="NVIDIA Container Toolkit"):
            build_container_device_config(
                {"GPU": ["0"]}, DeviceInjectionMode.AUTO, environ, logger
            )


class TestResolveNvidiaDevices:
    def test_empty_allocation(self):
        assert resolve_nvidia_devices([], logger) == []

    def test_falls_back_to_indices_when_nvml_is_unavailable(
        self, monkeypatch, no_nvml, caplog
    ):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with caplog.at_level(logging.WARNING):
            devices = resolve_nvidia_devices(["0", "3"], logger)
        assert [device.reference for device in devices] == ["0", "3"]
        assert "NVML is not available" in caplog.text

    def test_ray_ids_index_the_nodes_visible_devices(self, monkeypatch, no_nvml):
        # A raylet started with CUDA_VISIBLE_DEVICES=2,5 hands out instance ids 0 and 1,
        # which address host devices 2 and 5.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
        devices = resolve_nvidia_devices(["1"], logger)
        assert [device.reference for device in devices] == ["5"]

    def test_rejects_ids_beyond_the_nodes_visible_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        with pytest.raises(ValueError, match="only makes"):
            resolve_nvidia_devices(["1"], logger)

    def test_mig_uuids_from_the_node_are_kept(self, monkeypatch, no_nvml):
        # Operators hand MIG instances to Ray by putting their UUIDs in
        # CUDA_VISIBLE_DEVICES, so the ids Ray allocates map onto MIG UUIDs.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-aaa,MIG-bbb")
        devices = resolve_nvidia_devices(["0"], logger)
        assert [device.reference for device in devices] == ["MIG-aaa"]
        assert [device.kind for device in devices] == ["mig"]

    def test_uuids_are_kept_when_nvml_is_reachable(self, monkeypatch, fake_nvml):
        # Ray ids that are already UUIDs address a device directly, so there is
        # nothing for NVML to resolve.
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        devices = resolve_nvidia_devices(["GPU-abc", "MIG-def"], logger)
        assert [device.reference for device in devices] == ["GPU-abc", "MIG-def"]
        assert [device.kind for device in devices] == ["gpu", "mig"]
        assert fake_nvml.handles_requested == []

    def test_indices_are_resolved_to_uuids(self, monkeypatch, fake_nvml):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        devices = resolve_nvidia_devices(["0", "2"], logger)
        assert [device.reference for device in devices] == ["GPU-nvml-0", "GPU-nvml-2"]
        assert fake_nvml.handles_requested == [0, 2]
        assert fake_nvml.shutdown_calls == 1

    def test_uuid_bytes_from_nvml_are_decoded(self, monkeypatch, fake_nvml):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        fake_nvml.uuids_as_bytes = True
        devices = resolve_nvidia_devices(["1"], logger)
        assert devices[0].reference == "GPU-nvml-1"

    def test_mig_enabled_parent_is_flagged(self, monkeypatch, fake_nvml, caplog):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        fake_nvml.mig_enabled_indices = {0}
        with caplog.at_level(logging.WARNING):
            devices = resolve_nvidia_devices(["0"], logger)
        assert devices[0].kind == "mig-parent"
        assert "MIG is enabled" in caplog.text

    def test_vgpu_is_flagged(self, monkeypatch, fake_nvml):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        fake_nvml.vgpu_indices = {0}
        assert resolve_nvidia_devices(["0"], logger)[0].kind == "vgpu"

    def test_nvml_is_shut_down_even_when_resolution_fails(self, monkeypatch, fake_nvml):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        fake_nvml.raise_on_handle = True
        with pytest.raises(RuntimeError, match="boom"):
            resolve_nvidia_devices(["0"], logger)
        assert fake_nvml.shutdown_calls == 1


class TestRdmaDevices:
    def test_disabled_by_default(self):
        assert resolve_rdma_device_args({}, logger) == []

    def test_explicit_device_list(self, tmp_path):
        uverbs0 = tmp_path / "uverbs0"
        uverbs0.write_text("")
        environ = {"RAY_RUNTIME_ENV_CONTAINER_RDMA_DEVICES": str(uverbs0)}
        assert resolve_rdma_device_args(environ, logger) == [
            "--device",
            str(uverbs0),
            "--ulimit",
            "memlock=-1:-1",
        ]

    def test_skips_devices_that_do_not_exist(self, tmp_path, caplog):
        environ = {
            "RAY_RUNTIME_ENV_CONTAINER_RDMA_DEVICES": str(tmp_path / "uverbs9"),
        }
        with caplog.at_level(logging.WARNING):
            assert resolve_rdma_device_args(environ, logger) == []
        assert "does not exist" in caplog.text

    def test_auto_enumerates_infiniband(self, tmp_path, monkeypatch):
        (tmp_path / "uverbs0").write_text("")
        (tmp_path / "rdma_cm").write_text("")
        monkeypatch.setattr(container_devices, "INFINIBAND_DEV_DIR", str(tmp_path))
        environ = {"RAY_RUNTIME_ENV_CONTAINER_RDMA_DEVICES": "auto"}
        assert resolve_rdma_device_args(environ, logger) == [
            "--device",
            str(tmp_path / "rdma_cm"),
            "--device",
            str(tmp_path / "uverbs0"),
            "--ulimit",
            "memlock=-1:-1",
        ]

    def test_auto_tolerates_a_node_without_infiniband(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            container_devices, "INFINIBAND_DEV_DIR", str(tmp_path / "nope")
        )
        environ = {"RAY_RUNTIME_ENV_CONTAINER_RDMA_DEVICES": "auto"}
        assert resolve_rdma_device_args(environ, logger) == []

    def test_rdma_devices_are_added_alongside_gpus(self, tmp_path, stub_nvidia_devices):
        uverbs0 = tmp_path / "uverbs0"
        uverbs0.write_text("")
        environ = {"RAY_RUNTIME_ENV_CONTAINER_RDMA_DEVICES": str(uverbs0)}
        config = build_container_device_config(
            {"GPU": ["0"]}, DeviceInjectionMode.CDI, environ, logger
        )
        assert config.run_args == [
            "--device",
            str(uverbs0),
            "--ulimit",
            "memlock=-1:-1",
            "--device",
            "nvidia.com/gpu=GPU-uuid-0",
        ]


class TestValidateContainerRunOptions:
    @pytest.mark.parametrize(
        "run_options",
        [
            ["--cpus", "2"],
            # A specific device node, or /dev/shm, is not a blanket /dev mount.
            ["--device=/dev/nvidia0"],
            ["--device", "/dev/infiniband/uverbs0"],
            ["-v", "/dev/shm:/dev/shm"],
            ["--env", "NVIDIA_VISIBLE_DEVICES=GPU-2b0f9c31"],
        ],
    )
    def test_accepts_ordinary_options(self, run_options, caplog):
        with caplog.at_level(logging.WARNING):
            validate_container_run_options(run_options, logger, {})
        assert caplog.text == ""

    def test_accepts_no_options(self):
        validate_container_run_options(None, logger, {})

    @pytest.mark.parametrize(
        "run_options",
        [
            ["--privileged"],
            ["-v", "/dev:/dev"],
            ["--volume", "/dev:/dev"],
            ["--volume=/dev:/dev"],
            ["--device=/dev"],
            ["--env", "NVIDIA_VISIBLE_DEVICES=all"],
        ],
    )
    def test_warns_about_options_that_defeat_isolation(self, run_options, caplog):
        with caplog.at_level(logging.WARNING):
            validate_container_run_options(run_options, logger, {})
        assert "device isolation" in caplog.text

    @pytest.mark.parametrize(
        "run_options", [["--privileged"], ["--env", "NVIDIA_VISIBLE_DEVICES=all"]]
    )
    def test_raises_under_strict_isolation(self, run_options):
        environ = {"RAY_RUNTIME_ENV_CONTAINER_STRICT_DEVICE_ISOLATION": "1"}
        with pytest.raises(ValueError, match="device isolation"):
            validate_container_run_options(run_options, logger, environ)


class TestResolveDeviceArgsPlaceholder:
    def test_command_without_the_placeholder_is_untouched(self):
        command = "podman run --rm image"
        assert resolve_device_args_placeholder(command, {}, logger) == command

    def test_placeholder_is_replaced_with_the_device_flags(self, stub_nvidia_devices):
        environ = {
            "RAY_ASSIGNED_ACCELERATOR_IDS": '{"GPU": ["2"]}',
            "RAY_RUNTIME_ENV_CONTAINER_DEVICE_MODE": "cdi",
        }
        command = f"podman run {container_devices.DEVICE_ARGS_PLACEHOLDER} image"
        resolved = resolve_device_args_placeholder(command, environ, logger)
        assert container_devices.DEVICE_ARGS_PLACEHOLDER not in resolved
        assert "--device nvidia.com/gpu=GPU-uuid-2" in resolved
        assert f"--env {CUDA}={VOID}" in resolved

    def test_cpu_only_worker_placeholder(self):
        environ = {
            "RAY_ASSIGNED_ACCELERATOR_IDS": "{}",
            "RAY_RUNTIME_ENV_CONTAINER_DEVICE_MODE": "nvidia",
        }
        command = f"podman run {container_devices.DEVICE_ARGS_PLACEHOLDER} image"
        resolved = resolve_device_args_placeholder(command, environ, logger)
        assert resolved == f"podman run --env {CUDA}={VOID} image"

    def test_failures_fall_back_to_no_gpu(self, caplog):
        environ = {"RAY_ASSIGNED_ACCELERATOR_IDS": "{not json"}
        command = f"podman run {container_devices.DEVICE_ARGS_PLACEHOLDER} image"
        with caplog.at_level(logging.ERROR):
            resolved = resolve_device_args_placeholder(command, environ, logger)
        assert resolved == f"podman run --env {CUDA}={VOID} image"
        assert "without any GPU device" in caplog.text


class TestContainerDeviceConfig:
    def test_shell_quotes_values(self):
        config = ContainerDeviceConfig(
            run_args=["--device", "/dev/odd name"], env_vars={"FOO": "a b"}
        )
        assert config.to_command_string() == (
            "--device '/dev/odd name' --env 'FOO=a b'"
        )

    def test_empty_config_is_an_empty_string(self):
        assert ContainerDeviceConfig().to_command_string() == ""


def test_image_uri_plugin_emits_the_placeholder():
    from ray._private.runtime_env.context import RuntimeEnvContext
    from ray._private.runtime_env.image_uri import _modify_context_impl

    context = RuntimeEnvContext()
    _modify_context_impl(
        image_uri="podman://cuda-ray",
        worker_path="/worker.py",
        run_options=["--cpus", "1"],
        context=context,
        logger=logger,
        ray_tmp_dir="/tmp/ray",
    )
    # The placeholder must sit before the user's run options so that they keep the
    # last word, and before the image name so podman still parses it as a flag.
    command = context.py_executable.split()
    assert container_devices.DEVICE_ARGS_PLACEHOLDER in command
    assert command.index(container_devices.DEVICE_ARGS_PLACEHOLDER) < command.index(
        "--cpus"
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
