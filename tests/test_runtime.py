from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tracepilot.runtime import RuntimePolicy, affinity_filter, check_cpus, gpu_devices, sandbox_command


class RuntimePolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.raw = {
            "code_root": str(self.root / "code" / "project"), "data_root": str(self.root / "data" / "project"),
            "cpu_ids": [0, 1], "gpu_uuid": "GPU-1234-abcd", "conda_executable": "/opt/conda/bin/conda",
        }
        (self.root / "code").mkdir()
        (self.root / "data").mkdir()
        self.path = self.root / "policy.json"
        self.path.write_text(json.dumps(self.raw))
        self.policy = RuntimePolicy.load(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_cache_and_environment_locations_respect_data_root(self):
        self.policy.prepare()
        env = self.policy.environment()
        for name in ("TMPDIR", "CONDA_ENVS_PATH", "CONDA_PKGS_DIRS", "PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME",
                     "MPLCONFIGDIR", "TRACEPILOT_ARTIFACTS_ROOT"):
            self.assertTrue(Path(env[name]).is_relative_to(self.policy.data_root), name)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(self.policy.environment(gpu=True)["CUDA_VISIBLE_DEVICES"], "GPU-1234-abcd")
        self.assertEqual(self.policy.environment_prefix.name, "agent")

    def test_nested_or_symlinked_roots_are_rejected(self):
        self.raw["data_root"] = self.raw["code_root"] + "/data"
        self.path.write_text(json.dumps(self.raw))
        with self.assertRaises(ValueError):
            RuntimePolicy.load(self.path)
        (self.root / "linked").symlink_to(self.root / "data", target_is_directory=True)
        self.raw["data_root"] = str(self.root / "linked" / "project")
        self.path.write_text(json.dumps(self.raw))
        with self.assertRaises(ValueError):
            RuntimePolicy.load(self.path)

    def test_existing_subdirectory_cannot_redirect_writes(self):
        self.policy.code_root.mkdir()
        self.policy.data_root.mkdir()
        (self.policy.data_root / "cache").symlink_to(self.root / "code", target_is_directory=True)
        with self.assertRaises(ValueError):
            self.policy.prepare()
        self.assertFalse((self.root / "code" / "pip").exists())

    def test_cpu_cap_checks_physical_and_logical_limits(self):
        def cores(cpus):
            return {("0", str(cpu)) for cpu in cpus}
        with patch("tracepilot.runtime.os.sched_getaffinity", create=True, return_value=set(range(4))), \
                patch("tracepilot.runtime.physical_cores", side_effect=cores):
            check_cpus(self.policy)
        with patch("tracepilot.runtime.os.sched_getaffinity", create=True, return_value={0, 1, 2}), \
                patch("tracepilot.runtime.physical_cores", side_effect=cores):
            with self.assertRaises(ValueError):
                check_cpus(self.policy)

    def test_sandbox_has_only_two_writable_host_mounts_and_one_gpu_maximum(self):
        with patch("tracepilot.runtime.shutil.which", return_value="/usr/bin/bwrap"):
            command = sandbox_command(self.policy, ["python", "--version"], filter_fd=12)
            self.assertEqual(command.count("--bind"), 2)
            self.assertIn("--ro-bind", command)
            self.assertIn("--seccomp", command)
            self.assertNotIn("--dev-bind", command)
            with self.assertRaises(ValueError):
                sandbox_command(self.policy, ["python"], filter_fd=12, devices=["/dev/nvidia0", "/dev/nvidia1"])
        with patch("tracepilot.runtime.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                sandbox_command(self.policy, ["python"], filter_fd=12)

    def test_non_5090_or_unknown_uuid_is_rejected(self):
        class Result:
            stdout = "0, NVIDIA GeForce RTX 4090, GPU-1234-abcd\n"
        with patch("tracepilot.runtime.subprocess.run", return_value=Result()):
            with self.assertRaises(ValueError):
                gpu_devices(self.policy)
        Result.stdout = "0, NVIDIA GeForce RTX 5090, GPU-9999\n"
        with patch("tracepilot.runtime.subprocess.run", return_value=Result()):
            with self.assertRaises(ValueError):
                gpu_devices(self.policy)

    def test_affinity_filter_denies_native_and_x32_affinity_changes(self):
        rows = list(struct.iter_unpack("HBBI", affinity_filter()))
        def execute(architecture, syscall):
            pc, accumulator = 0, 0
            while pc < len(rows):
                code, jt, jf, operand = rows[pc]
                if code == 0x20:
                    accumulator = architecture if operand == 4 else syscall
                elif code == 0x54:
                    accumulator &= operand
                elif code == 0x15:
                    pc += jt if accumulator == operand else jf
                elif code == 0x06:
                    return operand
                pc += 1
            self.fail("filter has no terminal instruction")
        self.assertEqual(execute(0xC000003E, 203), 0x00050001)
        self.assertEqual(execute(0xC000003E, 203 | 0x40000000), 0x00050001)
        self.assertEqual(execute(0xC000003E, 1), 0x7FFF0000)
        self.assertEqual(execute(0x40000003, 241), 0x80000000)


if __name__ == "__main__":
    unittest.main()
