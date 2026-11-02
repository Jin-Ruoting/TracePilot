"""Linux launcher: two writable project roots, bounded CPUs, and zero or one GPU."""
from __future__ import annotations

import argparse
import errno
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimePolicy:
    code_root: Path
    data_root: Path
    cpu_ids: tuple[int, ...]
    gpu_uuid: str
    conda_executable: str

    @classmethod
    def load(cls, path: str | Path) -> RuntimePolicy:
        raw = json.loads(Path(path).read_text())
        roots = []
        for name in ("code_root", "data_root"):
            original = Path(raw[name]).expanduser()
            if not original.is_absolute() or len(original.parts) < 4:
                raise ValueError("each root must be a dedicated absolute project subdirectory")
            resolved = original.resolve()
            if resolved != original:
                raise ValueError("project roots must not contain symlinks")
            roots.append(resolved)
        if roots[0].is_relative_to(roots[1]) or roots[1].is_relative_to(roots[0]):
            raise ValueError("code and data roots must be separate")
        cpus = raw["cpu_ids"]
        if not cpus or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in cpus) \
                or len(cpus) != len(set(cpus)):
            raise ValueError("cpu_ids must be unique nonnegative integers")
        gpu = raw["gpu_uuid"]
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
            raise ValueError("configure exactly one GPU UUID")
        return cls(*roots, tuple(sorted(cpus)), gpu, raw["conda_executable"])

    @property
    def environment_prefix(self) -> Path:
        return self.data_root / "conda" / "envs" / "agent"

    def environment(self, *, gpu: bool = False) -> dict[str, str]:
        code, data = self.code_root, self.data_root
        values = {
            "HOME": str(code / "home"), "XDG_CONFIG_HOME": str(code / "config" / "xdg"),
            "XDG_CACHE_HOME": str(data / "cache"), "XDG_DATA_HOME": str(data / "runtime" / "share"),
            "XDG_STATE_HOME": str(data / "runtime" / "state"),
            "TMPDIR": str(data / "tmp"), "TEMP": str(data / "tmp"), "TMP": str(data / "tmp"),
            "CONDA_ENVS_PATH": str(data / "conda" / "envs"), "CONDA_PKGS_DIRS": str(data / "conda" / "pkgs"),
            "PIP_CACHE_DIR": str(data / "cache" / "pip"), "HF_HOME": str(data / "cache" / "huggingface"),
            "TORCH_HOME": str(data / "cache" / "torch"), "CUDA_CACHE_PATH": str(data / "cache" / "cuda"),
            "MPLCONFIGDIR": str(data / "cache" / "matplotlib"), "NUMBA_CACHE_DIR": str(data / "cache" / "numba"),
            "UV_CACHE_DIR": str(data / "cache" / "uv"), "JOBLIB_TEMP_FOLDER": str(data / "tmp" / "joblib"),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(code / "repo" / "src"), "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": str(len(self.cpu_ids)),
            "CUDA_VISIBLE_DEVICES": self.gpu_uuid if gpu else "",
            "TRACEPILOT_GPU_UUID": self.gpu_uuid if gpu else "",
            "TRACEPILOT_CPU_LIMIT": str(len(self.cpu_ids)),
            "TRACEPILOT_STATE_ROOT": str(code / "workspaces"),
            "TRACEPILOT_ARTIFACTS_ROOT": str(data / "experiments"),
            "TRACEPILOT_DATA_ROOT": str(data),
            "TRACEPILOT_CODE_ROOT": str(code),
            "TRACEPILOT_ROUTER_CONFIG": os.environ.get("TRACEPILOT_ROUTER_CONFIG", str(code / "config" / "router.local.json")),
            "PATH": str(self.environment_prefix / "bin") + ":" + str(Path(self.conda_executable).parent)
                    + ":/usr/local/bin:/usr/bin:/bin",
        }
        return values

    def prepare(self) -> None:
        if not self.code_root.parent.is_dir() or not self.data_root.parent.is_dir():
            raise ValueError("the authorized parent directories must already exist")
        directories = [
            self.code_root, self.code_root / "config", self.code_root / "home",
            self.code_root / "workspaces", self.code_root / "config" / "xdg",
            self.data_root, self.data_root / "tmp", self.data_root / "tmp" / "joblib",
            self.data_root / "logs", self.data_root / "experiments",
            self.data_root / "datasets", self.data_root / "checkpoints",
            self.data_root / "conda" / "envs", self.data_root / "conda" / "pkgs",
            self.data_root / "runtime" / "share", self.data_root / "runtime" / "state",
        ]
        directories += [self.data_root / "cache" / name for name in
                        ("pip", "huggingface", "torch", "cuda", "matplotlib", "numba", "uv")]
        # Validate every existing parent before any mkdir; a planted symlink must not redirect writes.
        for directory in directories:
            if directory.resolve() != directory:
                raise ValueError("project subdirectory contains a symlink: " + str(directory))
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def physical_cores(cpu_ids: set[int]) -> set[tuple[str, str]]:
    cores = set()
    for cpu in cpu_ids:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        cores.add(((root / "physical_package_id").read_text().strip(),
                   (root / "core_id").read_text().strip()))
    return cores


def check_cpus(policy: RuntimePolicy) -> None:
    allowed = set(os.sched_getaffinity(0))
    requested = set(policy.cpu_ids)
    if not requested <= allowed:
        raise ValueError("configured CPUs are outside this session's permitted CPU set")
    available_cores = physical_cores(allowed)
    if len(physical_cores(requested)) > len(available_cores) // 2:
        raise ValueError("CPU request exceeds half the available physical cores")
    if len(requested) > len(allowed) // 2:
        raise ValueError("CPU request exceeds half the available logical CPUs")


def affinity_filter() -> bytes:
    """x86-64 seccomp filter denies changing CPU affinity, including via x32 syscalls."""
    instructions = [
        (0x20, 0, 0, 4),              # seccomp_data.arch
        (0x15, 1, 0, 0xC000003E),     # AUDIT_ARCH_X86_64
        (0x06, 0, 0, 0x80000000),     # KILL_PROCESS for another syscall ABI
        (0x20, 0, 0, 0),              # syscall number
        (0x54, 0, 0, 0xBFFFFFFF),     # strip x32 syscall bit
        (0x15, 0, 1, 203),            # __NR_sched_setaffinity
        (0x06, 0, 0, 0x00050000 | errno.EPERM),
        (0x06, 0, 0, 0x7FFF0000),     # ALLOW
    ]
    return b"".join(struct.pack("HBBI", *instruction) for instruction in instructions)


def gpu_devices(policy: RuntimePolicy) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=10)
    matches = []
    for line in result.stdout.splitlines():
        index, name, uuid = [part.strip() for part in line.split(",")]
        if uuid == policy.gpu_uuid:
            if "RTX 5090" not in name or not index.isdigit():
                raise ValueError("configured GPU is not an RTX 5090")
            matches.append("/dev/nvidia" + index)
    if len(matches) != 1:
        raise ValueError("configured GPU UUID is absent or ambiguous")
    common = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]
    return matches + [path for path in common if Path(path).exists()]


def sandbox_command(policy: RuntimePolicy, argv: list[str], *, filter_fd: int,
                    devices: list[str] | None = None) -> list[str]:
    if not argv:
        raise ValueError("a command is required")
    if len([device for device in devices or [] if re.fullmatch(r"/dev/nvidia\d+", device)]) > 1:
        raise ValueError("at most one GPU device may be exposed")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap is required; refusing unrestricted execution")
    command = [
        bwrap, "--ro-bind", "/", "/", "--bind", str(policy.code_root), str(policy.code_root),
        "--bind", str(policy.data_root), str(policy.data_root),
        "--proc", "/proc", "--dev", "/dev", "--unshare-pid", "--unshare-ipc",
        "--unshare-uts", "--new-session", "--cap-drop", "ALL", "--seccomp", str(filter_fd),
        "--chdir", str(policy.code_root / "repo" if (policy.code_root / "repo").is_dir() else policy.code_root),
    ]
    for device in devices or []:
        command.extend(["--dev-bind", device, device])
    return command + ["--"] + argv


def launch(policy: RuntimePolicy, argv: list[str], *, gpu: bool = False) -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("this restricted server launcher requires x86-64 Linux")
    check_cpus(policy)
    policy.prepare()
    devices = gpu_devices(policy) if gpu else []
    # Some conda Python builds omit memfd_create; an unlinked temporary file
    # supplies the same inherited seccomp descriptor within the data directory.
    with tempfile.TemporaryFile(dir=policy.data_root / "tmp") as stream:
        stream.write(affinity_filter())
        stream.flush()
        stream.seek(0)
        fd = stream.fileno()
        os.set_inheritable(fd, True)
        environment = dict(os.environ, **policy.environment(gpu=gpu))
        os.sched_setaffinity(0, policy.cpu_ids)
        command = sandbox_command(policy, argv, filter_fd=fd, devices=devices)
        os.execvpe(command[0], command, environment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", action="store_true", help="Expose only the configured RTX 5090; default is CPU only.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    launch(RuntimePolicy.load(args.config), command, gpu=args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
