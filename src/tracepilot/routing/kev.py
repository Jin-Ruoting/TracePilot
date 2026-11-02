"""Install or serve Kev-4B inside the project's restricted server launcher."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
import venv

from .decision import load_router_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "serve"))
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_router_config(args.config)
    try:
        code = Path(os.environ["TRACEPILOT_CODE_ROOT"])
        data = Path(os.environ["TRACEPILOT_DATA_ROOT"])
    except KeyError:
        parser.error("run this command through scripts/agent.sh")
    checkout = code / "services" / "kev"
    environment = data / "services" / "kev" / "env"
    python = environment / "bin" / "python"
    directories = [checkout.parent, environment.parent, data / "models" / "huggingface",
                   data / "cache" / "triton", data / "cache" / "torch-extensions"]
    for directory in directories + [checkout, environment]:
        if not directory.is_absolute() or directory.resolve() != directory:
            raise ValueError("service directories must be absolute and contain no symlinks")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    if args.action == "install":
        if not checkout.exists():
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                            "--branch", config["source_ref"], "https://github.com/jaredpalmer/kev.git",
                            str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", "kev"], check=True)
        else:
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin",
                            config["source_ref"]], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"], check=True)
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(environment)
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade",
                        "-e", str(checkout) + "[serve]"], check=True)
        print("Kev service code and dependencies installed; no tests or inference requests were run.")
        return

    if not python.exists():
        raise RuntimeError("install Kev service dependencies first")
    if not os.environ.get("TRACEPILOT_GPU_UUID"):
        raise RuntimeError("serve Kev-4B through scripts/agent.sh --gpu")
    endpoint = urlsplit(config["endpoint"])
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"} \
            or endpoint.path != "/v1/systemone":
        raise ValueError("the managed Kev service must use a local HTTP /v1/systemone endpoint")
    child_env = dict(os.environ)
    child_env.update({
        "PATH": str(environment / "bin") + ":" + os.environ.get("PATH", ""),
        "HF_HUB_CACHE": str(data / "models" / "huggingface"),
        "TRITON_CACHE_DIR": str(data / "cache" / "triton"),
        "TORCH_EXTENSIONS_DIR": str(data / "cache" / "torch-extensions"),
        "TOKENIZERS_PARALLELISM": "false", "KEV_BACKEND": "torch", "KEV_DTYPE": "bf16",
        "KEV_CUDA_GRAPHS": "0", "KEV_FUSED": "0", "KEV_PREFIX_CACHE": "1",
        "KEV_PREFIX_MAX_TOKENS": "8192",
    })
    if config["api_key_env"] in os.environ:
        child_env["KEV_API_KEY"] = os.environ[config["api_key_env"]]
    command = [str(python), "-B", "-m", "kev.serve", "--run", config["checkpoint"],
               "--host", endpoint.hostname, "--port", str(endpoint.port or 8009), "--device", "cuda"]
    os.execve(str(python), command, child_env)


if __name__ == "__main__":
    main()
