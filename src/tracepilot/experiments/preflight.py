"""Import checks run with the actual experiment interpreter before enqueuing a batch."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any

PROGRAM = """
import importlib, json, platform, sys
modules = json.loads(sys.argv[1])
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, '__version__', 'unknown'))
print(json.dumps({'python': platform.python_version(), 'executable': sys.executable, 'modules': versions}))
"""


def preflight(python: str, modules: list[str]) -> dict[str, Any]:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([python, "-c", PROGRAM, json.dumps(modules)], capture_output=True, text=True,
                            timeout=30, env=environment)
    if result.returncode:
        raise RuntimeError("Experiment environment import preflight failed:\n" + result.stderr[-3500:])
    return json.loads(result.stdout)
