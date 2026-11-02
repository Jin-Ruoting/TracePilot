#!/usr/bin/env bash
set -euo pipefail

tracepilot_scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tracepilot_root="$(cd "$tracepilot_scripts/../.." && pwd)"
tracepilot_config="${TRACEPILOT_RUNTIME_CONFIG:-$tracepilot_root/config/runtime.local.json}"
tracepilot_python="$(python3 -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_root"] + "/conda/envs/agent/bin/python")' "$tracepilot_config")"
exec "$tracepilot_python" -B "$tracepilot_scripts/../src/tracepilot/runtime.py" --config "$tracepilot_config" "$@"
