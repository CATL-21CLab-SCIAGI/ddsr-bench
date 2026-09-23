#!/usr/bin/env bash
set -euo pipefail
# Supply mounted DDSR source, job config, offline wheels, and dotenv paths.
repo=${DDSR_REPO:?set DDSR_REPO to the mounted DDSR checkout}
config=${DDSR_JOB_CONFIG:?set DDSR_JOB_CONFIG}
wheels=${DDSR_RUNTIME_WHEELS:?set DDSR_RUNTIME_WHEELS}
python=${DDSR_BASE_PYTHON:-python3} # Use PATH, or explicitly select an interpreter.
runtime=$(mktemp -d /tmp/ddsr-critpt-runtime.XXXXXX)
"$python" -m venv "$runtime"
"$runtime/bin/python" -m pip install --no-index --no-deps "$wheels"/*.whl
cd "$repo"
export PYTHONPATH="$repo"
command=("$runtime/bin/python" -u scripts/critpt/run_job.py --config "$config")
if [[ -n ${DDSR_ENV_FILE:-} ]]; then
    command+=(--env-file "$DDSR_ENV_FILE")
fi
if [[ ${DDSR_RESUME:-0} == 1 ]]; then
    command+=(--resume)
fi
# DSW uses direct routing for Alibaba endpoints; DLC containers usually have no proxy.
if command -v direct >/dev/null 2>&1; then
    exec direct "${command[@]}"
fi
exec "${command[@]}"
