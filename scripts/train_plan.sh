#!/usr/bin/env bash
set -euo pipefail

exec "${PYTHON_BIN:-python}" -m anima_reflora.train_plan "$@"
