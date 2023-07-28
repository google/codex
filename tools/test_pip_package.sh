#!/usr/bin/env bash

set -ex  # Fail if any command fails, echo commands.

PYTHON_BIN="${1}"
WHEEL_PATH="${2}"

# Creates clean virtual environment for test.
"${PYTHON_BIN}" -m venv "${TEMP}/codex-test-venv"
source "${TEMP}/codex-test-venv/bin/activate"
python --version

# Installs built pip package.
python -m pip install -U pip
python -m pip install -U "${WHEEL_PATH}"
python -m pip show -f jax-codex

# Runs unit tests.
python -m pip install -U pytest chex tensorflow-probability
python -m pip list -v

pushd "${TEMP}"
pytest --pyargs codex
popd

deactivate

