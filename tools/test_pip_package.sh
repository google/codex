#!/usr/bin/env bash

set -ex  # Fail if any command fails, echo commands.

PYTHON_BIN="${1}"
WHEEL_PATH="${2}"

VENV="$(mktemp -d)" || exit 1
trap 'rm -rf -- "${VENV}"' EXIT

# Creates a clean virtual environment for test.
"${PYTHON_BIN}" -m venv "${VENV}"
source "${VENV}/bin/activate"
python --version

# Installs built pip package.
python -m pip install -U pip
python -m pip install -U "${WHEEL_PATH}"
python -m pip show -f jax-codex

# Runs unit tests.
python -m pip install -U pytest chex distrax
python -m pip list -v

pushd "${VENV}"
# `pytest --pyargs codex`` should work, but seems to cause recursion into
# /var/agentx/... on Darwin, which leads to a permission failure.
pytest lib/python*/site-packages/codex
popd

deactivate
