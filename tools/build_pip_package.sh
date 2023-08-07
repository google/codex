#!/usr/bin/env bash

set -ex  # Fail if any command fails, echo commands.

PYTHON_BIN="${1}"

# Creates clean virtual environment for build.
"${PYTHON_BIN}" -m venv "${TEMP}/codex-build-venv"
source "${TEMP}/codex-build-venv/bin/activate"
python --version

# Run all tests registered to bazel.
# Currently there is no test configured in bazel. The following line
# ends in error status if no test is found.
# bazel test -c opt //... --test_output=all

# Fetch the C++ library file.
bazel build -c opt //codex/cc:range_ans_pybind.so
cp $(bazel info -c opt bazel-genfiles)/codex/cc/range_ans_pybind.so \
  $(bazel info -c opt workspace)/codex/cc/
chmod =rw,+X $(bazel info -c opt workspace)/codex/cc/range_ans_pybind.so

# Clean up bazel files.
bazel clean

# flit may complain if there is an untracked file.
git add codex/cc/range_ans_pybind.so

python -m pip install -U flit pip setuptools wheel
flit build

deactivate

