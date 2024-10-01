#!/usr/bin/env bash

set -ex  # Fail if any command fails, echo commands.

PYTHON_BIN="${1}"
WORKSPACE_DIR="$(bazel info -c opt workspace)"

VENV="$(mktemp -d)" || exit 1
trap 'rm -rf -- "${VENV}"' EXIT

# Creates a clean virtual environment for build.
"${PYTHON_BIN}" -m venv "${VENV}"
source "${VENV}/bin/activate"
python --version

# Run all tests registered to bazel.
# Currently there is no test configured in bazel. The following line
# ends in error status if no test is found.
bazel test -c opt //... --test_output=all

# Build the C++ library file.
bazel build -c opt --verbose_failures //codex/cc:range_ans_pybind.so

PKGDIR="$(mktemp -d)" || exit 1
trap 'rm -rf -- "${PKGDIR}"' EXIT

# Copy files to a temp directory to build wheel.
cp "${WORKSPACE_DIR}/CONTRIBUTING.md" "${PKGDIR}"
cp "${WORKSPACE_DIR}/LICENSE" "${PKGDIR}"
cp "${WORKSPACE_DIR}/README.md" "${PKGDIR}"
cp "${WORKSPACE_DIR}/pyproject.toml" "${PKGDIR}"
cp -r "${WORKSPACE_DIR}/codex" "${PKGDIR}"

cp "$(bazel info -c opt bazel-genfiles)/codex/cc/range_ans_pybind.so" \
  "${PKGDIR}/codex/cc/"

python -m pip install -U flit pip setuptools wheel

pushd "${PKGDIR}"
flit build
mv dist "${WORKSPACE_DIR}"
popd

deactivate
