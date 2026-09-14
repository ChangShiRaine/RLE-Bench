#!/bin/bash
# Reference solution (Harbor Oracle) for method-agnostic.
set -eu

mkdir -p /logs/artifacts
cp -r /solution/payload/. /logs/artifacts/
echo "method-agnostic reference solution staged."
