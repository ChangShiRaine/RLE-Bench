#!/bin/bash
# Reference solution (Harbor Oracle) for rgb-only.
set -eu

mkdir -p /logs/artifacts
cp -r /solution/payload/. /logs/artifacts/
echo "rgb-only reference solution staged."
