#!/bin/bash
# Reference solution: copy all three co-designed hardware/software payloads.
set -eu

mkdir -p /logs/artifacts
cp -r /solution/payload/. /logs/artifacts/
