#!/bin/bash
# Harbor Oracle for task11 (full solution, declared reward 0.6047): stages
# the public-sensor baseline packer as the deliverable. It uses only the
# agent's observations; the reward scale leaves headroom above it.
set -euo pipefail
mkdir -p /logs/artifacts
rm -rf /logs/artifacts/policy
cp -r /solution/payload/policy /logs/artifacts/policy
echo "task11 baseline policy staged."
