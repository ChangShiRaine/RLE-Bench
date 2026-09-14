#!/bin/bash
set -euo pipefail
cd /workspace
python - <<'PYTHON'
import sys
sys.path.insert(0, "/solution")
from harness.client import TabletopClient
import oracle

with TabletopClient() as sim:
    oracle.solve(sim)
    print(sim.finish())
PYTHON
