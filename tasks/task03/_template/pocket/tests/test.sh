#!/bin/bash
# Harbor uploads this to /tests and runs it in the AGENT'S OWN container, as root.
#
# NOTHING SENSITIVE LIVES HERE, on purpose. This file is world-readable inside a
# container the agent also runs in, so it names paths and never numbers: the scorer,
# the thresholds and the evaluation seeds all sit in /opt/private (root:root 0700) and
# the ledger in /var/lib/rlebench (root:root 0700). The agent can read this script and
# learn nothing it did not already know.
#
# The explicit PYTHONPATH is the point: the scorer must be imported from the root-only
# tree, never from anywhere the agent's uid can write.
set -uo pipefail

mkdir -p /logs/verifier
/usr/local/bin/python -I -c 'import sys; sys.path.insert(0,"/opt/private"); from harness.verify_main import main; raise SystemExit(main())'
status=$?

if [ ! -f /logs/verifier/reward.json ]; then
    # The scorer died before writing. Emit an explicit zero rather than leaving Harbor
    # to guess. Flat and numeric: Harbor parses this into dict[str, float | int], and a
    # string or bool here would fail validation -- turning a bad run into a lost one.
    # The reason goes next door, where non-numeric fields are allowed.
    printf '{"reward": 0.0, "success_rate": 0.0, "component_outcome": 0.0, "component_efficiency": 0.0}\n' \
        > /logs/verifier/reward.json
    printf '{"ledger_ok": false, "ledger_reason": "scorer did not produce a reward file"}\n' \
        > /logs/verifier/diagnosis.json
fi
find /logs/verifier -type d -exec chmod a+rwx {} +
python -I - <<'PYVERIFY'
import json
from pathlib import Path
out = Path('/logs/verifier')
reward = json.loads((out / 'reward.json').read_text())
diagnosis = json.loads((out / 'diagnosis.json').read_text())
results = {**reward, **diagnosis}
results['qtm_complete'] = results.get('qtm_ok', False) and results.get('unclassified_transitions') == 0
for key in ('optimal_qtm', 'actual_qtm', 'unclassified_transitions'):
    results.setdefault(key, None)
counts = {key: reward[key] for key in ('optimal_qtm', 'actual_qtm', 'unclassified_transitions', 'recoveries') if key in reward}
if counts:
    (out / 'recovery.json').write_text(json.dumps(counts, indent=2) + '\n')
(out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
PYVERIFY
exit "${status}"
