#!/bin/bash
# Score the root-owned ledger in the shared container.
set -uo pipefail

mkdir -p /logs/verifier
/usr/local/bin/python -I -c 'import sys; sys.path.insert(0, "/opt/private"); from harness.verify_main import main; raise SystemExit(main())'
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
exit "${status}"
