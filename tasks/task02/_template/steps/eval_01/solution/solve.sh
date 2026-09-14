#!/bin/bash
# Reference solution for ONE graded trial. Harbor's oracle runs this in place of a fresh
# agent, once per evaluation step.
#
# It does what the instruction tells an agent to do, in order: read the manual, look at the
# scene, import the inherited perception and controllers, perceive, act, hand back. That is
# the whole point of the oracle here -- the thing task02 has to prove works is the HANDOFF,
# and the handoff is exactly "a later step imports files an earlier step wrote".
#
# It will not solve the task. No scripted controller solves a RoboCasa composite task, and
# the stub library the develop step leaves behind does not try to. A near-zero score from
# the oracle is the expected result; a CRASH is not, and that is what this catches.
set -uo pipefail

HARNESS="${RLEBENCH_HARNESS_DIR:-/workspace/agent_harness}"

if [ -f "${HARNESS}/MANUAL.md" ]; then
    echo "[solve] manual found ($(wc -c < "${HARNESS}/MANUAL.md") bytes)"
else
    echo "[solve] WARNING: no manual at ${HARNESS}/MANUAL.md -- the development step did "
    echo "        not leave one, and a real agent would be working blind here"
fi

PYTHONPATH=/opt/rlebench:${HARNESS} python - <<'PY'
from harness.client import ToolsmithClient
from harness.controller import ObsSpec

with ToolsmithClient() as sim:
    if sim.phase() != "evaluation":
        print(f"[solve] phase is {sim.phase()!r}, not 'evaluation' -- nothing to drive")
        raise SystemExit(0)

    trial = sim.trial_info()
    if trial is None:
        print("[solve] no trial is open; it may already have been scored")
        raise SystemExit(0)
    print(f"[solve] trial {trial['index']} of {trial['total_trials']}")
    print(f"[solve] instruction: {trial['instruction']!r}")

    # Looking is free and this is the moment for it: a fresh agent knows nothing about
    # this kitchen.
    look = sim.observe(ObsSpec(width=256))
    print(f"[solve] observed at {look['resolution']}, "
          f"{len([k for k in look['obs'] if k.endswith('_image')])} cameras")

    # THE HANDOFF: perception and controllers written by a different agent, in a different
    # Harbor step, imported here from disk. If this import fails, the task has no handoff
    # at all. Both layers are tried, because a harness that hands over motion with no way
    # to aim it has handed over half of one.
    try:
        from controllers.primitives import nudge_forward, open_gripper, settle
        from perception.basic import aperture, is_settled
    except ImportError as exc:
        print(f"[solve] FAILED to import the inherited harness: {exc}")
        raise SystemExit(0)
    print("[solve] inherited perception and controllers imported")

    # An inherited primitive run BEFORE any step is spent: free, and the only way to check
    # a claim in the manual against this particular kitchen.
    free = sim.observe(ObsSpec(cameras=()))["obs"]
    print(f"[solve] before acting: aperture={aperture(free):.4f} "
          f"settled={is_settled(free)}")

    # The same controllers the development agent validated, called the same way.
    for name, run in (("settle", settle), ("open_gripper", open_gripper),
                      ("nudge_forward", nudge_forward)):
        res = run(sim)
        print(f"[solve] {name}: {res}")
        if res.get("ended"):
            print(f"[solve] the trial ended: {res['ended']}")
            break

    # Deliberately NOT calling sim.reset(). It is a give-up move, and the harness advances
    # to the next trial between steps -- there is nothing here for this agent to move on
    # to.
    print(f"[solve] finished with {sim.status()['seconds_remaining']}s left")
PY

# Never fail the step: a non-zero exit reads as a broken task rather than a reference
# solution that scored badly, and it would abort the remaining evaluation steps.
exit 0
