# 04 — Pocket cube

One scrambled 2×2 cube, two tilted Panda arms, a single front RGB camera with
optional depth, and 50,000 metered steps in one attempt. `recover_drop(sim)`
costs one step, restores the last legal pre-drop puzzle on the support, and
homes both arms. There is no phase split, no reset-based reroll, and no cube
state in the observations.

The verifier judges physical success and decays the quality by quarter-turn
overhead relative to the optimal solve; unclassified transitions cap the reward
at 0.5. No Oracle solution ships for this task.
