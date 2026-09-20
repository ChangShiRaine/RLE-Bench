"""Every task02 default lives here. Nothing numeric or task-named is hardcoded elsewhere.

Each value below is the DEFAULT; the ones a deployment tunes are overridable from
`task.toml` (see `tasks/task02/README.md` for the knobs and the channel each travels on).

TASK02 IS A TRANSFER TASK, and this module defines the transfer: one RoboCasa activity
group, one member held out for grading, and `TRAIN_SET_SIZE` others CHOSEN -- not merely
left over -- so that every atomic primitive the held-out task requires is practised. Both
splits are composite tasks, so the held-out one recombines what the others exercise.
`check_split()` proves the shape; `ACTIVITY_GROUP` names the group; the coverage claim is
proved at build time by `tasks/task02/build_groups.py`.

Measured anchors (do not re-derive):
  step        ~38 ms      GPU EGL, cameras essentially free
  reset       ~4.8 s      steady state; the first reset of a process costs ~17 s
  horizon     1000        RoboCasa default, uniform across all 317 tasks
  action_dim  12          uniform across all 317 tasks
"""

from __future__ import annotations

import hashlib
import os
import re


# --- reading overrides out of the environment --------------------------------
# `task.toml` has no generic task-parameter mechanism, so every knob travels as an
# environment variable. Two channels, and the choice decides who may read it:
#
#   [environment.env]  reaches the whole container INCLUDING the agent, which can read
#                      /proc/self/environ. Fine for what the agent is told anyway --
#                      its budgets, its clocks, its training tasks.
#   [verifier.env]     reaches the verifier's exec environment only.
#
# NOTHING SECRET TRAVELS EITHER WAY. The splits are constants below, not configuration,
# because `docker exec` builds an agent's environment from the container's config and no
# entrypoint can scrub that.
#
# Malformed values raise rather than falling back silently: a run that quietly ignored its
# own configuration would be scored under rules nobody chose.
def _env(name: str, default, cast, kind: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be {kind}, got {raw!r}") from None


def env_int(name: str, default: int) -> int:
    return _env(name, default, int, "an integer")


def env_float(name: str, default: float) -> float:
    return _env(name, default, float, "a number")


# =============================================================================
# THE SPLITS. One table for every shipped group, a constant in the root-only tree, so
# the held-out names never travel by any run-time channel. RLEBENCH_GROUP selects the
# entry; it names only the group, which the agent is told anyway. Regenerate with
#     python tasks/task02/build_groups.py --emit-all
# which rewrites everything between the BEGIN/END markers below.
# =============================================================================

# How many members of the activity group the agent practises on.
#
# THE SET IS CHOSEN, NOT LEFT OVER: `Group.train` in tasks/task02/build_groups.py names
# the members explicitly, and they cover every primitive the held-out task requires. Small
# enough that each is attempted often enough for its success rate to be legible rather than
# a single result; more than one, because the task's premise is factoring reusable pieces
# out of SEVERAL long procedures.
#
# Changing this means re-choosing every group's set:
#     .venv-robocasa/bin/python tasks/task02/build_groups.py --recompute-split
TRAIN_SET_SIZE = 3

# --- BEGIN GENERATED SPLIT (tasks/task02/build_groups.py) ---

SPLITS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    # Washing Dishes: 17 members, 3 trained on.
    # Train: PlaceOnDishRack 2, SortingCleanup 5, StackBowlsInSink 2. Held out: DumpLeftovers 3.
    "01-washing-dishes": (
        "Washing Dishes",
        ("PlaceOnDishRack", "SortingCleanup", "StackBowlsInSink"),
        ("DumpLeftovers",),
    ),
    # Sauteing Vegetables: 7 members, 3 trained on.
    # Train: AdjustHeat 3, StirVegetables 4, TiltPan 2. Held out: PlaceVegetablesEvenly 2.
    "02-sauteing-vegetables": (
        "Sauteing Vegetables",
        ("AdjustHeat", "StirVegetables", "TiltPan"),
        ("PlaceVegetablesEvenly",),
    ),
    # Baking: 6 members, 3 trained on.
    # Train: CookieDoughPrep 7, CoolBakedCake 8, CupcakeCleanup 2. Held out: PastryDisplay 2.
    "03-baking": (
        "Baking",
        ("CookieDoughPrep", "CoolBakedCake", "CupcakeCleanup"),
        ("PastryDisplay",),
    ),
    # Reheating Food: 5 members, 3 trained on.
    # Train: KettleBoiling (from another activity), PanTransfer (from another activity), WarmCroissant 2. Held out: SimmeringSauce 4.
    "04-reheating-food": (
        "Reheating Food",
        ("KettleBoiling", "PanTransfer", "WarmCroissant"),
        ("SimmeringSauce",),
    ),
    # Chopping Food: 5 members, 3 trained on.
    # Train: BreadSetupSlicing 2, MeatTransfer 3, OrganizeVegetables 2. Held out: ClearCuttingBoard 3.
    "05-chopping-food": (
        "Chopping Food",
        ("BreadSetupSlicing", "MeatTransfer", "OrganizeVegetables"),
        ("ClearCuttingBoard",),
    ),
    # Setting the Table: 13 members, 3 trained on.
    # Train: ArrangeDrinkware 7, BeverageOrganization 7, SetupWineGlasses 7. Held out: AlignSilverware 7.
    "06-setting-the-table": (
        "Setting the Table",
        ("ArrangeDrinkware", "BeverageOrganization", "SetupWineGlasses"),
        ("AlignSilverware",),
    ),
    # Portioning Meals: 7 members, 3 trained on.
    # Train: DistributeChicken 7, PortionOnSize 7, ScalePortioning 6. Held out: PortionHotDogs 4.
    "07-portioning-meals": (
        "Portioning Meals",
        ("DistributeChicken", "PortionOnSize", "ScalePortioning"),
        ("PortionHotDogs",),
    ),
    # Defrosting Food: 6 members, 3 trained on.
    # Train: MoveToCounter 3, QuickThaw 2, ThawInSink 2. Held out: DefrostByCategory 4.
    "08-defrosting-food": (
        "Defrosting Food",
        ("MoveToCounter", "QuickThaw", "ThawInSink"),
        ("DefrostByCategory",),
    ),
    # Arranging Buffet: 5 members, 3 trained on.
    # Train: ArrangeBuffetDessert 8, DivideBuffetTrays 16, TongBuffetSetup 4. Held out: PlaceBeveragesTogether 11.
    "09-arranging-buffet": (
        "Arranging Buffet",
        ("ArrangeBuffetDessert", "DivideBuffetTrays", "TongBuffetSetup"),
        ("PlaceBeveragesTogether",),
    ),
    # Serving Beverages: 6 members, 3 trained on.
    # Train: AlcoholServingPrep 8, PrepareCocktailStation 12, PrepareDrinkStation 11. Held out: MatchCupAndDrink 7.
    "10-serving-beverages": (
        "Serving Beverages",
        ("AlcoholServingPrep", "PrepareCocktailStation", "PrepareDrinkStation"),
        ("MatchCupAndDrink",),
    ),
    # Loading Fridge: 8 members, 3 trained on.
    # Train: LoadCondimentsInFridge 9, LoadPreparedFood 4, PlaceVeggiesInDrawer 8. Held out: MoveFreezerToFridge 2.
    "11-loading-fridge": (
        "Loading Fridge",
        ("LoadCondimentsInFridge", "LoadPreparedFood", "PlaceVeggiesInDrawer"),
        ("MoveFreezerToFridge",),
    ),
    # Managing Freezer Space: 8 members, 3 trained on.
    # Train: ClearFreezer 11, MoveFridgeToFreezer 2, ReorganizeFrozenVegetables 3. Held out: SeparateFreezerRack 7.
    "12-managing-freezer-space": (
        "Managing Freezer Space",
        ("ClearFreezer", "MoveFridgeToFreezer", "ReorganizeFrozenVegetables"),
        ("SeparateFreezerRack",),
    ),
    # Clearing Table: 7 members, 3 trained on.
    # Train: ClearReceptaclesForCleaning 8, CondimentCollection 2, FoodCleanup 2. Held out: CandleCleanup 8.
    "13-clearing-table": (
        "Clearing Table",
        ("ClearReceptaclesForCleaning", "CondimentCollection", "FoodCleanup"),
        ("CandleCleanup",),
    ),
    # Microwaving Food: 6 members, 3 trained on.
    # Train: FilterMicrowavableItem 6, ReheatMeal 5, ReturnHeatedFood 4. Held out: PlaceMicrowaveSafeItem 3.
    "14-microwaving-food": (
        "Microwaving Food",
        ("FilterMicrowavableItem", "ReheatMeal", "ReturnHeatedFood"),
        ("PlaceMicrowaveSafeItem",),
    ),
    # Storing Leftovers: 5 members, 3 trained on.
    # Train: FreezeCookedFood 4, PrepareStoringLeftovers 7, StoreDumplings 11. Held out: StoreLeftoversInBowl 5.
    "15-storing-leftovers": (
        "Storing Leftovers",
        ("FreezeCookedFood", "PrepareStoringLeftovers", "StoreDumplings"),
        ("StoreLeftoversInBowl",),
    ),
}

# --- END GENERATED SPLIT ---


def _group() -> str:
    """The slug this container runs, from RLEBENCH_GROUP; the first entry when unset
    (the dev suite). Malformed values raise: a run that quietly graded another group
    would be worse than one that failed to start."""
    raw = os.environ.get("RLEBENCH_GROUP", "").strip() or next(iter(SPLITS))
    if raw not in SPLITS:
        raise ValueError(f"RLEBENCH_GROUP must be one of {', '.join(SPLITS)}, got {raw!r}")
    return raw


ACTIVITY_GROUP, TRAIN_TASKS, EVAL_TASKS = SPLITS[_group()]


def check_split(train: tuple[str, ...] = None,
                evaluation: tuple[str, ...] = None) -> None:
    """Prove the split is what the rule says, or raise.

    One group, one member held out, TRAIN_SET_SIZE others chosen to cover the primitives
    it needs -- so the properties worth asserting are shape and disjointness. The coverage
    itself is proved at build time, against the audit in tasks/task02/dev/data/task02_primitive_audit.json,
    because deciding it here would need RoboCasa source at daemon startup.

    The daemon calls this before accepting connections, so a mis-staged image fails at boot
    rather than after an agent has spent its phase on it.
    """
    train = TRAIN_TASKS if train is None else train
    evaluation = EVAL_TASKS if evaluation is None else evaluation

    if len(evaluation) != 1:
        raise ValueError(
            f"expected exactly one evaluation task, got {len(evaluation)}: the task is "
            "graded on a single held-out member of the activity group")
    overlap = set(train) & set(evaluation)
    if overlap:
        raise ValueError(
            f"training and evaluation splits overlap on {sorted(overlap)}; a held-out "
            "task that was practised measures memorisation")
    if len(train) != TRAIN_SET_SIZE:
        raise ValueError(
            f"the training split has {len(train)} tasks, expected TRAIN_SET_SIZE="
            f"{TRAIN_SET_SIZE}; the image was staged from a table that disagrees with it")
    if len(set(train)) != len(train):
        raise ValueError(f"the training split repeats a task: {sorted(train)}")


def check_registry() -> None:
    """Both splits are real RoboCasa COMPOSITE tasks. Needs RoboCasa importable, so it is
    kept apart from check_split, which the daemon calls at startup."""
    from robocasa.utils.dataset_registry import COMPOSITE_TASK_DATASETS

    missing = [t for t in TRAIN_TASKS + EVAL_TASKS if t not in COMPOSITE_TASK_DATASETS]
    if missing:
        raise ValueError(f"not composite RoboCasa tasks: {missing}")


# --- budgets ----------------------------------------------------------------
# DEVELOPMENT INTERACTION, in env steps. Sized against TRAIN_SET_SIZE: the two together
# decide how many attempts each training task gets.
#
# A HARD CAP AND NOTHING ELSE: no reward for frugality, since the question is whether the
# harness transfers, not how cheaply it was built. Requests past the cap are refused
# rather than silently truncated. Override with RLEBENCH_INTERACTION_STEPS.
#
# FOR WHOEVER TUNES THIS: end-to-end throughput is an order of magnitude below a bare
# `env.step`, because rendering, transfer, env construction and the agent's own deliberation
# all land on the wall clock. So a phase clock set too low will expire before this budget is
# spent, which silently changes what the task measures. If a run ends with budget unspent,
# raise RLEBENCH_DEVELOP_SECONDS and [steps.agent] timeout_sec together -- do not read it as
# the agent having chosen to stop early.
INTERACTION_STEPS = 75_000

# How many charged steps may accrue before an `interact` ledger record is written. The
# figure is only ever summed and split by task, so batching costs nothing and keeps the
# ledger readable; every phase and episode boundary flushes regardless.
LEDGER_FLUSH_STEPS = 500

# Per-trial step ceiling. The env never signals done -- create_env sets ignore_done=True
# and robosuite computes `done = (timestep >= horizon) and not ignore_done` -- so this cap
# is what actually ends a trial. Override with RLEBENCH_MAX_STEPS_PER_TRIAL.
#
# Above RoboCasa's own default, which is a teleop-demo horizon rather than a budget for an
# agent that must perceive, probe and correct. Sized to hold a whole composite procedure:
# an evaluation task's clauses sit at different fixtures, and the drives between them cost
# more than the manipulation at either end, so a ceiling that fits only the grasps ends
# trials mid-traversal with clauses never attempted.
#
# The wall clock (RLEBENCH_TRIAL_SECONDS) is the other ceiling, and which of the two binds
# first is a property of the run, not a guarantee -- measured trials have hit either. The
# two are sized together; raising one alone only moves which of them ends the trial.
MAX_STEPS_PER_TRIAL = 5_000

# The DEVELOPMENT episode ceiling. Bound to the trial ceiling, not merely equal to it: a
# practice episode must hold a whole composite procedure, because that is the regime the
# agent inheriting the harness will be driving, and a shorter one makes "do these pieces
# chain?" unanswerable.
MAX_EPISODE_STEPS = MAX_STEPS_PER_TRIAL

# How many envs stay resident during development. Construction costs ~20-40 s and an env
# is not small, so hopping between tasks is bounded here rather than refused: the
# least-recently-used env is closed. Override with RLEBENCH_ENV_CACHE.
ENV_CACHE_SIZE = 3

# --- observation resolution --------------------------------------------------
# What an unsized observation delivers, and the ceiling `observe()` may ask for. Two
# numbers because they trade against different things: an unsized observation is on the
# hot path and wants to stay small, while `observe()` costs no interaction budget and may
# ask for detail. Requests above the ceiling are clamped, not refused.
#
# THIS IS PART OF THE OBSERVATION CONTRACT, not a deployment knob -- the agent is graded
# against what it can see, so a run that quietly rendered something else would not be
# comparable. Change it here, in source, and rebuild.
OBS_RESOLUTION = 256
OBS_MAX_RESOLUTION = 512

# EVERY camera is baked at the ceiling, and nothing ever asks MuJoCo for more. Growing
# the offscreen framebuffer mid-run makes robosuite free and rebuild the LIVE GL context
# (`binding_utils.update_offscreen_size`); that rebuild is the only operation in this
# harness that ever logged `OpenGL error 0x501 in or before mjr_makeContext`, and twice
# it left a context that killed the daemon outright or wedged it at 100% CPU, losing the
# trials behind it. Baking at the ceiling deletes the path: a SMALLER render never
# resizes the buffer, and a smaller delivery is a resample of a frame already in hand.
RENDER_RESOLUTION = OBS_MAX_RESOLUTION

# --- scene splits ------------------------------------------------------------
# RoboCasa's own train/test boundary, honoured verbatim: development draws scenes and
# object instances from "pretrain", evaluation from the held-out "target". The published
# protocol, not a knob, which is why no caller may choose it.
DEV_SPLIT = "pretrain"
EVAL_SPLIT = "target"

# How many graded trials each evaluation task gets. Several trials on one held-out task
# measure whether the harness transfers repeatably rather than once.
#
# NOT OVERRIDABLE FROM THE ENVIRONMENT, unlike the budgets above. One trial is one Harbor
# step is one fresh agent, so this number IS the number of `eval_NN` steps in task.toml,
# and task.toml's step list is static. A run-time override would desync the two: too high
# and the last trials are never reached and score 0, too low and the surplus `next-trial`
# hooks run past the end of the plan. Change it with
#     python tasks/task02/build_groups.py --trials N
# which rewrites this line, the step directories and the [[steps]] list together.
TRIALS_PER_TASK = 5

# --- seed derivation --------------------------------------------------------
# Trial seeds are DERIVED, never published: this module ships only in the root-only tree
# and a trial reply reports no seed, so the graded episodes cannot be enumerated. The salt
# keeps trials private even if the code is public.
# FROZEN: the literal string feeds the hidden eval seeds; renaming it re-rolls every seed.
_EVAL_DOMAIN = "rlebench.task02.eval"
_SEED_MODULUS = 2**31 - 1


def eval_plan(tasks: tuple[str, ...] = EVAL_TASKS,
              salt: str | None = None,
              trials_per_task: int | None = None) -> tuple[tuple[str, int], ...]:
    """Expand the evaluation split into per-trial (task, seed) entries.

    `TRIALS_PER_TASK` trials per task, each a fresh draw. Scenes are NOT pinned: composite
    classes declare EXCLUDE_LAYOUTS / EXCLUDE_STYLES, so a pinned (layout, style) pair can
    be illegal for a task and only RoboCasa's sampler knows which. The seed makes the draw
    reproducible.

    Seeds are keyed on the trial's position in the WHOLE plan, so narrowing either the
    task list or the trial count yields a prefix of the full plan rather than renumbering
    what is left -- a reduced-scale run grades the same episodes as the full one.
    """
    if not tasks:
        raise ValueError("evaluation split must contain at least one task")
    per_task = TRIALS_PER_TASK if trials_per_task is None else int(trials_per_task)
    if per_task < 1:
        raise ValueError(f"trials_per_task must be at least 1, got {per_task}")
    entries = []
    for position, task in enumerate(tasks):
        for k in range(per_task):
            i = position * per_task + k
            h = hashlib.sha256(
                f"{_EVAL_DOMAIN}|{salt or ''}|{task}|trial{i}".encode()
            ).digest()[:8]
            entries.append((task, int.from_bytes(h, "big") % _SEED_MODULUS))
    return tuple(entries)


# -- what the agent is allowed to observe -------------------------------------
#
# IMAGES AND PROPRIOCEPTION ONLY. RoboCasa's observation dict also carries ground-truth
# poses for every object and fixture -- fruit_pos, obj_to_robot0_eef_pos, ... -- which a
# real robot could only get through perception. Handing them over turns a perception
# problem into arithmetic, and a composite success predicate is a conjunction over named
# objects, so an unfiltered observation would hand over exactly what the stage function
# scores.
#
# The rule is "keep robot<N>_*", exact rather than approximate, verified against a live
# env: `object-state` concatenates precisely the object keys and must go too, while
# `robot0_proprio-state` carries none and stays.
#
# World-frame ego pose (robot0_base_pos/quat, robot0_eef_pos/quat) is kept: the robot's
# own pose is odometry, and useless for finding an object whose position is not given.
_AGENT_OBS_RE = re.compile(r"^robot\d+_")


def agent_visible_obs(obs):
    """Filter an observation down to what the agent may see.

    Applied at the WIRE, in the daemon: the harness keeps the full observation for success
    checking, the stage function and the transcript, and only the agent's view is
    narrowed. Non-dict input passes through untouched.
    """
    if not isinstance(obs, dict):
        return obs
    return {k: v for k, v in obs.items() if _AGENT_OBS_RE.match(k)}
