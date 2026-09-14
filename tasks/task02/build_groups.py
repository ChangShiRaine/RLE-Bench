"""Emit one Harbor task per RoboCasa activity group.

    python tasks/task02/build_groups.py --emit-all          # a task dir per active group
    python tasks/task02/build_groups.py --emit "Baking"     # one, by name or by slug
    python tasks/task02/build_groups.py --trials 5          # graded trials per eval task
    python tasks/task02/build_groups.py --check             # dirs, splits and images agree
    python tasks/task02/build_groups.py --list              # the table, with coverage notes
    python tasks/task02/build_groups.py --regenerate-groups # re-derive GROUPS from the docs
    python tasks/task02/build_groups.py --recompute-split   # re-choose every `train` set

`tasks/task02/` is a Harbor DATASET directory: `NN-slug/` per group, each a task generated
from `_template/`. All of them run ONE image, built from `image/`: the root-only config
carries every group's split as a table, and each task.toml selects its entry with
RLEBENCH_GROUP. The canonical tasks/task02/harness/config.py holds the same table so the
dev suite has a split to import; `--emit` rewrites both.

THE SPLIT: hold out the member named in `held_out` as the sole evaluation task, and train
on whatever `train` names. Every
primitive it requires must have a provider in `train` -- checked here against
tasks/task02/dev/data/task02_primitive_audit.json, and fatal.

A `train` entry may be a member of the group (a COMPOSITE, a long procedure to factor
controllers out of) or an ATOMIC task drilling one interaction per episode. Every shipping
entry is composites only; the atomic path stays supported because a group whose `need` set
no member covers can still be filled that way.

`train` IS TAKEN AS WRITTEN -- any tasks, any number, either kind. `--recompute-split`
will fill it for a group that has no opinion, but the entry always wins, and each emitted
entry is checked against config.TRAIN_SET_SIZE.

`active` is written on EVERY entry, so switching a group off is a one-line edit at the
group rather than the absence of a line. `active=False` keeps it in the table and never
emits it: `--emit-all` skips it with a reason, `--emit` refuses it, and
`--check` does not expect its directory. Never delete an entry -- a group's position IS
its slug number, so removing one renames every directory after it.

THE TRIAL COUNT is the other thing written here. It lives in `config.TRIALS_PER_TASK` and
the `[[steps]]` list in `task.toml`; `--trials` writes both together and `--check` asserts
they agree, because a mismatch is silent -- it runs `next-trial` past the end of the plan,
or leaves trials unreached and scored 0.

`_template/steps/` carries ONE trial directory, `eval_01`, and `--emit` copies it out to
the count the plan names. So the graded agents differ only in which trial the daemon has
open, and that holds by construction rather than by review: checked-in copies could drift
from each other, generated ones cannot.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "_template"
IMAGE_DIR = HERE / "image"
IMAGE = "rlebench-task02-agent:dev"
TASK_TOML = TEMPLATE / "task.toml.in"
SLUG = "@@SLUG@@"
CONFIG = HERE / "harness" / "config.py"
REPO = HERE.parents[1]


def _docs_dir() -> Path:
    """The vendored RoboCasa docs, derived from the installed package so the table is
    regenerated from the same tree the registry comes from. ROBOCASA_DOCS overrides."""
    import os

    override = os.environ.get("ROBOCASA_DOCS", "").strip()
    if override:
        return Path(override)
    import robocasa

    return Path(robocasa.__file__).resolve().parents[1] / "docs"


BEGIN = "# --- BEGIN GENERATED SPLIT (tasks/task02/build_groups.py) ---"
END = "# --- END GENERATED SPLIT ---"

STEPS_BEGIN = "# --- BEGIN GENERATED EVAL STEPS (tasks/task02/build_groups.py) ---"
STEPS_END = "# --- END GENERATED EVAL STEPS ---"

# The eval step directory every other one is copied from at emit time, and the only one in
# the repo -- a real step rather than a separate template tree, because a template nobody
# runs is a template that rots.
STEP_TEMPLATE = "eval_01"


def eval_step_names(trials: int) -> list[str]:
    return [f"eval_{i:02d}" for i in range(1, trials + 1)]


# -- the primitive audit -------------------------------------------------------
# Checked in rather than derived live, because `--check` and `--emit` run from the
# Makefile's python3, which has no RoboCasa. Regenerate after a pin bump with
#     .venv-robocasa/bin/python tasks/task02/dev/decompose.py --audit --out <AUDIT>
# using --out, not a shell redirect: importing robosuite prints a banner to stdout.
AUDIT = REPO / "tasks" / "task02" / "dev" / "data" / "task02_primitive_audit.json"


def _train_set_size() -> int:
    """config.TRAIN_SET_SIZE, read late so importing this module does not depend on the
    repo being importable."""
    sys.path.insert(0, str(REPO))
    from harness import config as C

    return C.TRAIN_SET_SIZE


def _trials_per_task() -> int:
    """config.TRIALS_PER_TASK, read late for the same reason as `_train_set_size`."""
    sys.path.insert(0, str(REPO))
    from harness import config as C

    return C.TRIALS_PER_TASK


def load_audit() -> dict[str, dict]:
    import json

    if not AUDIT.is_file():
        raise SystemExit(
            f"missing {AUDIT.relative_to(REPO)}. Regenerate it with\n"
            "    .venv-robocasa/bin/python tasks/task02/dev/decompose.py --audit --out "
            f"{AUDIT.relative_to(REPO)}")
    return json.loads(AUDIT.read_text())


_COMPOSITES: frozenset[str] | None = None


def _composite_names() -> frozenset[str]:
    """Every RoboCasa COMPOSITE task, as the checked-in audit sees them. Cached: the
    composite/atomic split is asked once per training task per render."""
    global _COMPOSITES
    if _COMPOSITES is None:
        _COMPOSITES = frozenset(load_audit())
    return _COMPOSITES


def coverage(group: Group, audit: dict[str, dict] | None = None) -> dict[str, list[str]]:
    """Each primitive the held-out task requires -> the `train` entries providing it. An
    atomic provides itself and nothing else, which is the point of putting one in."""
    audit = load_audit() if audit is None else audit
    need = audit.get(group.eval_task, {}).get("need", [])
    return {p: [t for t in group.train
                if p == t or p in audit.get(t, {}).get("need", [])]
            for p in need}


def gaps(group: Group, audit: dict[str, dict] | None = None) -> tuple[str, ...]:
    """Primitives the held-out task needs that NO training task provides. Always fatal:
    the evaluation clause would be unsolvable by construction, and its zero would be a
    statement about the split rather than about the agent."""
    return tuple(p for p, prov in coverage(group, audit).items() if not prov)


def thin(group: Group, audit: dict[str, dict] | None = None) -> tuple[str, ...]:
    """Primitives reachable through exactly one training task. Advisory, not fatal.

    The count proxies how often practice REACHES the primitive, which only makes sense
    for composites, where it sits some way into a procedure that may fail first. A
    primitive drilled by its own atomic is never thin.
    """
    return tuple(p for p, prov in coverage(group, audit).items()
                 if len(prov) == 1 and p not in group.atomics)


def undecomposable(group: Group, audit: dict[str, dict] | None = None) -> str | None:
    """Why the held-out task's coverage claim cannot be trusted, or None. Neither case is
    a gap: a task `decompose.py` REFUSED may need primitives the table knows nothing
    about, and one that decomposes to NOTHING is covered only vacuously."""
    audit = load_audit() if audit is None else audit
    row = audit.get(group.eval_task)
    if row is None:
        return f"{group.eval_task} is not in the primitive audit"
    if row["unknown"]:
        return (f"{group.eval_task} did not decompose: {', '.join(row['unknown'])}. Its "
                "coverage cannot be checked, so it must not be graded")
    if not row["need"]:
        return (f"{group.eval_task} decomposes to no primitives at all, so 'every "
                "primitive is covered' is vacuous rather than clean")
    return None


@dataclass(frozen=True)
class Group:
    """One RoboCasa activity, its members with their published subtask counts, and the
    split chosen over them."""

    name: str
    tasks: tuple[tuple[str, int], ...]
    # The held-out member.
    held_out: str
    # What the agent practises on, TAKEN AS WRITTEN -- any tasks, any number, either
    # kind. A member of the group is a composite; anything else is an atomic task.
    # `--recompute-split` fills this for a group that has no opinion of its own.
    train: tuple[str, ...] = ()
    # THE BUILD SWITCH, written explicitly on every entry. False = kept for the record
    # and never emitted, whether because the coverage audit disqualifies it or because it
    # was switched off by hand. `--recompute-split` preserves a hand-set False; the
    # audit's own verdict it re-derives.
    active: bool = True
    # FREE-TEXT NOTES about this group, carried by ACTIVE and INACTIVE entries alike and
    # printed by `--list`. Whatever a reader of the table needs and cannot recompute: the
    # measured difficulty of the held-out task, why a member was passed over, what a graded
    # sweep showed. Nothing parses it.
    #
    # For an `active=False` entry it doubles as the reason, and `why_inactive` falls back to
    # it -- but only when the audit has NO complaint of its own, since an audit verdict is
    # re-derived and always wins. So this can never contradict the audit; it covers the
    # gates the audit cannot compute, difficulty being the one that matters.
    info: str = ""

    def __post_init__(self):
        # Only what would BREAK a build, never a preference about the split. `held_out`
        # names the graded task and indexes the subtask counts the writer emits.
        if self.held_out not in self.names:
            raise ValueError(f"{self.name}: held_out={self.held_out!r} is not a member")
        if self.held_out in self.train:
            raise ValueError(
                f"{self.name}: train contains the held-out task {self.held_out!r}; a "
                "graded task that was practised measures memorisation")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t for t, _ in self.tasks)

    @property
    def eval_task(self) -> str:
        return self.held_out

    # composites/atomics are DERIVED, not a second field: two fields could disagree.
    #
    # The question is "is it a COMPOSITE TASK", not "is it in this group". Those coincide
    # only while every training task is drawn from the group's own pool, and they stop
    # coinciding the moment a group has to borrow one -- Reheating Food holds out
    # SimmeringSauce, whose `PickPlaceCounterToStove` no member provides, so it trains on
    # KettleBoiling and PanTransfer from other activities. Read as membership, those two
    # come out "atomic", and `gradeable` then rejects the split for naming atomics that are
    # not in ATOMIC_TASK_DATASETS. The audit's keys ARE the registered composites, and it is
    # checked in, so this stays answerable without RoboCasa.
    @property
    def composites(self) -> tuple[str, ...]:
        names = _composite_names()
        return tuple(t for t in self.train if t in names)

    @property
    def atomics(self) -> tuple[str, ...]:
        names = _composite_names()
        return tuple(t for t in self.train if t not in names)

    @property
    def pool(self) -> tuple[str, ...]:
        """Every member `train` could have been chosen from."""
        return tuple(t for t in self.names if t != self.eval_task)


# Every activity group with >= 5 members THIS ROBOCASA PIN REGISTERS, so a training set can
# be chosen from the pool rather than taken whole. A cache with provenance, not a magic
# literal: `--regenerate-groups` rebuilds the pools from the vendored docs intersected with
# COMPOSITE_TASK_DATASETS, and `--recompute-split` re-chooses every `train`. The
# intersection is not cosmetic -- the docs describe 300 composite tasks, the pin registers
# 252, which is the difference between 30 groups and 23.
#
# ORDER IS LOAD-BEARING THREE TIMES OVER. The default split is GROUPS[0]; a group's
# position is its emitted slug number; and the sequence itself now records the measured
# difficulty band -- EASY, then MEDIUM, then HARD, then everything switched off, five in
# each active band. The banners below mark the boundaries.
#
# So a group that changes band has to MOVE, and moving it renumbers every slug after it:
# `--check` will report the emitted directories as mismatched until `--emit-all` runs and
# the stale directories are deleted. Job directories under jobs/ keep the numbers they were
# run with, so a slug in a run's path is only meaningful against the table of that day.
# Never delete an entry -- mark it `active=False` and move it to the inactive block.
#
# COVERAGE IS A GATE. Every active group's held-out task decomposes cleanly and has a
# provider in `train` for each primitive it needs; `--check` and `--emit` both
# refuse otherwise. Inactive entries carry the reason in `info`.
#
# DIFFICULTY IS A SECOND GATE, and it is the one `active` carries by hand -- nothing here
# computes it. A held-out task ships only if it clears all three:
#
#   1. it is a member of its own group (enforced in __post_init__);
#   2. it reflects what the group is named for, so the graded task is representative of
#      the activity rather than the one easy chore that happens to sit in the pool;
#   3. every primitive it needs has a provider in `train` (this is the coverage gate).
#
# WHAT DECIDES THE BAND is the primitive vocabulary. The OPEN-SURFACE SET -- CheesyBread,
# NavigateKitchen, PickPlaceCounterToStove, PickPlaceCounterToSink -- is what development
# sweeps show an agent can actually land; hinged doors and enclosed cavities sit an order
# of magnitude below it, near enough to zero that a clause needing one is a clause the
# budget cannot buy. Fixture CONTROLS go with the doors: a faucet lever or a knob means
# finding the control, working out which way it turns and leaving it in a commanded state,
# none of which is the pick-place-and-carry competence being measured.
#
# So EASY and MEDIUM are open-surface work and HARD is deliberately the door and cavity
# groups. HARD is not a mistake to be fixed -- a benchmark wants a floor nothing clears
# yet -- but nothing in it is expected to score, and a group is only worth a HARD slot if
# its held-out task is representative of the activity rather than merely difficult.
#
# A FOURTH GATE IS IMPLICIT: the held-out task needs a stage function in
# tasks/task02/harness/stages.py, and only 47 tasks have one. That is what disqualifies most
# of the inactive entries below -- not that no easier member exists, but that the easier
# member cannot be graded. Writing its stage function is how those groups come back.
GROUPS: tuple[Group, ...] = (
    # ---------------- EASY: solved in every graded trial ----------------
    Group(
        # DIFFICULTY: held out on DumpLeftovers rather than the audit's SortingCleanup,
        # which needs PickPlaceCounterToCabinet and CloseCabinet. Everything here is the
        # counter and the open sink basin.
        name="Washing Dishes",
        held_out="DumpLeftovers",
        train=("PlaceOnDishRack", "SortingCleanup", "StackBowlsInSink"),
        active=True,
        info=("EASY, measured - the counter and the open sink basin, nothing hinged. "
              "Three conjuncts, resolution 2."),
        tasks=(
            ("ClearSink", 4),
            ("CollectWashingSupplies", 2),
            ("DivideBasins", 2),
            ("DryDishes", 2),
            ("DryDrinkware", 2),
            ("DumpLeftovers", 3),
            ("PlaceDishesBySink", 7),
            ("PlaceOnDishRack", 2),
            ("PreRinseStation", 2),
            ("PreSoakPan", 3),
            ("ReturnWashingSupplies", 8),
            ("RinseBowls", 3),
            ("ScrubBowl", 2),
            ("SoakSponge", 2),
            ("SortingCleanup", 5),
            ("StackBowlsInSink", 2),
            ("TransportCookware", 7),
        ),
    ),
    Group(
        name="Sauteing Vegetables",
        held_out="PlaceVegetablesEvenly",
        train=("AdjustHeat", "StirVegetables", "TiltPan"),
        active=True,
        info=("EASY, measured - two vegetables into a pan ALREADY on the stove; the "
              "not-stacked and spread-apart constraints did not bite."),
        tasks=(
            ("AdjustHeat", 3),
            ("ButterOnPan", 2),
            ("PlaceVegetablesEvenly", 2),
            ("PreheatPot", 2),
            ("ShakePan", 2),
            ("StirVegetables", 4),
            ("TiltPan", 2),
        ),
    ),
    Group(
        # DIFFICULTY: held out on PastryDisplay. The audit's OrganizeBakingIngredients
        # decomposes to no primitives at all, so its coverage claim was vacuous; this one
        # is two pastries onto two plates on one counter.
        name="Baking",
        held_out="PastryDisplay",
        train=("CookieDoughPrep", "CoolBakedCake", "CupcakeCleanup"),
        active=True,
        info=("EASY label, measured. CHECK: the rate is short of 1, so the definition "
              "says medium, and PortionHotDogs sits at the same rate under a medium label "
              "- the two labels disagree with each other. One sweep is too few to separate "
              "them; a second would settle it, as it did for PortionHotDogs."),
        tasks=(
            ("CookieDoughPrep", 7),
            ("CoolBakedCake", 8),
            ("CupcakeCleanup", 2),
            ("MixCakeFrosting", 5),
            ("OrganizeBakingIngredients", 3),
            ("PastryDisplay", 2),
        ),
    ),
    Group(
        # DIFFICULTY: held out on SimmeringSauce rather than WarmCroissant, which is a
        # croissant into a pan ALREADY on the stove plus a knob -- two conjuncts, both easy.
        # The other three members are all microwave-door work.
        # TRAIN BORROWS, and has to: no member of this group moves anything onto a burner,
        # so SimmeringSauce's PickPlaceCounterToStove has no provider here. KettleBoiling
        # (Brewing) puts a kettle from the counter onto a burner and lights it; PanTransfer
        # (Serving Food) returns a pan to the stove. All three are composites, none is held
        # out anywhere else, all are open-surface, and every primitive gets two providers.
        name="Reheating Food",
        held_out="SimmeringSauce",
        train=("KettleBoiling", "PanTransfer", "WarmCroissant"),
        active=True,
        info=("EASY, predicted - the pan goes on a RANDOMLY NAMED burner given in the "
              "instruction, which is MealPrepStaging's measured difficulty plus language "
              "grounding; then two foods into that pan and that same burner lit. Four "
              "conjuncts, NONE free at reset, so full resolution. No traverse - "
              "init_robot_base_ref is the stove - and its only fixture step is a knob, the "
              "one fixture interaction that measures workable."),
        tasks=(
            ("HeatMug", 4),
            ("MakeLoadedPotato", 6),
            ("SimmeringSauce", 4),
            ("WaffleReheat", 4),
            ("WarmCroissant", 2),
        ),
    ),
    Group(
        name="Chopping Food",
        held_out="ClearCuttingBoard",
        train=("BreadSetupSlicing", "MeatTransfer", "OrganizeVegetables"),
        active=True,
        info=("EASY, measured - two vegetables onto a board and a non-vegetable cleared "
              "off it, one counter, CheesyBread the only primitive."),
        tasks=(
            ("ArrangeVegetables", 2),
            ("BreadSetupSlicing", 2),
            ("ClearCuttingBoard", 3),
            ("MeatTransfer", 3),
            ("OrganizeVegetables", 2),
        ),
    ),
    # ---------------- MEDIUM: 0 < rate < 1, the band that ranks ----------------
    Group(
        # DIFFICULTY: held out on AlignSilverware rather than the audit's SetupButterPlate,
        # which needs OpenFridge and never once solved. Fork and spoon start on
        # the counter, the plate is already on the dining counter, and nothing opens.
        # TRAIN is the whole door-free pool; the other nine members spawn in a cabinet, a
        # fridge or a toaster oven. SetupWineGlasses is the closest rehearsal -- same
        # predicate shape, same 0.25/0.10 thresholds, same stool frame.
        name="Setting the Table",
        held_out="AlignSilverware",
        train=("ArrangeDrinkware", "BeverageOrganization", "SetupWineGlasses"),
        active=True,
        info=("MEDIUM, predicted - precise left/right placement against a reference object "
              "in the stool's frame plus a counter-to-dining-counter carry, four of five "
              "conjuncts free. CHECK: unmeasured, and the traverse never once succeeded in "
              "development, for this task or for the structurally identical SetupWineGlasses "
              "or the NavigateKitchen atomic, so this may be hard. Note the member it "
              "replaced, SetupButterPlate, is the suite's best-evidenced hard task, so "
              "reverting is the cheap way to another HARD."),
        tasks=(
            ("AlignSilverware", 7),
            ("ArrangeBreadBasket", 5),
            ("ArrangeBreadBowl", 5),
            ("ArrangeDrinkware", 7),
            ("BeverageOrganization", 7),
            ("DateNight", 8),
            ("SeasoningSpiceSetup", 8),
            ("SetBowlsForSoup", 8),
            ("SetupBowls", 7),
            ("SetupButterPlate", 8),
            ("SetupFruitBowl", 8),
            ("SetupWineGlasses", 7),
            ("SizeSorting", 2),
        ),
    ),
    Group(
        name="Portioning Meals",
        held_out="PortionHotDogs",
        train=("DistributeChicken", "PortionOnSize", "ScalePortioning"),
        active=True,
        info=("MEDIUM, measured over two sweeps of the same task and train set. A bun and "
              "a sausage onto each of two plates on open counters. The first sweep alone "
              "would read as easy; pooling the two is what puts it in the band."),
        tasks=(
            ("DistributeChicken", 7),
            ("PortionFruitBowl", 4),
            ("PortionHotDogs", 4),
            ("PortionInTupperware", 4),
            ("PortionOnSize", 7),
            ("PortionYogurt", 4),
            ("ScalePortioning", 6),
        ),
    ),
    Group(
        name="Defrosting Food",
        held_out="DefrostByCategory",
        train=("MoveToCounter", "QuickThaw", "ThawInSink"),
        active=True,
        info=("MEDIUM, measured - four items sorted between the sink and a bowl, "
              "resolution 4 of 5, built from the two open-surface primitives that land most "
              "often (CheesyBread and PickPlaceCounterToSink)."),
        tasks=(
            ("DefrostByCategory", 4),
            ("MicrowaveThawing", 4),
            ("MicrowaveThawingFridge", 5),
            ("MoveToCounter", 3),
            ("QuickThaw", 2),
            ("ThawInSink", 2),
        ),
    ),
    Group(
        # DIFFICULTY: held out on PlaceBeveragesTogether rather than
        # ArrangeBuffetDessert, which starts both sweets inside a fridge nothing
        # opens. Three drinks from the counter to the dining counter, clustered.
        name="Arranging Buffet",
        held_out="PlaceBeveragesTogether",
        train=("ArrangeBuffetDessert", "DivideBuffetTrays", "TongBuffetSetup"),
        active=True,
        info=("MEDIUM, predicted - three drinks carried to the dining counter and clustered "
              "there, resolution 4 of 5. CHECK: unmeasured, and it needs NavigateKitchen, "
              "the traverse that never landed anywhere in the dining-counter family, so this "
              "may be hard."),
        tasks=(
            ("ArrangeBuffetDessert", 8),
            ("CutBuffetPizza", 2),
            ("DivideBuffetTrays", 16),
            ("PlaceBeveragesTogether", 11),
            ("TongBuffetSetup", 4),
        ),
    ),
    Group(
        name="Serving Beverages",
        held_out="MatchCupAndDrink",
        train=("AlcoholServingPrep", "PrepareCocktailStation", "PrepareDrinkStation"),
        active=True,
        info=("MEDIUM, predicted - MatchCupAndDrink needs NavigateKitchen, the "
              "traverse that never once succeeded in development, and none of the "
              "three train tasks landed either; only a sweep will settle it."),
        tasks=(
            ("AlcoholServingPrep", 8),
            ("DeliverStraw", 4),
            ("MatchCupAndDrink", 7),
            ("PrepareCocktailStation", 12),
            ("PrepareDrinkStation", 11),
            ("SetupSodaBowl", 8),
        ),
    ),
    # ---------------- HARD: no trial solved ----------------
    Group(
        name="Loading Fridge",
        held_out="MoveFreezerToFridge",
        train=("LoadCondimentsInFridge", "LoadPreparedFood", "PlaceVeggiesInDrawer"),
        active=True,
        info=("HARD, measured across three sweeps with no trial solved - MoveFreezerToFridge "
              "needs OpenFridge AND CloseFridge, and every member is fridge-door work. "
              "OpenFridge never landed in development either."),
        tasks=(
            ("CreateChildFriendlyFridge", 11),
            ("LoadCondimentsInFridge", 9),
            ("LoadFridgeByType", 7),
            ("LoadFridgeFifo", 4),
            ("LoadPreparedFood", 4),
            ("MoveFreezerToFridge", 2),
            ("PlaceVeggiesInDrawer", 8),
            ("RearrangeFridgeItems", 3),
        ),
    ),
    Group(
        name="Managing Freezer Space",
        held_out="SeparateFreezerRack",
        train=("ClearFreezer", "MoveFridgeToFreezer", "ReorganizeFrozenVegetables"),
        active=True,
        info=("HARD, measured across three sweeps with no trial solved - _setup_scene opens "
              "the freezer, but two containers must then land on NAMED racks deep in the "
              "cavity, the bowl_in_cabinet failure mode. Resolution is only 2 of 7: five "
              "conjuncts are true at reset, so the reward is nearly binary either way."),
        tasks=(
            ("ClearFreezer", 11),
            ("FreezeBottledWaters", 8),
            ("FreezeIceTray", 4),
            ("MaximizeFreezerSpace", 3),
            ("MoveFridgeToFreezer", 2),
            ("MoveToFreezerDrawer", 2),
            ("ReorganizeFrozenVegetables", 3),
            ("SeparateFreezerRack", 7),
        ),
    ),
    Group(
        name="Clearing Table",
        held_out="CandleCleanup",
        train=("ClearReceptaclesForCleaning", "CondimentCollection", "FoodCleanup"),
        active=True,
        info=("HARD label, measured over four sweeps with a single trial solved. CHECK: "
              "one solved trial is strictly a non-zero rate, so the definition says medium, "
              "though it is the lowest in the suite and the HARD entries around it sit at a "
              "clean zero. CandleCleanup needs CloseCabinet AND PickPlaceCounterToCabinet, "
              "both of which barely landed in development, and every one of its four "
              "primitives has exactly ONE provider."),
        tasks=(
            ("BowlAndCup", 4),
            ("CandleCleanup", 8),
            ("ClearReceptaclesForCleaning", 8),
            ("CondimentCollection", 2),
            ("DessertAssembly", 2),
            ("DrinkwareConsolidation", 3),
            ("FoodCleanup", 2),
        ),
    ),
    Group(
        name="Microwaving Food",
        held_out="PlaceMicrowaveSafeItem",
        train=("FilterMicrowavableItem", "ReheatMeal", "ReturnHeatedFood"),
        active=True,
        info=("HARD, measured with no trial solved - every member is a microwave-door task, "
              "and PlaceMicrowaveSafeItem never once closed the door or started the "
              "microwave. PickPlaceCounterToMicrowave barely landed in development."),
        tasks=(
            ("FilterMicrowavableItem", 6),
            ("MicrowaveCorrectMeal", 5),
            ("MicrowaveDefrostMeat", 5),
            ("PlaceMicrowaveSafeItem", 3),
            ("ReheatMeal", 5),
            ("ReturnHeatedFood", 4),
        ),
    ),
    Group(
        name="Storing Leftovers",
        held_out="StoreLeftoversInBowl",
        train=("FreezeCookedFood", "PrepareStoringLeftovers", "StoreDumplings"),
        active=True,
        info=("HARD label, but a trial WAS solved in the one sweep that ran it. CHECK: a "
              "non-zero rate means the stated definition says MEDIUM, not hard. It does need "
              "OpenFridge, which never landed in development, none of the three train tasks "
              "landed either, and coverage is as thin as it can be - StoreDumplings is the "
              "sole provider of BOTH primitives."),
        tasks=(
            ("FreezeCookedFood", 4),
            ("PrepareStoringLeftovers", 7),
            ("StoreDumplings", 11),
            ("StoreLeftoversByType", 7),
            ("StoreLeftoversInBowl", 5),
        ),
    ),
    # ---------------- NOT ACTIVATED ----------------
    Group(
        # DIFFICULTY: the whole group is the sink, and washing means the faucet.
        name="Washing Fruits and Vegetables",
        held_out="AfterwashSorting",
        train=("ClearClutter", "DrainVeggies", "PrewashFoodAssembly"),
        active=False,
        info=("every member needs the sink faucet or a cabinet: AfterwashSorting, "
                "AirDryFruit, ClearClutter, DrainVeggies, PrewashFoodAssembly and "
                "WashFruitColander all turn the water on or off, and PrewashFoodSorting -- "
                "the one that does not -- starts two of its three foods inside the cabinet"),
        tasks=(
            ("AfterwashSorting", 4),
            ("AirDryFruit", 3),
            ("ClearClutter", 6),
            ("ClearSinkSpace", 2),
            ("DrainVeggies", 4),
            ("PrewashFoodAssembly", 5),
            ("PrewashFoodSorting", 8),
            ("WashFruitColander", 4),
        ),
    ),
    Group(
        name="Arranging Cabinets",
        held_out="RestockPantry",
        train=("BeverageSorting", "ResetCabinetDoors", "RestockBowls"),
        active=False,
        info=("RestockPantry needs PickPlaceCounterToCabinet, and the group's only other "
                "gradeable member, GatherTableware, decomposes to no primitives at all"),
        tasks=(
            ("BeverageSorting", 9),
            ("GatherTableware", 4),
            ("ResetCabinetDoors", 5),
            ("RestockBowls", 4),
            ("RestockPantry", 2),
            ("StackCans", 2),
            ("StockingBreakfastFoods", 3),
        ),
    ),
    Group(
        name="Frying",
        held_out="MealPrepStaging",
        train=("AssembleCookingArray", "SearingMeat", "SetupFrying"),
        active=False,
        info=("HELD BACK by hand - the stage function's judgement is not trusted for this "
              "task. For the record, MealPrepStaging is the best-evidenced MEDIUM in the "
              "suite, consistent across five sweeps, so if a medium slot is ever short this "
              "is the group with the evidence already in hand."),
        tasks=(
            ("AssembleCookingArray", 4),
            ("FryingPanAdjustment", 2),
            ("MealPrepStaging", 4),
            ("PressChicken", 2),
            ("RotatePan", 2),
            ("SearingMeat", 3),
            ("SetupFrying", 2),
        ),
    ),
    Group(
        # INACTIVE: SweetenCoffee needs OpenFridge, which no member of the group
        # provides.
        name="Brewing",
        held_out="SweetenCoffee",
        train=(),
        active=False,
        tasks=(
            ("ArrangeTea", 3),
            ("DeliverBrewedCoffee", 3),
            ("KettleBoiling", 2),
            ("OrganizeCoffeeCondiments", 2),
            ("PrepareCoffee", 2),
            ("SweetenCoffee", 5),
        ),
    ),
    Group(
        # INACTIVE: PrepareSmoothie needs OpenFridge, which no member of the group
        # provides.
        name="Making Smoothies",
        held_out="PrepareSmoothie",
        train=(),
        active=False,
        tasks=(
            ("AddIceCubes", 2),
            ("AddSweetener", 2),
            ("BlendIngredients", 4),
            ("PlaceStraw", 2),
            ("PrepareSmoothie", 7),
        ),
    ),
    Group(
        name="Serving Food",
        held_out="DessertUpgrade",
        train=("PlaceFoodInBowls", "PrepareSoupServing", "ServeSteak"),
        active=False,
        info=("HELD BACK - no member lands in the medium band. DessertUpgrade measures "
              "easy: two desserts onto one tray on one counter. ServeSteak needs "
              "NavigateKitchen, which no member provides and which never landed anywhere, so "
              "it is likely hard. PanTransfer is genuine tool use but sits in "
              "stages.LOW_RESOLUTION at 1 free conjunct, so its reward is binary. The other "
              "two are cabinet work."),
        tasks=(
            ("DessertUpgrade", 2),
            ("PanTransfer", 3),
            ("PlaceFoodInBowls", 4),
            ("PrepareSoupServing", 3),
            ("ServeSteak", 4),
        ),
    ),
    Group(
        name="Snack Preparation",
        held_out="VeggieDipPrep",
        train=("BreadAndCheese", "CerealAndBowl", "MakeFruitBowl"),
        active=False,
        info=("HELD BACK - no member lands in the medium band. VeggieDipPrep is three "
              "objects onto a tray on one counter, the same shape as PortionHotDogs but "
              "easier; BreadAndCheese is simpler still. CerealAndBowl, MakeFruitBowl and "
              "YogurtDelightPrep are all cabinet work, which barely lands in development, "
              "and YogurtDelightPrep is refused by the audit on custom-state:num_fruit."),
        tasks=(
            ("BreadAndCheese", 2),
            ("CerealAndBowl", 4),
            ("MakeFruitBowl", 3),
            ("VeggieDipPrep", 3),
            ("YogurtDelightPrep", 2),
        ),
    ),
    Group(
        # INACTIVE: ToastOnCorrectRack did not decompose: custom-state:bread_rack_level,
        # custom-state:meat_rack_level. Its coverage cannot be checked, so it must not
        # be graded.
        name="Toasting Bread",
        held_out="ToastOnCorrectRack",
        train=("GetToastedBread", "ServeWarmCroissant", "ToastBagel"),
        active=False,
        tasks=(
            ("GetToastedBread", 4),
            ("ServeWarmCroissant", 3),
            ("ToastBagel", 5),
            ("ToastBaguette", 5),
            ("ToastOnCorrectRack", 7),
        ),
    ),

)

# The first group that SHIPS, not GROUPS[0]: the entry config selects when
# RLEBENCH_GROUP is unset (the dev suite).
DEFAULT_GROUP = next(g for g in GROUPS if g.active)


def active_groups() -> tuple[tuple[int, Group], ...]:
    """(slug index, group) for the groups that ship. The index is the position in the FULL
    table, so deactivating a group never renumbers the others."""
    return tuple((i, g) for i, g in enumerate(GROUPS, 1) if g.active)


# -- writing ------------------------------------------------------------------

def render_entry(index: int, group: Group) -> str:
    counts = dict(group.tasks)

    def tup(items):
        return "(" + "".join(f'"{i}", ' for i in items).rstrip(", ") + ("," if len(items) == 1 else "") + ")"

    train = ", ".join(f"{t} {counts[t]}" if t in counts else f"{t} (from another activity)"
                      for t in group.train)
    return "\n".join([
        f"    # {group.name}: {len(group.tasks)} members, {len(group.train)} "
        + ("atomics drilled." if group.atomics else "trained on."),
        f"    # Train: {train}. Held out: {group.eval_task} {counts[group.eval_task]}.",
        f'    "{slug_for(index, group)}": (',
        f'        "{group.name}",',
        f"        {tup(group.train)},",
        f"        {tup((group.eval_task,))},",
        "    ),",
    ])


def render_split(groups: tuple[tuple[int, Group], ...]) -> str:
    """The SPLITS table: every shipped group, keyed by slug. The training set is TOLD TO
    THE AGENT via `sim.list_tasks()`; the held-out member is never revealed."""
    return "\n".join([
        BEGIN, "",
        "SPLITS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {",
        *(render_entry(i, g) for i, g in groups),
        "}", "",
        END,
    ])


def _drop_bytecode(path: Path) -> None:
    """Delete cached bytecode for a source file we just rewrote.

    CPython validates a .pyc by (mtime, size) ONLY. `--trials 2` and `--trials 5` differ
    by one digit, so a rewrite keeps the size and, run back to back, the mtime -- and the
    stale .pyc is accepted. Whatever writes config.py must call this, or `--check`
    verifies a config that is not on disk.
    """
    for pyc in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
        pyc.unlink()


def write_config(groups: tuple[tuple[int, Group], ...] | None = None) -> None:
    """Write the SPLITS table into the canonical tasks/task02/harness/config.py, and pin
    TRAIN_SET_SIZE to the one size every entry has (`check_split` refuses a daemon whose
    group disagrees with it)."""
    groups = active_groups() if groups is None else groups
    sizes = {len(g.train) for _, g in groups}
    if len(sizes) != 1:
        raise SystemExit(f"groups disagree on the training set size: {sorted(sizes)}")
    text = CONFIG.read_text()
    start, end = text.index(BEGIN), text.index(END) + len(END)
    text = text[:start] + render_split(groups) + text[end:]
    text, count = re.subn(r"^TRAIN_SET_SIZE = \d+$", f"TRAIN_SET_SIZE = {sizes.pop()}",
                          text, count=1, flags=re.M)
    if count != 1:
        raise SystemExit(f"could not find `TRAIN_SET_SIZE = <n>` in {CONFIG}")
    CONFIG.write_text(text)
    _drop_bytecode(CONFIG)


# -- the graded steps ---------------------------------------------------------
# One trial is one Harbor step is one fresh agent, so THREE things carry the trial count:
# config.TRIALS_PER_TASK, the [[steps]] list in task.toml, and the steps/ directories.
# Written together because a mismatch is silent.

def _trial_timeout() -> float:
    """The per-trial agent clock, read off `RLEBENCH_TRIAL_SECONDS` in task.toml.in.

    Derived rather than written twice: the step's `timeout_sec` MUST mirror the clock the
    daemon reports, or the agent is told it has time Harbor is about to take away. The
    variable sits outside the generated region, so reading it back here is safe.
    """
    m = re.search(r'RLEBENCH_TRIAL_SECONDS = "\$\{RLEBENCH_TRIAL_SECONDS:-(\d+)\}"',
                  TASK_TOML.read_text())
    if not m:
        raise SystemExit("could not read RLEBENCH_TRIAL_SECONDS from task.toml.in")
    return float(m.group(1))


def render_eval_steps(trials: int) -> str:
    """The [[steps]] block for the graded trials, seal hook included."""
    out = [STEPS_BEGIN, ""]
    trial_timeout = _trial_timeout()
    for name in eval_step_names(trials):
        out += [
            "[[steps]]",
            f'name = "{name}"',
            "[steps.agent]",
            f"timeout_sec = {trial_timeout}",
            'user = "agent"',
            "[steps.verifier]",
            "timeout_sec = 600.0",
            "[[steps.verifier.collect]]",
            'command = "PYTHONSAFEPATH=1 PYTHONPATH=/opt/private python -m '
            'harness.control next-trial"',
            'user = "root"',
            "timeout_sec = 180.0",
            "",
        ]
    out += [
        "# Seal AFTER the last trial is scored, and ONLY here. Sealing kills the daemon,",
        "# which serves every step, so on any earlier one it would strand the trials",
        "# behind it.",
        "#",
        "# NOTE that sealing is worth NO reward. It happens unconditionally, so it says",
        "# nothing about whether the agent did anything.",
        "[[steps.verifier.collect]]",
        'command = "/opt/seal_ledger.sh"',
        'user = "root"',
        "timeout_sec = 120.0",
        "",
        STEPS_END,
    ]
    return "\n".join(out)


def write_trials(trials: int) -> None:
    """Rewrite TRIALS_PER_TASK and the [[steps]] list together.

    The step DIRECTORIES are not touched: only eval_01 is checked in, and `emit` copies it
    out to the count the plan names.
    """
    text = CONFIG.read_text()
    new, count = re.subn(r"^TRIALS_PER_TASK = \d+$", f"TRIALS_PER_TASK = {trials}",
                         text, count=1, flags=re.M)
    if count != 1:
        raise SystemExit("could not find `TRIALS_PER_TASK = <n>` in config.py")
    CONFIG.write_text(new)
    _drop_bytecode(CONFIG)

    toml = TASK_TOML.read_text()
    start = toml.index(STEPS_BEGIN)
    end = toml.index(STEPS_END) + len(STEPS_END)
    TASK_TOML.write_text(toml[:start] + render_eval_steps(trials) + toml[end:])


def sync_step_dirs(steps: Path, trials: int) -> None:
    """Make `steps` hold exactly the eval directories the step list names.

    Every one past the first is a copy of eval_01, which is what makes the trials
    interchangeable: the agents differ only in which trial the daemon has open, never in
    what they were told. Only eval_01 is checked in, and this is the reason -- copies in
    the repo could drift from it, whereas copies made here cannot.
    """
    template = steps / STEP_TEMPLATE
    if not template.is_dir():
        raise SystemExit(f"missing step template {template}")

    wanted = eval_step_names(trials)
    for name in wanted[1:]:
        target = steps / name
        if target.is_dir():
            shutil.rmtree(target)
        shutil.copytree(template, target)
    for stale in sorted(steps.glob("eval_*")):
        if stale.name not in wanted:
            shutil.rmtree(stale)


# -- emitting one Harbor task per group ----------------------------------------
# Each group is a copy of `_template/`; the payloads are staged once, into image/.

def slug_for(index: int, group: Group) -> str:
    return f"{index:02d}-" + group.name.lower().replace(" ", "-")


def disqualified(group: Group, audit: dict[str, dict] | None = None) -> str | None:
    """Why this group's SPLIT is unsound, or None. Pure coverage -- see `gradeable` for
    the full emit test. An ACTIVE group reaching here with a reason is a bug in the table;
    it is re-derived anyway because GROUPS is hand-editable."""
    audit = load_audit() if audit is None else audit
    why = undecomposable(group, audit)
    if why:
        return f"{group.name}: {why}"
    missing = gaps(group, audit)
    if missing:
        return (f"{group.name}: {group.eval_task} needs {', '.join(missing)}, which no "
                f"training task provides. The evaluation clause would be unsolvable by "
                f"construction, so its zero would measure the split rather than the agent")
    return None


def why_inactive(group: Group, audit: dict[str, dict] | None = None) -> str:
    """The reason an `active=False` group cannot ship, RE-DERIVED rather than read back
    from the entry's `info`. If the two disagree, `info` is wrong."""
    audit = load_audit() if audit is None else audit
    why = undecomposable(group, audit)
    if why:
        return why
    # Search the WHOLE pool, not `train`: an inactive group may carry no training set at
    # all, and "every primitive is uncovered" would be a useless thing to report.
    _, why = choose_train(group, audit)
    return why or group.info or "marked inactive by hand; no reason recorded"


def gradeable(group: Group) -> str | None:
    """Why this group cannot be emitted, or None if it can."""
    if not group.active:
        return why_inactive(group)

    sys.path.insert(0, str(REPO))
    from harness.stages import has_stages

    if not has_stages(group.eval_task):
        return f"no stage function for {group.eval_task}"
    return disqualified(group)


def emitted_groups() -> list[tuple[str, Path]]:
    """(slug, path) for every group directory currently on disk, in numbered order."""
    return [(p.name, p) for p in sorted(HERE.glob("[0-9][0-9]-*")) if p.is_dir()]


def _staged_config() -> Path:
    """The config.py the image ships -- one definition, because `emit_image` writes it
    and `--check` reads it back, and a check pointed elsewhere checks nothing."""
    return IMAGE_DIR / "payload_private" / "harness" / "config.py"


def _staged_train_set_size() -> int | None:
    cfg = _staged_config()
    if not cfg.is_file():
        return None
    found = re.search(r"^TRAIN_SET_SIZE = (\d+)$", cfg.read_text(), re.M)
    return int(found.group(1)) if found else None


def _staged_split(slug: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(train, eval) the image ships for one slug. Both halves get checked: a wrong
    EVAL_TASKS grades a task nobody chose, a wrong TRAIN_TASKS hands the agent a set
    that may not cover it, and neither complains at run time."""
    cfg = _staged_config()
    if not cfg.is_file():
        return (), ()
    entry = re.search(rf'^    "{re.escape(slug)}": \((.*?)^    \),', cfg.read_text(),
                      re.S | re.M)
    if not entry:
        return (), ()
    lines = [l.strip() for l in entry.group(1).strip().splitlines()]
    train, evaluation = (tuple(re.findall(r'"(\w+)"', l)) for l in lines[1:3])
    return train, evaluation


def staged_eval_tasks(slug: str) -> tuple[str, ...]:
    return _staged_split(slug)[1]


def staged_train_tasks(slug: str) -> tuple[str, ...]:
    return _staged_split(slug)[0]


def emit_image() -> Path:
    """The docker build context every group runs in."""
    sys.path.insert(0, str(HERE))
    import build_assets

    write_config()
    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)
    shutil.copytree(TEMPLATE / "image", IMAGE_DIR)
    build_assets.stage(IMAGE_DIR / "payload_private", build_assets.DAEMON_MODULES)
    build_assets.stage_core(IMAGE_DIR / "payload_private")
    build_assets.stage(IMAGE_DIR / "payload_agent", build_assets.AGENT_MODULES)
    leaks = build_assets.check(IMAGE_DIR / "payload_agent", all_eval_tasks())
    if leaks:
        raise SystemExit(f"BOUNDARY VIOLATION in image/: {leaks}")
    return IMAGE_DIR


def all_eval_tasks() -> tuple[str, ...]:
    """Every held-out name the shared image carries; none may reach the agent's tree."""
    return tuple(g.eval_task for _, g in active_groups())


def emit(index: int, group: Group) -> Path:
    """Write tasks/task02/NN-slug/ for one group."""
    slug = slug_for(index, group)
    dest = HERE / slug
    if dest.exists():
        shutil.rmtree(dest)
    for sub in ("environment", "steps", "tests"):
        shutil.copytree(TEMPLATE / sub, dest / sub)
    # The template carries eval_01 only; the rest of the trials are made here.
    sync_step_dirs(dest / "steps", _trials_per_task())
    (dest / "task.toml").write_text(TASK_TOML.read_text().replace(SLUG, slug))
    return dest


# -- checking -----------------------------------------------------------------

def differences() -> list[str]:
    sys.path.insert(0, str(REPO))
    from harness import config as C

    problems = []
    if len(C.EVAL_TASKS) != 1:
        problems.append(f"config declares {len(C.EVAL_TASKS)} evaluation tasks; expected 1")
    if len(C.TRAIN_TASKS) != C.TRAIN_SET_SIZE:
        problems.append(f"config declares {len(C.TRAIN_TASKS)} training tasks; "
                        f"expected {C.TRAIN_SET_SIZE} (TRAIN_SET_SIZE)")

    # Coverage re-derived rather than trusted: GROUPS is hand-editable.
    audit = load_audit()
    for _, group in active_groups():
        why = disqualified(group, audit)
        if why:
            problems.append(f"COVERAGE: {why}")

    # The emitted directories against the table they came from: a folder whose staged
    # split has drifted ships a group nobody chose, silently.
    expected_groups = {slug_for(i, g): g for i, g in active_groups()
                       if gradeable(g) is None}
    present_groups = dict(emitted_groups())
    for slug in sorted(set(present_groups) - set(expected_groups)):
        problems.append(f"{slug} is emitted but is not a gradeable group")
    for slug in sorted(set(expected_groups) - set(present_groups)):
        problems.append(f"{slug} is gradeable but not emitted; run --emit-all")
    if not (IMAGE_DIR / "Dockerfile").is_file():
        problems.append("image/ has no build context; run --emit-all")
    for slug, group in sorted(expected_groups.items()):
        path = present_groups.get(slug)
        if path is None:
            continue
        if staged_eval_tasks(slug) != (group.eval_task,):
            problems.append(f"{slug} ships {staged_eval_tasks(slug)}; "
                            f"expected ('{group.eval_task}',)")
        if staged_train_tasks(slug) != group.train:
            problems.append(f"{slug} ships TRAIN_TASKS={staged_train_tasks(slug)}; "
                            f"expected {group.train}")
        staged_size = _staged_train_set_size()
        if staged_size != len(group.train):
            problems.append(f"{slug} ships TRAIN_SET_SIZE={staged_size}; expected "
                            f"{len(group.train)}, so check_split would refuse to start "
                            f"its daemon")
        toml = (path / "task.toml").read_text()
        if f'"rlebench/task02-{slug}"' not in toml:
            problems.append(f"{slug}/task.toml does not name itself")
        if f'"{IMAGE}"' not in toml:
            problems.append(f"{slug}/task.toml does not name the image")
        if f'RLEBENCH_GROUP = "{slug}"' not in toml:
            problems.append(f"{slug}/task.toml does not select its group")
        if group.eval_task in toml:
            problems.append(f"{slug}/task.toml names its evaluation task")

    # The trial count lives in three places and a mismatch is silent: surplus steps run
    # `next-trial` past the end of the plan, missing ones leave trials unreached and
    # scored 0. This is the check that exists to catch that.
    expected = ["develop"] + eval_step_names(C.TRIALS_PER_TASK)
    declared = re.findall(r'^name = "(\w+)"', TASK_TOML.read_text(), re.M)
    if declared != expected:
        problems.append(f"task.toml declares {declared}; expected {expected} "
                        f"for TRIALS_PER_TASK={C.TRIALS_PER_TASK}")
    for name in ("develop", STEP_TEMPLATE):
        if not (TEMPLATE / "steps" / name / "instruction.md").is_file():
            problems.append(f"_template/steps/{name}/instruction.md is missing")

    # ONE trial directory is checked in, and the trials must be interchangeable or the run
    # measures the instructions rather than the harness. Copies in the repo could drift
    # from eval_01; copies `emit` makes cannot, so the invariant is enforced by there
    # being nothing else here to disagree with.
    extra = sorted(p.name for p in (TEMPLATE / "steps").glob("eval_*")
                   if p.name != STEP_TEMPLATE)
    if extra:
        problems.append(f"_template/steps holds {extra}; {STEP_TEMPLATE} is the only eval "
                        "directory that belongs in the repo -- emit copies it per trial")

    # The emitted trees are where the copies must actually appear, and a directory left
    # over from a different --trials would run `next-trial` past the end of the plan.
    for slug, path in sorted(present_groups.items()):
        got = sorted(p.name for p in (path / "steps").glob("eval_*"))
        if got != expected[1:]:
            problems.append(f"{slug}/steps holds {got}; expected {expected[1:]} "
                            f"for TRIALS_PER_TASK={C.TRIALS_PER_TASK}; run --emit-all")

    if len(C.eval_plan()) != len(C.EVAL_TASKS) * C.TRIALS_PER_TASK:
        problems.append("eval_plan does not yield TRIALS_PER_TASK trials per task")
    return problems


def regenerate_groups() -> str:
    """Rebuild the GROUPS literal from the vendored docs, so the table has provenance."""
    import json

    docs = _docs_dir()

    attrs = json.loads(re.search(
        r"= (\{.*\});?\s*$",
        (docs / "composite_tasks/composite_task_attributes.js").read_text(), re.S).group(1))
    html = (docs / "tasks/_generated/composite_tasks_details.md").read_text()
    # INTERSECT WITH THE PINNED REGISTRY: the docs ship a newer RoboCasa than
    # sim/robocasa/pins.env, 300 composite tasks against the 252 this checkout
    # registers, so taking them at face value yields tasks that cannot be constructed.
    sys.path.insert(0, str(REPO))
    from robocasa.utils.dataset_registry import COMPOSITE_TASK_DATASETS

    found = []
    for blk in html.split('<details class="rc-activity"')[1:]:
        title = re.search(r'rc-activity-title">([^<]+)<', blk).group(1)
        tasks = [t for t in re.findall(r"<td><code>(\w+)</code></td>", blk)
                 if t in COMPOSITE_TASK_DATASETS and t in attrs]
        if len(tasks) >= 5:
            found.append((title, tasks))
    found.sort(key=lambda g: (g[0] != DEFAULT_GROUP.name, -len(g[1]), g[0]))
    lines = ["GROUPS: tuple[Group, ...] = ("]
    for title, tasks in found:
        lines += ["    Group(", f'        name="{title}",', "        tasks=("]
        lines += [f'            ("{t}", {attrs[t]["subtasks"]}),' for t in tasks]
        lines += ["        ),", "    ),"]
    return "\n".join(lines + [")"])


# -- choosing the training set --------------------------------------------------
# `train` is searched for under a hard coverage constraint and a stated preference order,
# and the result is pasted back into GROUPS so the table stays a literal anyone can read.

def choose_train(group: Group, audit: dict[str, dict],
                 size: int | None = None) -> tuple[tuple[str, ...], str | None]:
    """(the chosen training set, why the group cannot ship).

    Exhaustive over `size`-subsets of the pool -- at most 16 members, so a few hundred
    combinations. Coverage is an absolute constraint: a subset leaving any primitive with
    zero providers is not a candidate at any score. Candidates rank by

      1. REDUNDANCY, counting a second provider and no more -- a third adds nothing.
      2. DISTINCT PRIMITIVES, since breadth is still material to factor controllers from.
      3. SUBTASK MASS. Longer procedures give more to factor.
      4. Name, so the search is deterministic.

    Groups that train on atomics never reach here; the audit fixes their set.
    """
    import itertools

    size = _train_set_size() if size is None else size
    need = audit.get(group.eval_task, {}).get("need", [])
    counts = dict(group.tasks)
    pool = group.pool
    if len(pool) < size:
        return (), f"{len(pool)} members besides {group.eval_task}; need {size}"

    best = None
    for combo in itertools.combinations(sorted(pool), size):
        prov = {p: sum(1 for t in combo if p in audit.get(t, {}).get("need", []))
                for p in need}
        if any(v == 0 for v in prov.values()):
            continue
        key = (sum(min(v, 2) for v in prov.values()),
               len({p for t in combo for p in audit.get(t, {}).get("need", [])}),
               sum(counts[t] for t in combo),
               tuple(sorted(combo)))
        if best is None or key > best[0]:
            best = (key, combo)

    if best is None:
        # Name WHICH primitive is unreachable: the fix is re-designating the held-out task
        # or dropping the group, and both need it.
        orphan = [p for p in need
                  if not any(p in audit.get(t, {}).get("need", []) for t in pool)]
        return (), (f"{group.eval_task} needs {', '.join(orphan)}, which no member of the "
                    f"group provides" if orphan else
                    f"no {size}-task subset covers {group.eval_task}")
    return best[1], None


def recompute_split(size: int | None = None) -> str:
    """Re-choose every group's `train` and print a fresh GROUPS literal.

    Prints rather than rewrites, like `--regenerate-groups`: the table is source, and a
    generator that edits source in place is one nobody reviews.
    """
    size = _train_set_size() if size is None else size
    audit = load_audit()
    lines = ["GROUPS: tuple[Group, ...] = ("]
    for g in GROUPS:
        # Atomics-trained groups are left as authored -- the audit fixes their set, and
        # searching would silently turn them back into ordinary groups.
        if g.atomics:
            train, why = g.train, None
        else:
            train, why = choose_train(g, audit, size)
        why = why or undecomposable(g, audit)
        lines.append("    Group(")
        if why:
            lines += _wrap_comment(f"INACTIVE: {why}.", "        ")
        if g.atomics:
            lines += _wrap_comment(
                f"WITH ATOMICS: trained on the held-out task's primitives as atomic "
                f"tasks instead of on composites -- {', '.join(g.atomics)}.", "        ")
        lines += [f'        name="{g.name}",', f'        held_out="{g.held_out}",']
        lines.append(f"        train={_tuple_literal(train)},")
        # ALWAYS WRITTEN, and a hand-set False SURVIVES. `why` is re-derived, so a group
        # switched off by hand is one the audit has no complaint about -- deriving the
        # field from `why` alone would switch it back on and the regeneration would read
        # like the audit had cleared it.
        lines.append(f"        active={bool(g.active and not why)},")
        # Hand-written notes survive regeneration. Dropping them would lose the measured
        # difficulty of every held-out task, and leave a switched-off group saying only
        # "off by hand, no reason recorded".
        if g.info:
            lines.append(f"        info={_wrapped_string(g.info)},")
        lines.append("        tasks=(")
        lines += [f'            ("{t}", {n}),' for t, n in g.tasks]
        lines += ["        ),", "    ),"]
    return "\n".join(lines + [")"])


def _wrapped_string(text: str, indent: str = "                ") -> str:
    """A Python string literal for `text`, split across lines by implicit concatenation.
    `--recompute-split` output is pasted back into this file by hand, so it must read as
    source rather than run off the right margin."""
    import textwrap

    parts = textwrap.wrap(text, width=74)
    if len(parts) == 1:
        return f'"{parts[0]}"'
    body = [f'"{parts[0]} "'] + [
        f'{indent}"{p}{"" if i == len(parts) - 2 else " "}"'
        for i, p in enumerate(parts[1:])]
    return "(" + "\n".join(body) + ")"


def _wrap_comment(text: str, indent: str, width: int = 88) -> list[str]:
    import textwrap

    return [f"{indent}# {line}" for line in
            textwrap.wrap(text, width=width - len(indent) - 2)]


def _tuple_literal(items: tuple[str, ...], indent: str = "        ") -> str:
    """A tuple literal, wrapped one-per-line when the flat form would run off the line.
    `--recompute-split` output is pasted back here by hand, so it must read as source."""
    if not items:
        return "()"
    if len(items) == 1:
        return f'("{items[0]}",)'
    flat = "(" + ", ".join(f'"{t}"' for t in items) + ")"
    if len(indent) + len("train=") + len(flat) + 1 <= 92:
        return flat
    inner = "\n".join(f'{indent}    "{t}",' for t in items)
    return f"(\n{inner}\n{indent})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int,
                    help="graded trials per evaluation task; rewrites "
                         "TRIALS_PER_TASK, the [[steps]] list and steps/ together")
    ap.add_argument("--check", action="store_true", help="verify without writing")
    ap.add_argument("--list", action="store_true", help="show the table")
    ap.add_argument("--regenerate-groups", action="store_true",
                    help="print a fresh GROUPS literal from the vendored docs")
    ap.add_argument("--recompute-split", action="store_true",
                    help="re-choose every group's `train` from the primitive audit and "
                         "print a fresh GROUPS literal")
    ap.add_argument("--emit", metavar="GROUP",
                    help="write tasks/task02/NN-slug/ for one activity group")
    ap.add_argument("--emit-all", action="store_true",
                    help="write a task directory for every gradeable group")
    args = ap.parse_args()

    # `--check` returns before adoption, so pairing it with a writing flag would discard
    # that flag and print "split ok: <the group you did not adopt>" -- which reads like
    # confirmation. Refuse rather than pick a winner.
    if args.check and (args.trials or args.emit or args.emit_all):
        ap.error("--trials/--emit write, --check verifies; run them as two commands")

    if args.trials is not None and args.trials < 1:
        ap.error("--trials must be at least 1")

    if args.regenerate_groups:
        print(regenerate_groups())
        return 0

    if args.recompute_split:
        print(recompute_split())
        return 0

    if args.list:
        import textwrap

        audit = load_audit()
        n_thin = 0
        for i, g in enumerate(GROUPS, 1):
            counts = dict(g.tasks)
            mark = "*" if g.name == DEFAULT_GROUP.name else " "
            head = (f"{mark}{i:2d} {g.name:34s} {len(g.tasks):2d} pool  "
                    f"eval={g.eval_task}({counts[g.eval_task]})")
            if not g.active:
                print(f"{head}  INACTIVE: {gradeable(g)}")
                continue
            skinny = thin(g, audit)
            n_thin += bool(skinny)
            print(f"{head}  "
                  + (f"drill={', '.join(g.atomics)}" if g.atomics
                     else f"train={', '.join(g.composites)}")
                  + (f"  [thin: {', '.join(skinny)}]" if skinny else ""))
            # An active group's notes, which is the only place the measured difficulty of
            # its held-out task is written down. Inactive entries print theirs as the
            # INACTIVE reason above, so this branch would repeat it.
            if g.info:
                for line in textwrap.wrap(g.info, width=92,
                                          initial_indent=" " * 6, subsequent_indent=" " * 6):
                    print(line)
        n_active = len(active_groups())
        print(f"\n* = default. {len(GROUPS)} groups, {n_active} active, "
              f"{len(GROUPS) - n_active} inactive; every active one covers its evaluation "
              f"task's primitives, {n_thin} with a single provider for one of them.")
        return 0

    if args.check:
        problems = differences()
        if problems:
            print("SPLIT OUT OF SYNC:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        sys.path.insert(0, str(REPO))
        from harness import config as C

        print(f"split ok: {len(C.SPLITS)} groups, {C.TRAIN_SET_SIZE} train + 1 eval "
              f"each x {C.TRIALS_PER_TASK} trials, {len(C.eval_plan())} graded steps")
        return 0

    if args.trials is not None:
        write_trials(args.trials)
        print(f"trials per task set to {args.trials}: config.py, the template's "
              f"[[steps]] and steps/ rewritten together")
        if not (args.emit or args.emit_all):
            return 0

    if args.emit or args.emit_all:
        # By name or by slug, so the Makefile's per-group targets can name the directory.
        wanted = [(i, g) for i, g in enumerate(GROUPS, 1)
                  if args.emit_all or args.emit.lower() in (g.name.lower(),
                                                            slug_for(i, g))]
        if not wanted:
            raise SystemExit(f"no activity group named {args.emit!r}; --list to see them")
        emit_image()
        skipped = []
        for i, g in wanted:
            why = gradeable(g)
            if why:
                skipped.append(f"{slug_for(i, g)}: {why}")
                continue
            print(f"emitted {emit(i, g).name}")
        for line in skipped:
            print(f"SKIPPED {line}")
        if skipped and not args.emit_all:
            return 1
        print(f"\n{len(wanted) - len(skipped)} emitted, {len(skipped)} skipped")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
