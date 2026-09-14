"""Work out which atomic primitives a RoboCasa composite task requires.

This is what makes task02's central claim checkable rather than asserted. The claim is
that every evaluation task is composed of primitives the agent could have practised, and
with 252 composites to choose from, hand-authoring that map is not a thing anyone will
keep correct.

HOW IT READS A TASK. Source, not prose: `_check_success` is what actually has to be true,
and `_setup_scene` closing a door is what forces an open. The instruction text is mined
as well, and the two are UNIONED, because the predicate alone under-approximates --
`BlendIngredients` must open and close the blender lid, but the predicate never mentions
it, since the lid ends where it started.

Under-approximating is the dangerous direction: a primitive missed here is one the
training split will not contain, making the evaluation task unsolvable by construction
and its zero a statement about the split rather than the agent. So this over-reports
where it is unsure, and refuses outright where it does not understand.

WHERE IT REFUSES. Any `self.<attr>` in a predicate that is not a fixture it models is
reported as unknown and the task is not decomposed. That rule exists because of
`WashFish`, whose predicate is `self.fish_rinsed and ...` -- defined in the class body as
"rinsed for at least 25 timesteps in RUNNING water", so it needs `TurnOnSinkFaucet`, and
nothing in the `OU.` vocabulary says so.

    python tasks/task02/dev/decompose.py --eval TaskA,TaskB      # required primitives
    python tasks/task02/dev/decompose.py --audit                 # every composite, to stdout
    python tasks/task02/dev/decompose.py --audit --out FILE      # ... to a file

USE `--out` FOR THE CHECKED-IN AUDIT, never a shell redirect: importing robosuite prints a
banner to STDOUT, so `--audit > file` writes a file whose first line is not JSON. That
audit is tasks/task02/dev/data/task02_primitive_audit.json, which `build_groups.py` reads to gate coverage
without needing RoboCasa.

Validated against a hand-authored map of ten tasks: recovers every primitive, misses
none. It is still a derivation -- treat a new evaluation split's output as a strong draft
to check against source, not as ground truth.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# fixture attribute name -> the atomic tasks that touch it, by role
FIXTURES = {
    "sink":            dict(into="PickPlaceCounterToSink", out="PickPlaceSinkToCounter",
                            on="TurnOnSinkFaucet", off="TurnOffSinkFaucet"),
    "microwave":       dict(into="PickPlaceCounterToMicrowave", out="PickPlaceMicrowaveToCounter",
                            opn="OpenMicrowave", cls="CloseMicrowave", on="TurnOnMicrowave"),
    "cab":             dict(into="PickPlaceCounterToCabinet", out="PickPlaceCabinetToCounter",
                            opn="OpenCabinet", cls="CloseCabinet"),
    "cabinet":         dict(into="PickPlaceCounterToCabinet", out="PickPlaceCabinetToCounter",
                            opn="OpenCabinet", cls="CloseCabinet"),
    "drawer":          dict(into="PickPlaceCounterToDrawer", out="PickPlaceDrawerToCounter",
                            opn="OpenDrawer", cls="CloseDrawer"),
    "blender":         dict(into="PickPlaceCounterToBlender", opn="OpenBlenderLid",
                            cls="CloseBlenderLid", on="TurnOnBlender"),
    "stove":           dict(into="PickPlaceCounterToStove", out="PickPlaceStoveToCounter",
                            on="TurnOnStove", off="TurnOffStove"),
    "oven":            dict(into="PickPlaceCounterToOven", opn="OpenOven", cls="CloseOven",
                            on="PreheatOven"),
    "toaster_oven":    dict(into="PickPlaceCounterToToasterOven", out="PickPlaceToasterOvenToCounter",
                            opn="OpenToasterOvenDoor", cls="CloseToasterOvenDoor",
                            on="TurnOnToasterOven"),
    "toaster":         dict(out="PickPlaceToasterToCounter", on="TurnOnToaster"),
    "fridge":          dict(opn="OpenFridge", cls="CloseFridge"),
    "stand_mixer":     dict(into="PickPlaceCounterToStandMixer", opn="OpenStandMixerHead",
                            cls="CloseStandMixerHead"),
    "coffee_machine":  dict(on="StartCoffeeMachine"),
    "electric_kettle": dict(opn="OpenElectricKettleLid", cls="CloseElectricKettleLid",
                            on="TurnOnElectricKettle"),
    "dishwasher":      dict(opn="OpenDishwasher", cls="CloseDishwasher"),
    # A SEAT, and the frame "left" and "right" are measured in: several predicates read
    # `self.stool.rot` to orient the dining counter. Listed with NO roles because no atomic
    # task touches a stool -- which is the point. Without an entry the WashFish rule below
    # sees `self.stool` as hidden task state and refuses the task, and a static layout
    # rotation is the one thing that cannot be hidden state: the agent cannot change it and
    # it implies no primitive. Contact or containment against it still reports unknown.
    "stool":           dict(),
    "counter":         dict(into="__COUNTER__"),
    "dining_counter":  dict(into="__COUNTER__"),
    # The same surface under the other name several tasks give it. Across the pin's 19
    # `self.dining_table` refs it registers as DINING_COUNTER 15 times, COUNTER twice and
    # STOOL twice -- and no atomic task names placing an object onto any of the three, which
    # is exactly what the sentinel means, so the alias is safe in all 19.
    "dining_table":    dict(into="__COUNTER__"),
    # The stretch of counter BESIDE the sink, registered FixtureType.COUNTER -- a work
    # surface, not the basin. Without the entry, "taken out of" falls through to `self.sink`
    # and invents a retrieval from standing water.
    "sink_counter":    dict(into="__COUNTER__"),
    "distr_counter":   dict(into="__COUNTER__"),
}

# Placing an object into another OBJECT -- a bowl, a plate, a pan, a cutting board.
# CheesyBread is the ONLY atomic task of that shape; every other pick-place ends at a
# fixture. Resolving `check_obj_in_receptacle` to whichever storage fixture a task happens
# to register instead silently drops CheesyBread from the tasks that need it.
OBJECT_PLACEMENT = "CheesyBread"

TEXT_RULES = [
    (r"open the (?:\w+ )?cabinet", ["OpenCabinet"]),
    (r"clos(?:e|ing) the (?:\w+ )?cabinet", ["CloseCabinet"]),
    (r"open the (?:\w+ )?drawer", ["OpenDrawer"]),
    (r"clos(?:e|ing) the (?:\w+ )?drawer", ["CloseDrawer"]),
    (r"open the (?:\w+ )?microwave", ["OpenMicrowave"]),
    (r"clos(?:e|ing) the (?:\w+ )?microwave", ["CloseMicrowave"]),
    (r"open the (?:\w+ )?fridge|from the fridge|in the fridge", ["OpenFridge"]),
    (r"clos(?:e|ing) the (?:\w+ )?fridge", ["CloseFridge"]),
    (r"open the (?:\w+ )?oven(?! rack)", ["OpenOven"]),
    (r"clos(?:e|ing) the (?:\w+ )?oven", ["CloseOven"]),
    (r"open the (?:\w+ )?dishwasher", ["OpenDishwasher"]),
    (r"clos(?:e|ing) the (?:\w+ )?dishwasher", ["CloseDishwasher"]),
    (r"open the blender|tak(?:e|ing) off the lid", ["OpenBlenderLid"]),
    (r"close the lid|place the lid|closing the lid", ["CloseBlenderLid"]),
    (r"start the blender|turn on the blender", ["TurnOnBlender"]),
    (r"turn on the (?:sink|water|faucet)|run the water|turn the (?:sink|water) on",
     ["TurnOnSinkFaucet"]),
    (r"turn off the (?:sink|water|faucet)|turn the (?:sink|water) off",
     ["TurnOffSinkFaucet"]),
    (r"turn on the .{0,20}burner|turn on the stove|set the heat", ["TurnOnStove"]),
    (r"turn off the .{0,20}burner|turn off the stove", ["TurnOffStove"]),
    (r"start the coffee|button on the coffee", ["StartCoffeeMachine"]),
    (r"turn on the toaster oven|set the timer", ["TurnOnToasterOven"]),
    (r"push down the lever|turn on the toaster\b", ["TurnOnToaster"]),
    (r"turn on the microwave|press the start button", ["TurnOnMicrowave"]),
    (r"dining (?:counter|table)", ["NavigateKitchen"]),
]

# Attributes a predicate may name that are not fixtures we model, and are not evidence
# of hidden task state: the simulator handles and RoboCasa's own bookkeeping.
_BENIGN_ATTRS = {"objects", "sim", "obj_body_id", "knob", "check_contact", "rng",
                 "robots", "_ep_meta"}

# STATIC SEAT GEOMETRY, benign for the same reason but by prefix, because the seats are
# enumerated: `stool`, `stool1`, `stool2`, and the `stool_positions` / `stool_rotations`
# lists that `_setup_kitchen_references` fills from `_ep_meta["refs"]`. Every one is layout
# data fixed at reset -- it decides where "in front of a seat" IS, so a predicate must read
# it, but the agent cannot alter it and no atomic task touches a stool. Treating it as
# hidden state refused five composites, four of them otherwise clean.
_SEAT_PREFIX = "stool"


def _method_src(src: str, name: str) -> str:
    m = re.search(rf"\n    def {name}\(self.*?\):\n(.*?)(?=\n    def |\Z)", src, re.S)
    return m.group(1) if m else ""


def _signature(name: str) -> dict | None:
    from robosuite.environments.base import REGISTERED_ENVS

    cls = REGISTERED_ENVS.get(name)
    if cls is None:
        return None
    try:
        src = inspect.getsource(cls)
    except OSError:
        return None
    doc = cls.__doc__ or ""
    m = re.search(r"Steps:\s*\n(.*?)(?:\n\s*\n|\n\s*Args:|\Z)", doc, re.S)
    steps = " ".join(l.strip() for l in m.group(1).splitlines() if l.strip()) if m else ""
    meta = _method_src(src, "get_ep_meta")
    lang = " ".join(re.findall(r'"([^"]{20,})"', meta))
    return {"success": _method_src(src, "_check_success"),
            "setup": _method_src(src, "_setup_kitchen_references") + _method_src(src, "_setup_scene"),
            "cfgs": _method_src(src, "_get_obj_cfgs"),
            "text": f"{steps} {lang}"}


def _placement_fixture(cfgs: str, obj: str) -> str | None:
    """The fixture attribute `obj` is SPAWNED on, from `_get_obj_cfgs`, or None if that
    cannot be read.

    This is the difference between "the fork starts in the cabinet" and "the fork starts on
    the stretch of counter NEAREST the cabinet" -- `placement=dict(fixture=self.counter,
    sample_region_kwargs=dict(ref=self.cabinet))` is the second, and a scan of the whole
    method body cannot tell them apart, because both name `self.cabinet`.
    """
    # Each cfg is a dict literal opening with its name; slice from this object's name to
    # the next one so a sibling's placement cannot be misread as this one's. The optional
    # `f` matters: SetupWineGlasses writes `name=f"wine_glass1"` with nothing interpolated,
    # and missing it made the placement unreadable, so the caller fell back to the scan and
    # billed the task for a cabinet its glasses never enter.
    start = re.search(rf'name\s*=\s*f?["\']{re.escape(obj)}["\']', cfgs)
    if not start:
        return None
    rest = cfgs[start.end():]
    nxt = re.search(r'name\s*=\s*f?["\']', rest)
    block = rest[:nxt.start()] if nxt else rest
    m = re.search(r"placement\s*=\s*dict\(\s*fixture\s*=\s*self\.(\w+)", block)
    return m.group(1) if m else None


def _stays_put(src: str, obj: str, spawn: str) -> bool:
    """Does the predicate SHOW that `obj` ends where it spawned?

    Positive test, and the default is the other way: an object we cannot prove stayed is
    assumed to have been retrieved, since over-reporting is the safe direction. Two proofs:

      TESTED AGAINST ITS OWN SPAWN FIXTURE -- `check_obj_fixture_contact(self, "pan",
      self.stove)` on a pan that spawned on the stove asks that it is STILL there.

      NAMED ONLY AS A RECEPTACLE -- PlaceVegetablesEvenly's pan spawns on the stove and the
      predicate's sole mention of it is `check_obj_in_receptacle(self, veg, "pan")`, where it
      is the destination. The vegetables move; the pan is furniture.
    """
    if re.search(rf'(?:check_obj_fixture_contact|obj_inside_of)\(\s*self,\s*'
                 rf'["\']?{re.escape(obj)}["\']?\s*,\s*self\.{re.escape(spawn)}\b', src):
        return True
    # `self.stove.check_obj_location_on_stove(env=self, obj_name="pan", threshold=...)` --
    # keyword form, and the pan it asks about is one that spawned on the stove and stays.
    if re.search(rf'check_obj_location_on_stove\([^)]*["\']{re.escape(obj)}["\']', src):
        return True
    mentions = re.findall(rf'["\']{re.escape(obj)}["\']', src)
    as_receptacle = re.findall(
        rf'check_obj_in_receptacle\(\s*self,\s*[^,()]+,\s*["\']{re.escape(obj)}["\']', src)
    return bool(as_receptacle) and len(mentions) == len(as_receptacle)


def decompose(name: str) -> tuple[list[str], list[str]]:
    """(required primitives, reasons we are unsure). A non-empty second element means
    the task was not understood and must not be used without checking it by hand."""
    s = _signature(name)
    if s is None:
        return [], [f"not-registered:{name}"]
    src, setup, text = s["success"], s["setup"], s["text"].lower()
    cfgs = s["cfgs"]
    need: set[str] = set()
    unknown: list[str] = []
    resolved: set[str] = set()      # spawn fixtures a named object was traced to

    for fx in re.findall(r"obj_inside_of\(\s*self,\s*[\"']?\w+[\"']?,\s*self\.(\w+)", src):
        e = FIXTURES.get(fx, {})
        need.add(e["into"]) if e.get("into") and e["into"] != "__COUNTER__" \
            else unknown.append(f"inside_of:{fx}")
    for obj, fx in re.findall(
            r"check_obj_fixture_contact\(\s*self,\s*[\"']?(\w+)[\"']?,\s*self\.(\w+)", src):
        e = FIXTURES.get(fx, {})
        if e.get("into") == "__COUNTER__":
            # It ends on a counter, so it was taken OUT of wherever it spawned. Ask the
            # placement first: a fixture named in the method body may only be a sampling
            # reference, and reading it as the source invents a retrieval the task never
            # performs -- which then forces a training set to cover a door nobody opens.
            src_fx = _placement_fixture(cfgs, obj)
            if src_fx is not None and FIXTURES.get(src_fx, {}).get("out"):
                need.add(FIXTURES[src_fx]["out"])
            elif src_fx is not None and FIXTURES.get(src_fx, {}).get("into") == "__COUNTER__":
                # COUNTER TO COUNTER, and no atomic task names the carry itself: every
                # registered PickPlace* starts or ends at a real fixture. But BETWEEN two
                # different counters it is a traverse, so read that off the source the same
                # way TEXT_RULES reads it off the prose -- otherwise a task whose language
                # never says "dining table" decomposes to nothing and reads as vacuous when
                # it is merely unnamed.
                if fx != src_fx:
                    need.add("NavigateKitchen")
            else:
                # PLACEMENT UNREADABLE -- the predicate named a loop variable rather than a
                # literal, as SetupWineGlasses does with `for wine_glass_name in ...`. Guess,
                # but only from fixtures an object actually SPAWNS on. Scanning the whole
                # class body instead reads every landmark as a source: those wine glasses sit
                # on the counter with `sample_region_kwargs=dict(ref=self.cabinet)`, and the
                # old scan billed the task for a cabinet retrieval on the strength of that
                # reference alone.
                spawns = set(re.findall(r"fixture\s*=\s*self\.(\w+)", cfgs))
                outs = {FIXTURES[x]["out"] for x in spawns if FIXTURES.get(x, {}).get("out")}
                if outs:
                    need.update(outs)
                elif any(x != fx and FIXTURES.get(x, {}).get("into") == "__COUNTER__"
                         for x in spawns):
                    need.add("NavigateKitchen")      # counter to a DIFFERENT counter
                elif not spawns:
                    need.add(OBJECT_PLACEMENT)       # nothing readable at all
        elif e.get("into"):
            need.add(e["into"])
        else:
            unknown.append(f"fixture_contact:{fx}")
    # OBJECTS THAT SPAWN INSIDE A FIXTURE the agent must reach into. Anchored on the
    # object, not on a fixture-contact test, because the predicate may constrain such an
    # object only THROUGH another object: ArrangeBreadBowl's bread starts on a toaster-oven
    # rack and is checked by `check_contact` against a bowl, never against a fixture, so a
    # fixture-anchored scan drops the retrieval entirely. Only objects the predicate
    # actually names count -- a distractor left in a drawer is nobody's primitive.
    for obj in re.findall(r'name\s*=\s*f?["\'](\w+)["\']', cfgs):
        if not re.search(rf'["\']{re.escape(obj)}["\']', src):
            continue
        spawn = _placement_fixture(cfgs, obj) or ""
        resolved.add(spawn)
        if FIXTURES.get(spawn, {}).get("out") and not _stays_put(src, obj, spawn):
            need.add(FIXTURES[spawn]["out"])

    # SPAWN FIXTURES NO OBJECT ACCOUNTED FOR. `_get_obj_cfgs` may build its cfgs in a loop,
    # leaving no `name="..."` literal to slice a placement out of -- SetupBowls spawns every
    # bowl `fixture=self.cabinet` inside a `while` over the stools, and its predicate tests
    # `check_obj_any_counter_contact`, so neither pass above sees the cabinet at all. It is
    # still a task that reaches into a cavity. Credit the retrieval whenever a retrievable
    # spawn fixture went unclaimed: we cannot say WHICH object comes out of it, only that
    # something does, and under-reporting is the direction that breaks a split.
    for fx in set(re.findall(r"fixture\s*=\s*self\.(\w+)", cfgs)) - resolved:
        if FIXTURES.get(fx, {}).get("out"):
            need.add(FIXTURES[fx]["out"])

    if "check_obj_in_receptacle" in src:
        need.add(OBJECT_PLACEMENT)
    if "check_obj_any_counter_contact" in src:
        need.add(OBJECT_PLACEMENT)
    for fx, key in re.findall(r"self\.(\w+)\.get_state\(\)\[[\"'](\w+)[\"']\]", src):
        e = FIXTURES.get(fx, {})
        if key == "turned_on" and e.get("on"):
            need.add(e["on"])
        elif key != "turned_on":
            unknown.append(f"state:{fx}.{key}")
    if "get_handle_state" in src and "water_on" in src:
        need.add("TurnOnSinkFaucet")
    if "is_burner_on" in src or "get_knobs_state" in src:
        need.add("TurnOnStove")
    if "check_obj_location_on_stove" in src:
        need.add("PickPlaceCounterToStove")
    for fx in re.findall(r"self\.(\w+)\.is_closed\(", src):
        e = FIXTURES.get(fx, {})
        need.add(e["cls"]) if e.get("cls") else unknown.append(f"is_closed:{fx}")
    for fx in re.findall(r"self\.(\w+)\.is_open\(", src):
        e = FIXTURES.get(fx, {})
        need.add(e["opn"]) if e.get("opn") else unknown.append(f"is_open:{fx}")
    for fx in re.findall(r"self\.(\w+)\.close_door\(", setup):
        e = FIXTURES.get(fx, {})
        need.add(e["opn"]) if e.get("opn") else unknown.append(f"reset_closed:{fx}")

    # Hidden task state -- the WashFish rule. See the module docstring.
    for attr in re.findall(r"self\.(\w+)\b(?!\s*\()", src):
        if (attr not in FIXTURES and attr not in _BENIGN_ATTRS
                and not attr.startswith("_") and not attr.startswith(_SEAT_PREFIX)):
            unknown.append(f"custom-state:{attr}")

    for pat, prims in TEXT_RULES:
        if re.search(pat, text):
            need.update(prims)

    need.discard("__COUNTER__")
    return sorted(need), sorted(set(unknown))


def required_primitives(tasks) -> dict[str, tuple[str, ...]]:
    """The REQUIRED_PRIMITIVES map for an evaluation split. Raises on any task we could
    not decompose -- a split we do not understand must not be silently shipped."""
    out, refused = {}, {}
    for t in tasks:
        need, unknown = decompose(t)
        if unknown:
            refused[t] = unknown
        else:
            out[t] = tuple(need)
    if refused:
        raise SystemExit(
            "cannot decompose:\n" +
            "\n".join(f"  {t}: {', '.join(u)}" for t, u in refused.items()) +
            "\nCheck these against source and add them to REQUIRED_PRIMITIVES by hand, "
            "or choose different tasks.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", help="comma-separated composite tasks")
    ap.add_argument("--audit", action="store_true",
                    help="decompose every composite task and write JSON")
    ap.add_argument("--out", metavar="FILE",
                    help="write the audit here instead of stdout; the only safe way to "
                         "capture it, since importing robosuite prints to stdout")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    importlib.import_module("robocasa.environments.kitchen.kitchen")  # registers every env

    if args.audit:
        from robocasa.utils.dataset_registry import COMPOSITE_TASK_DATASETS

        rows = {}
        for t in sorted(COMPOSITE_TASK_DATASETS):
            need, unknown = decompose(t)
            rows[t] = {"need": need, "unknown": unknown}
        doc = json.dumps(rows, indent=1) + "\n"
        if args.out:
            Path(args.out).write_text(doc)
            print(f"wrote {len(rows)} composite tasks to {args.out}")
        else:
            print(doc, end="")
        return 0

    if args.out:
        ap.error("--out is for --audit")

    if not args.eval:
        ap.error("give --eval or --audit")
    for task, prims in required_primitives(
            [t.strip() for t in args.eval.split(",") if t.strip()]).items():
        print(f'    "{task}": {prims!r},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
