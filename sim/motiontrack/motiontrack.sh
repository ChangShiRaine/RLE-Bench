#!/usr/bin/env bash
# ONE entry point for the task04 motion-tracking layer: vendored assets, the dev
# venv, and the resolved paths everything else derives from.
#
#     sim/motiontrack/motiontrack.sh assets              G1 model + the LAFAN1 clips
#     sim/motiontrack/motiontrack.sh setup               assets + venv
#     sim/motiontrack/motiontrack.sh setup --no-assets   venv only
#     sim/motiontrack/motiontrack.sh motion              build all five scored .npz files
#     sim/motiontrack/motiontrack.sh images              the task04 agent + verifier images
#     sim/motiontrack/motiontrack.sh env                 the resolved paths
#     sim/motiontrack/motiontrack.sh verify              are the assets complete?
#     sim/motiontrack/motiontrack.sh clean               remove the assets and the venv
#
# Everything lands INSIDE the repo: assets under third_party/ (gitignored), venv
# at .venv-motiontrack. A checkout is then self-describing, so no container
# mount or make target has to be told where the assets went, and no absolute
# path from one machine can be committed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# shellcheck source=pins.env
source "$HERE/pins.env"
MOTION_MANIFEST="$REPO/tasks/task04/motions.tsv"

VENDOR="${MOTIONTRACK_VENDOR:-$REPO/third_party/motiontrack}"
ROBOT_DIR="$VENDOR/unitree_g1"
MOTION_DIR="$VENDOR/lafan1"
VENV="$REPO/.venv-motiontrack"
PY="$VENV/bin/python"

MENAGERIE_URL=https://github.com/google-deepmind/mujoco_menagerie
LAFAN1_URL=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
LAFAN1_API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset

AGENT_IMAGE="${MOTIONTRACK_AGENT_IMAGE:-rlebench-task04-agent:dev}"
VERIFIER_IMAGE="${MOTIONTRACK_VERIFIER_IMAGE:-rlebench-task04-verifier:dev}"

step() { echo; echo "=== $* ==="; }
die() { echo "$*" >&2; exit 1; }
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

# A half-downloaded tree fails deep inside a container as a missing mesh. Each
# fetch stamps the revision it completed, and that stamp is the only thing any
# caller trusts -- a present directory proves nothing.
stamped() { [ -f "$1/.rev" ] && [ "$(cat "$1/.rev")" = "$2" ]; }

fetch_robot() {
    stamped "$ROBOT_DIR" "$MENAGERIE_SHA" && { echo "G1 model current"; return; }
    step "G1 model @ ${MENAGERIE_SHA:0:12}"
    rm -rf "$ROBOT_DIR"
    local tmp="$VENDOR/.menagerie"
    rm -rf "$tmp" && mkdir -p "$tmp"
    # Sparse + single-commit: the full menagerie is ~1 GB of meshes we do not want.
    git -C "$tmp" init -q
    git -C "$tmp" remote add origin "$MENAGERIE_URL"
    git -C "$tmp" sparse-checkout init --cone
    git -C "$tmp" sparse-checkout set unitree_g1
    git -C "$tmp" fetch -q --depth 1 origin "$MENAGERIE_SHA"
    git -C "$tmp" checkout -q FETCH_HEAD
    mv "$tmp/unitree_g1" "$ROBOT_DIR"
    rm -rf "$tmp"
    echo "$MENAGERIE_SHA" > "$ROBOT_DIR/.rev"
}

fetch_motions() {
    stamped "$MOTION_DIR" "$LAFAN1_REV" && { echo "LAFAN1 clips current"; return; }
    step "LAFAN1 G1-retargeted clips @ ${LAFAN1_REV:0:12}"
    mkdir -p "$MOTION_DIR"
    # The whole 40-clip set is 90 MB and ships to the agent: an algorithm that
    # pretrains or does BC across clips needs more than the one scored motion.
    local names
    names=$(curl -fsSL "$LAFAN1_API/tree/$LAFAN1_REV/g1" \
        | python3 -c 'import json,sys; print("\n".join(e["path"].rsplit("/",1)[-1] for e in json.load(sys.stdin)))')
    [ -n "$names" ] || die "could not list the LAFAN1 g1/ directory"
    echo "$names" | xargs -P 8 -I{} \
        curl -fsSL "$LAFAN1_URL/resolve/$LAFAN1_REV/g1/{}" -o "$MOTION_DIR/{}"
    echo "$LAFAN1_REV" > "$MOTION_DIR/.rev"
    echo "$(echo "$names" | wc -l) clips"
}

cmd_assets() {
    mkdir -p "$VENDOR"
    fetch_robot
    fetch_motions
    cmd_verify
}

cmd_verify() {
    stamped "$ROBOT_DIR" "$MENAGERIE_SHA" || die "\
$ROBOT_DIR is not the pinned G1 model. Run \`sim/motiontrack/motiontrack.sh assets\`."
    stamped "$MOTION_DIR" "$LAFAN1_REV" || die "\
$MOTION_DIR is not the pinned LAFAN1 set. Run \`sim/motiontrack/motiontrack.sh assets\`."
    [ -f "$ROBOT_DIR/g1.xml" ] || die "$ROBOT_DIR/g1.xml missing"
    [ -f "$MOTION_MANIFEST" ] || die "$MOTION_MANIFEST missing"
    local slug motion frames title
    while IFS=$'\t' read -r slug motion frames title; do
        [ -z "$slug" ] && continue
        [[ "$slug" == \#* ]] && continue
        [ -f "$MOTION_DIR/$motion.csv" ] \
            || die "scored clip $motion.csv missing from $MOTION_DIR"
    done < "$MOTION_MANIFEST"
    echo "assets ok: $(ls "$MOTION_DIR"/*.csv | wc -l) clips, G1 model present"
}

# The five reference clips the task matrix tracks. Motions and 1-based inclusive
# frame ranges come from motions.tsv, never duplicated in generated task files.
# Conversion uses the gain-corrected model robot.py compiles, not the raw XML.
cmd_motion() {
    cmd_verify
    [ -x "$PY" ] || die "$PY missing; run sim/motiontrack/motiontrack.sh setup first"
    mkdir -p "$VENDOR/motions"
    local slug motion frames title out
    while IFS=$'\t' read -r slug motion frames title; do
        [ -z "$slug" ] && continue
        [[ "$slug" == \#* ]] && continue
        out="$VENDOR/motions/$motion.npz"
        step "$slug: $motion frames $frames"
        PYTHONPATH="$REPO/tasks/task04" "$PY" -m harness.motion \
            --robot-dir "$ROBOT_DIR" \
            --csv "$MOTION_DIR/$motion.csv" \
            --frames "$frames" \
            --input-fps "$MOTIONTRACK_INPUT_FPS" \
            --fps "$MOTIONTRACK_FPS" \
            --out "$out"
    done < "$MOTION_MANIFEST"
}

# The two task04 images. Context is the REPO ROOT so the Dockerfiles copy the
# harness and the vendored assets straight from source -- there is no staging
# step, and .dockerignore keeps the ~23 GB of RoboCasa out of the context. Pins
# are passed as --build-arg from pins.env; the Dockerfiles declare those ARGs with
# NO defaults, so a bare `docker build` fails rather than forking the pins.
cmd_images() {
    cmd_verify
    local slug motion frames title missing=0
    while IFS=$'\t' read -r slug motion frames title; do
        [ -z "$slug" ] && continue
        [[ "$slug" == \#* ]] && continue
        [ -f "$VENDOR/motions/$motion.npz" ] || missing=1
    done < "$MOTION_MANIFEST"
    [ "$missing" -eq 0 ] || cmd_motion
    local args=()
    for key in TORCH_VER TORCH_INDEX MUJOCO_VER MJWARP_VER WARP_VER ONNX_VER \
               ONNXRUNTIME_VER ONNXSCRIPT_VER NUMPY_VER; do
        args+=(--build-arg "$key=${!key}")
    done
    step "agent image $AGENT_IMAGE"
    docker build -t "$AGENT_IMAGE" -f "$REPO/tasks/task04/_template/environment/Dockerfile" \
        "${args[@]}" "$REPO"
    step "verifier image $VERIFIER_IMAGE"
    docker build -t "$VERIFIER_IMAGE" -f "$REPO/tasks/task04/_template/tests/Dockerfile" \
        "${args[@]}" "$REPO"
    for image in "$AGENT_IMAGE" "$VERIFIER_IMAGE"; do
        docker images "$image" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}"
    done
}

cmd_setup() {
    [ "${1:-}" = "--no-assets" ] || cmd_assets
    step "dev venv $VENV"
    [ -x "$PY" ] || python3 -m venv "$VENV"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q --extra-index-url "$TORCH_INDEX" \
        "mujoco==$MUJOCO_VER" \
        "mujoco-warp==$MJWARP_VER" \
        "warp-lang==$WARP_VER" \
        "torch==$TORCH_VER" \
        "onnx==$ONNX_VER" \
        "onnxscript==$ONNXSCRIPT_VER" \
        "onnxruntime==$ONNXRUNTIME_VER" \
        "numpy==$NUMPY_VER" \
        pytest
    "$PY" -c 'import mujoco, warp, torch; print("mujoco", mujoco.__version__, "| warp", warp.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
}

# rm -rf would empty a writable bind mount of a shared tree; refuse any mount at or under it.
refuse_if_mounted() {
    local d; d="$(readlink -f "$1" 2>/dev/null || true)"
    [ -n "$d" ] && [ -e "$d" ] || return 0
    awk -v p="$d" '$2 == p || index($2, p "/") == 1 { f = 1 } END { exit !f }' /proc/mounts \
        && die "$1 has a filesystem mounted at or under it; unmount it before clean"
    return 0
}

# The layer's footprint only: the task04 images are `make task04-clean`.
cmd_clean() {
    refuse_if_mounted "$VENDOR"
    step "removing the motiontrack assets under $VENDOR and $VENV"
    rm -rf "$ROBOT_DIR" "$MOTION_DIR" "$VENDOR/motions" "$VENDOR/.menagerie" "$VENV"
    rmdir "$VENDOR" 2>/dev/null || true
}

cmd_env() {
    cat <<EOF
MOTIONTRACK_ROBOT_DIR=$ROBOT_DIR
MOTIONTRACK_MOTION_DIR=$MOTION_DIR
MOTIONTRACK_MANIFEST=$MOTION_MANIFEST
MOTIONTRACK_MOTION=$MOTIONTRACK_MOTION
MOTIONTRACK_MOTION_NPZ=$VENDOR/motions/$MOTIONTRACK_MOTION.npz
MOTIONTRACK_PY=$PY
EOF
}

case "${1:-}" in
    assets) shift; cmd_assets "$@" ;;
    motion) shift; cmd_motion "$@" ;;
    images) shift; cmd_images "$@" ;;
    setup)  shift; cmd_setup "$@" ;;
    verify) shift; cmd_verify "$@" ;;
    clean)  shift; cmd_clean "$@" ;;
    env)    shift; cmd_env "$@" ;;
    *)      usage; exit 1 ;;
esac
