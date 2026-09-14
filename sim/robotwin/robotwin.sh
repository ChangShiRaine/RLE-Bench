#!/usr/bin/env bash
# The RoboTwin 2.0 layer used by the task05 RoboTwin subtasks.
#
#     sim/robotwin/robotwin.sh setup     vendor third_party/robotwin unless it is at the pins
#     sim/robotwin/robotwin.sh fetch     vendor it again
#     sim/robotwin/robotwin.sh verify    is third_party/robotwin at the pins?
#     sim/robotwin/robotwin.sh sim       the simulator image (SAPIEN + cuRobo + RoboTwin code and assets)
#     sim/robotwin/robotwin.sh base      the task05 RoboTwin verifier base (FROM sim + verifier harness)
#     sim/robotwin/robotwin.sh images    setup, sim, base
#     sim/robotwin/robotwin.sh env
#     sim/robotwin/robotwin.sh clean     remove both images and third_party/robotwin
set -euo pipefail

ROBOTWIN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_REPO="$(cd "$ROBOTWIN_SCRIPT_DIR/../.." && pwd)"
# shellcheck source=pins.env
source "$ROBOTWIN_SCRIPT_DIR/pins.env"
ROBOTWIN_VENDOR="$ROBOTWIN_REPO/third_party/robotwin"
ROBOTWIN_TMP=""
trap 'rm -rf "$ROBOTWIN_TMP"' EXIT

robotwin_die() { echo "$*" >&2; exit 1; }
robotwin_step() { echo; echo "=== $* ==="; }
robotwin_usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

# A vendored tree records these in .pins; any change means vendoring again.
robotwin_pins() {
    printf '%s\n' "$ROBOTWIN_SHA" "$CUROBO_SHA" "$ROBOTWIN_ASSETS_REVISION"
    cat "$ROBOTWIN_SCRIPT_DIR"/patches/*/*.patch | sha256sum
}

robotwin_current() {
    [ -f "$ROBOTWIN_VENDOR/.pins" ] && [ "$(cat "$ROBOTWIN_VENDOR/.pins")" = "$(robotwin_pins)" ]
}

# One commit of a repository with this layer's patches for it, without history.
robotwin_checkout() {
    local url="$1" sha="$2" dest="$3" patch
    git init --quiet "$dest"
    git -C "$dest" fetch --quiet --depth 1 "$url" "$sha"
    git -C "$dest" checkout --quiet --detach FETCH_HEAD
    for patch in "$ROBOTWIN_SCRIPT_DIR/patches/$4"/*.patch; do
        git -C "$dest" apply "$patch"
    done
    rm -rf "$dest/.git"
}

# Assembled beside the tree and moved in last, so an interrupted fetch leaves nothing current.
cmd_fetch() {
    command -v git >/dev/null || robotwin_die "git is required"
    local hf="$ROBOTWIN_REPO/.venv/bin/hf"
    [ -x "$hf" ] || hf="$(command -v hf)" || robotwin_die "hf is required: make install"
    mkdir -p "$ROBOTWIN_REPO/third_party"
    ROBOTWIN_TMP="$(mktemp -d "$ROBOTWIN_REPO/third_party/.robotwin.XXXXXX")"
    local tree="$ROBOTWIN_TMP/robotwin" zips="$ROBOTWIN_TMP/zips" name

    robotwin_step "RoboTwin ${ROBOTWIN_SHA:0:12} + cuRobo ${CUROBO_SHA:0:12}"
    robotwin_checkout "$ROBOTWIN_GIT_URL" "$ROBOTWIN_SHA" "$tree" robotwin
    robotwin_checkout "$CUROBO_GIT_URL" "$CUROBO_SHA" "$tree/curobo" curobo

    robotwin_step "assets at $ROBOTWIN_ASSETS_HF_REPO@${ROBOTWIN_ASSETS_REVISION:0:12}"
    "$hf" download "$ROBOTWIN_ASSETS_HF_REPO" background_texture.zip embodiments.zip objects.zip \
        --repo-type dataset --revision "$ROBOTWIN_ASSETS_REVISION" --local-dir "$zips" --quiet >/dev/null
    (cd "$zips" && sha256sum --check --quiet) <<EOF
$ROBOTWIN_ASSETS_BACKGROUND_TEXTURE_SHA256  background_texture.zip
$ROBOTWIN_ASSETS_EMBODIMENTS_SHA256  embodiments.zip
$ROBOTWIN_ASSETS_OBJECTS_SHA256  objects.zip
EOF
    for name in background_texture embodiments objects; do
        python3 -m zipfile -e "$zips/$name.zip" "$tree/assets"
    done
    rm -rf "$tree/assets/__MACOSX"

    robotwin_pins > "$tree/.pins"
    rm -rf "$ROBOTWIN_VENDOR"
    mv "$tree" "$ROBOTWIN_VENDOR"
    echo "vendored $ROBOTWIN_VENDOR"
}

cmd_verify() {
    robotwin_current || robotwin_die "third_party/robotwin is missing or stale: sim/robotwin/robotwin.sh setup"
    echo "third_party/robotwin is at the pins"
}

cmd_setup() { if robotwin_current; then cmd_verify; else cmd_fetch; fi; }

robotwin_build() {
    local target="$1" tag="$2"
    command -v docker >/dev/null || robotwin_die "docker is required"
    cmd_verify
    robotwin_step "$tag"
    docker build --build-arg "BASE_IMAGE=$ROBOTWIN_CUDA_IMAGE" -f "$ROBOTWIN_REPO/sim/robotwin/Dockerfile" \
        --target "$target" -t "$tag" "$ROBOTWIN_REPO"
    docker images "$tag" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}"
}

cmd_sim() { robotwin_build sim "$ROBOTWIN_SIM_IMAGE"; }
cmd_base() { robotwin_build verifier "$ROBOTWIN_VERIFIER_BASE_IMAGE"; }
cmd_images() { cmd_setup; cmd_sim; cmd_base; }

cmd_env() {
    printf '%s\n' "ROBOTWIN_SIM_IMAGE=$ROBOTWIN_SIM_IMAGE" \
        "ROBOTWIN_VERIFIER_BASE_IMAGE=$ROBOTWIN_VERIFIER_BASE_IMAGE" \
        "ROBOTWIN_SHA=$ROBOTWIN_SHA" "CUROBO_SHA=$CUROBO_SHA" \
        "ROBOTWIN_ASSETS_REVISION=$ROBOTWIN_ASSETS_REVISION"
}

cmd_clean() {
    local image
    for image in "$ROBOTWIN_VERIFIER_BASE_IMAGE" "$ROBOTWIN_SIM_IMAGE"; do
        docker image inspect "$image" >/dev/null 2>&1 && docker rmi -f "$image" || echo "  $image absent"
    done
    rm -rf "$ROBOTWIN_VENDOR"
}

case "${1:-}" in
    setup)  shift; cmd_setup "$@" ;;
    fetch)  shift; cmd_fetch "$@" ;;
    verify) shift; cmd_verify "$@" ;;
    sim)    shift; cmd_sim "$@" ;;
    base)   shift; cmd_base "$@" ;;
    images) shift; cmd_images "$@" ;;
    env)    shift; cmd_env "$@" ;;
    clean)  shift; cmd_clean "$@" ;;
    help|-h|--help|"") robotwin_usage ;;
    *) robotwin_usage; exit 1 ;;
esac
