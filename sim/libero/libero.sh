#!/usr/bin/env bash
# Build and validate the shared task05 nanoVLA layer.
#
#     sim/libero/libero.sh bundles      the three encoder-bundle images; downloads the
#                                        encoders at their pinned commits when
#                                        NANOVLA_HF_{BASE,DINOV2,OPEN} are unset
#     sim/libero/libero.sh base         the three agent images + the verifier base (make sim-libero)
#     sim/libero/libero.sh verifiers    the four per-subtask verifier images (make task05; needs sim/robotwin too)
#     sim/libero/libero.sh images       all of the above
#     sim/libero/libero.sh check 01-libero-open-design   validate the external assets one subtask mounts
#     sim/libero/libero.sh env
#     sim/libero/libero.sh clean        remove the agent images, the bundles and the verifier base
set -euo pipefail

LIBERO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_REPO="$(cd "$LIBERO_SCRIPT_DIR/../.." && pwd)"

# shellcheck source=pins.env
source "$LIBERO_SCRIPT_DIR/pins.env"

# The fetcher needs huggingface_hub (requirements-dev.txt); everything else this
# script runs is stdlib, so a bare python3 stays the fallback.
LIBERO_PYTHON="$LIBERO_REPO/.venv/bin/python"
[ -x "$LIBERO_PYTHON" ] || LIBERO_PYTHON=python3

LIBERO_SUBTASKS=(01-libero-open-design 02-libero-robustness 03-robotwin-open-design 04-robotwin-robustness)
LIBERO_BUNDLES=(base dinov2 open)

libero_die() { echo "$*" >&2; exit 1; }
libero_step() { echo; echo "=== $* ==="; }
libero_usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

# The encoder bundles: one image per validated HF_HOME directory. Build-time
# inputs only; the subtasks never mount them.
#
# NANOVLA_HF_{BASE,DINOV2,OPEN} point at prepared directories. Unset, the bundle is
# downloaded from the Hub into third_party/libero/ at the commits validate_assets.py
# pins, so a fresh clone needs no manual asset step; set, the directory is used as it
# is and is only downloaded into if it does not already validate.
cmd_bundles() {
    command -v docker >/dev/null || libero_die "docker is required"
    local name var dir images=()
    for name in "${LIBERO_BUNDLES[@]}"; do
        var="NANOVLA_HF_$(echo "$name" | tr '[:lower:]' '[:upper:]')"
        dir="${!var:-$LIBERO_REPO/third_party/libero/hf-$name}"
        if ! python3 "$LIBERO_REPO/tasks/task05/harness/validate_assets.py" \
                bundle "$name" "$dir" >/dev/null 2>&1; then
            libero_step "fetching bundle $name into $dir"
            "$LIBERO_PYTHON" "$LIBERO_SCRIPT_DIR/fetch_bundles.py" "$name" "$dir"
        fi
        python3 "$LIBERO_REPO/tasks/task05/harness/validate_assets.py" bundle "$name" "$dir"
        libero_step "bundle image $LIBERO_HF_IMAGE_PREFIX-$name:dev"
        docker build -f "$LIBERO_REPO/sim/libero/Dockerfile.hf" -t "$LIBERO_HF_IMAGE_PREFIX-$name:dev" "$dir"
        images+=("$LIBERO_HF_IMAGE_PREFIX-$name:dev")
    done
    libero_list_images "${images[@]}"
}

cmd_base() {
    command -v docker >/dev/null || libero_die "docker is required"
    local build_args=(
        --build-arg "BASE_IMAGE=$LIBERO_CUDA_IMAGE"
        --build-arg "LIBERO_SHA=$LIBERO_SHA"
        --build-arg "LIBERO_PLUS_SHA=$LIBERO_PLUS_SHA"
        -f "$LIBERO_REPO/sim/libero/Dockerfile"
    )
    local name images=()
    for name in "${LIBERO_BUNDLES[@]}"; do
        docker image inspect "$LIBERO_HF_IMAGE_PREFIX-$name:dev" >/dev/null 2>&1 \
            || libero_die "$LIBERO_HF_IMAGE_PREFIX-$name:dev missing; run sim/libero/libero.sh bundles first"
        libero_step "agent image $LIBERO_AGENT_IMAGE_PREFIX-$name:dev"
        docker build "${build_args[@]}" --build-arg "HF_IMAGE=$LIBERO_HF_IMAGE_PREFIX-$name:dev" \
            --target agent -t "$LIBERO_AGENT_IMAGE_PREFIX-$name:dev" "$LIBERO_REPO"
        images+=("$LIBERO_AGENT_IMAGE_PREFIX-$name:dev")
    done
    libero_step "verifier base image $LIBERO_VERIFIER_BASE_IMAGE"
    docker build "${build_args[@]}" --target verifier \
        -t "$LIBERO_VERIFIER_BASE_IMAGE" "$LIBERO_REPO"
    libero_list_images "${images[@]}" "$LIBERO_VERIFIER_BASE_IMAGE"
}

# The per-subtask verifiers derive FROM the verifier base, so `base` comes first.
cmd_verifiers() {
    command -v docker >/dev/null || libero_die "docker is required"
    docker image inspect "$LIBERO_VERIFIER_BASE_IMAGE" >/dev/null 2>&1 \
        || libero_die "$LIBERO_VERIFIER_BASE_IMAGE missing; run sim/libero/libero.sh base (make sim-libero) first"
    docker image inspect rlebench-task05-robotwin-verifier-base:dev >/dev/null 2>&1 \
        || libero_die "rlebench-task05-robotwin-verifier-base:dev missing; run sim/robotwin/robotwin.sh images (make sim-robotwin) first"
    local subtask verifier_image images=()
    for subtask in "${LIBERO_SUBTASKS[@]}"; do
        verifier_image="rlebench-task05-${subtask}-verifier:dev"
        libero_step "verifier image $verifier_image"
        docker build -t "$verifier_image" "$LIBERO_REPO/tasks/task05/$subtask/tests"
        images+=("$verifier_image")
    done
    libero_list_images "${images[@]}"
}

cmd_images() {
    cmd_bundles
    cmd_base
    cmd_verifiers
}

libero_list_images() {
    local image
    for image in "$@"; do
        docker images "$image" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}"
    done
}

cmd_check() {
    local subtask="${1:-}"
    local known=" ${LIBERO_SUBTASKS[*]} "
    [[ "$known" == *" $subtask "* ]] || libero_die "choose one subtask: ${LIBERO_SUBTASKS[*]}"
    python3 "$LIBERO_REPO/tasks/task05/harness/validate_assets.py" "$subtask"
}

# By tag, never by id: an id may carry other tags, and -f on a tag only untags when
# a family image still derives from it (taskNN-clean removes those).
libero_remove_image() {
    command -v docker >/dev/null || { echo "  docker absent, skipping $1"; return; }
    docker image inspect "$1" >/dev/null 2>&1 || { echo "  $1 absent"; return; }
    docker rmi -f "$1"
}

# Nothing lands on disk for this layer; the per-subtask verifiers are `make task05-clean`.
cmd_clean() {
    libero_step "removing the libero images"
    local name
    for name in "${LIBERO_BUNDLES[@]}"; do
        libero_remove_image "$LIBERO_AGENT_IMAGE_PREFIX-$name:dev"
    done
    libero_remove_image "$LIBERO_VERIFIER_BASE_IMAGE"
    for name in "${LIBERO_BUNDLES[@]}"; do
        libero_remove_image "$LIBERO_HF_IMAGE_PREFIX-$name:dev"
    done
}

cmd_env() {
    printf '%s\n' \
        "LIBERO_AGENT_IMAGE_PREFIX=$LIBERO_AGENT_IMAGE_PREFIX" \
        "LIBERO_HF_IMAGE_PREFIX=$LIBERO_HF_IMAGE_PREFIX" \
        "LIBERO_VERIFIER_BASE_IMAGE=$LIBERO_VERIFIER_BASE_IMAGE" \
        "LIBERO_SHA=$LIBERO_SHA" \
        "LIBERO_PLUS_SHA=$LIBERO_PLUS_SHA"
}

case "${1:-}" in
    bundles)   shift; cmd_bundles "$@" ;;
    base)      shift; cmd_base "$@" ;;
    verifiers) shift; cmd_verifiers "$@" ;;
    images)    shift; cmd_images "$@" ;;
    check)     shift; cmd_check "$@" ;;
    env)       shift; cmd_env "$@" ;;
    clean)     shift; cmd_clean "$@" ;;
    help|-h|--help|"") libero_usage ;;
    *) libero_usage; exit 1 ;;
esac
