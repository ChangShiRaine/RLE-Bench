#!/usr/bin/env bash
# ONE entry point for the PERCEPTION layer: the two models the task01 L2/L3 harness
# serves. Same shape as sim/robocasa/robocasa.sh, which is the layer next door.
#
#     sim/perception/perception.sh setup     fetch if third_party/perception is incomplete
#     sim/perception/perception.sh fetch     clone both repos + download the gated weights
#     sim/perception/perception.sh verify    is third_party/perception complete?
#     sim/perception/perception.sh env       the resolved paths
#     sim/perception/perception.sh clean     remove third_party/perception
#
# There is no image here: the models are a STAGE inside task01's own image, fed from the
# tree below as a named build context. `make task01` is what builds them in.
#
# Everything lands INSIDE the repo, under third_party/perception (gitignored), so a
# checkout is self-describing and no container mount has to be told where things went.
#
# L1 NEEDS NONE OF THIS. It is the control condition and derives from the plain simulator
# image, so `make task01-L1` works on a machine that has never run this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SIM="$HERE"

# shellcheck source=pins.env
source "$SIM/pins.env"
VENDOR="${RLEBENCH_PERCEPTION_VENDOR:-$REPO/third_party/perception}"

step() { echo; echo "=== $* ==="; }
die()  { echo "$*" >&2; exit 1; }

usage() {
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Clone at a pin, idempotently. A branch name is not a pin: both upstreams move.
clone_at() {
    local url="$1" sha="$2" dest="$3"
    if [ -d "$dest/.git" ]; then
        if [ "$(git -C "$dest" rev-parse HEAD)" = "$sha" ]; then
            echo "  $(basename "$dest") already at ${sha:0:8}"
            return
        fi
        git -C "$dest" fetch --quiet origin "$sha"
    else
        mkdir -p "$(dirname "$dest")"
        git clone --quiet --filter=blob:none "$url" "$dest"
    fi
    git -C "$dest" checkout --quiet "$sha"
    echo "  $(basename "$dest") at ${sha:0:8}"
}

# Where a Hugging Face token may come from, in order. The login file is last and is why
# a machine that has run `huggingface-cli login` needs nothing extra -- it has to be read
# HERE, before the download overrides HF_HOME to point at the vendor cache.
hf_token() {
    local f="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
    if   [ -n "${HF_TOKEN:-}" ];               then printf '%s' "$HF_TOKEN"
    elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then printf '%s' "$HUGGING_FACE_HUB_TOKEN"
    elif [ -s "$f" ];                          then tr -d '[:space:]' < "$f"
    fi
}

# What `make task01` calls for L2/L3: fetch only what is not already there. Cheap and
# offline once the tree is complete, so it can sit in front of every build.
cmd_setup() {
    if vendor_complete; then
        echo "perception models complete at $VENDOR"
        return
    fi
    cmd_fetch
}

cmd_fetch() {
    # Read the token BEFORE anything slow, so a machine without one stops in a second
    # rather than after two clones.
    local token; token="$(hf_token || true)"   # an unreadable file is "no token", not a crash
    [ -n "$token" ] || die \
"NO HUGGING FACE TOKEN, and $SAM3_HF_REPO is gated, so the L2/L3 models cannot be
fetched. Supply one from an account with an accepted access request, by any of:

    HF_TOKEN=hf_... make task01        (or sim/perception/perception.sh setup)
    HUGGING_FACE_HUB_TOKEN=hf_...
    huggingface-cli login              (cached under \$HF_HOME/token)

Request access at https://huggingface.co/$SAM3_HF_REPO. This is the only step that needs
the token or the network -- the cache is baked into the image afterwards.

L1 needs none of this: TASK01_LEVELS=L1 make task01 builds the control condition."

    step "sources -> $VENDOR"
    clone_at "$SAM3_REPO" "$SAM3_SHA" "$VENDOR/sam3"
    clone_at "$CGN_REPO" "$CGN_SHA" "$VENDOR/contact_graspnet_pytorch"

    step "weights: $SAM3_HF_REPO"
    # `hf` since huggingface_hub 1.x; `huggingface-cli` is the pre-1.0 name and now
    # refuses to run, so prefer the new one and keep the old as the fallback.
    local cli=""
    for c in hf huggingface-cli; do
        if command -v "$c" >/dev/null 2>&1; then cli="$c"; break; fi
    done
    [ -n "$cli" ] || die "no Hugging Face CLI found: pip install 'huggingface_hub[cli]'"

    # HF_HOME points at the vendor cache so the download lands in the tree the image
    # takes as a build context, not in the caller's own cache.
    HF_HOME="$VENDOR/hf-cache" "$cli" download "$SAM3_HF_REPO" --token "$token" >/dev/null
    echo "  cached under $VENDOR/hf-cache"

    cmd_verify
}

# Is the vendor tree complete and at the pins? Returns the answer; says why only when
# asked to report, so `setup` can ask silently.
vendor_complete() {
    local report="${1:-}" bad=0
    for path in \
        "sam3/sam3/model_builder.py" \
        "contact_graspnet_pytorch/contact_graspnet_pytorch/contact_grasp_estimator.py" \
        "contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt"
    do
        if [ -e "$VENDOR/$path" ]; then
            if [ "$report" = report ]; then echo "  ok   $path"; fi
        else
            if [ "$report" = report ]; then echo "  MISSING $path" >&2; fi
            bad=1
        fi
    done
    # The weights, by the files the model builder actually loads -- not the cache
    # directory. The repo is gated, so an unauthenticated fetch still leaves a cache
    # skeleton (LICENSE, README) that a bare existence check mistakes for complete,
    # and the image then bakes an empty cache and fails at model load, offline.
    local snap="$VENDOR/hf-cache/hub/models--${SAM3_HF_REPO//\//--}/snapshots"
    for w in config.json sam3.pt; do
        if compgen -G "$snap/*/$w" >/dev/null; then
            if [ "$report" = report ]; then echo "  ok   hf-cache $w"; fi
        else
            if [ "$report" = report ]; then echo "  MISSING hf-cache $w" >&2; fi
            bad=1
        fi
    done
    # A pin that has drifted is worse than one that is absent: the image would build and
    # score a run against a model nobody chose.
    for pair in "sam3:$SAM3_SHA" "contact_graspnet_pytorch:$CGN_SHA"; do
        local dir="${pair%%:*}" want="${pair#*:}"
        [ -d "$VENDOR/$dir/.git" ] || continue
        local got; got="$(git -C "$VENDOR/$dir" rev-parse HEAD)"
        if [ "$got" != "$want" ]; then
            if [ "$report" = report ]; then
                echo "  DRIFT $dir at ${got:0:8}, pins say ${want:0:8}" >&2
            fi
            bad=1
        fi
    done
    return "$bad"
}

cmd_verify() {
    step "verify $VENDOR"
    vendor_complete report || die "incomplete: run sim/perception/perception.sh setup"
    echo "  complete"
}

# rm -rf would empty a writable bind mount of a shared tree; refuse any mount at or under it.
refuse_if_mounted() {
    local d; d="$(readlink -f "$1" 2>/dev/null || true)"
    [ -n "$d" ] && [ -e "$d" ] || return 0
    awk -v p="$d" '$2 == p || index($2, p "/") == 1 { f = 1 } END { exit !f }' /proc/mounts \
        && die "$1 has a filesystem mounted at or under it; unmount it before clean"
    return 0
}

# No image to remove: the models live inside task01's images (make task01-clean).
cmd_clean() {
    refuse_if_mounted "$VENDOR"
    step "removing the perception models under $VENDOR"
    rm -rf "$VENDOR/sam3" "$VENDOR/contact_graspnet_pytorch" "$VENDOR/hf-cache"
    rmdir "$VENDOR" 2>/dev/null || true
}

cmd_env() {
    case "${1:-all}" in
        vendor) echo "$VENDOR" ;;
        all)
            echo "RLEBENCH_PERCEPTION_VENDOR=$VENDOR"
            echo "SAM3_SHA=$SAM3_SHA"
            echo "CGN_SHA=$CGN_SHA"
            echo "TORCH_VERSION=$TORCH_VERSION"
            ;;
        *) die "env: unknown key '$1' (vendor|all)" ;;
    esac
}

case "${1:-}" in
    setup)  shift; cmd_setup "$@" ;;
    fetch)  shift; cmd_fetch "$@" ;;
    verify) shift; cmd_verify "$@" ;;
    env)    shift; cmd_env "$@" ;;
    clean)  shift; cmd_clean "$@" ;;
    ""|-h|--help|help) usage ;;
    *) die "unknown command '$1'; try --help" ;;
esac
