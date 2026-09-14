#!/usr/bin/env bash
# ONE entry point for everything "RoboCasa": the simulator image, the asset
# dataset, and the host-side development environment.
#
# The Makefile is a thin facade over this file (`make sim-robocasa` runs `image`
# then `setup`; everything finer is a subcommand here). Logic lives here, not there:
# these are shell operations -- clones, downloads, mounts, docker builds -- and
# splitting them across make recipes and a second script is what made this hard
# to follow. Everything the layer knows how to do is in this one file.
#
# USER MODE -- run a task. Needs the simulator image and the dataset. No venv,
# no vendored sources, no Python beyond a stdlib python3 for the asset check.
#
#     sim/robocasa/robocasa.sh image                 the shared simulator image
#     sim/robocasa/robocasa.sh assets mount          a packed dataset, read-only
#     sim/robocasa/robocasa.sh assets verify         is a tree complete and current?
#     sim/robocasa/robocasa.sh env                   the resolved paths
#     sim/robocasa/robocasa.sh clean                 remove the image, sources, dataset, venv
#
# DEV MODE -- change the harness or run the host-side suites. Adds the vendored
# sources, the dataset and a venv importing them, all inside the repo.
#
#     sim/robocasa/robocasa.sh setup                 sources + venv + assets
#     sim/robocasa/robocasa.sh setup --no-assets     skip the ~15 GB download
#     sim/robocasa/robocasa.sh setup --force-assets  re-download even if current
#     sim/robocasa/robocasa.sh assets download       just the dataset
#     sim/robocasa/robocasa.sh assets pack           a populated tree -> .sqfs
#
# Everything lands INSIDE the repo: sources and dataset under third_party/
# (gitignored), venv at .venv-robocasa. A checkout is then self-describing, so
# no target, script or container mount has to be told where the simulator went.
#
# RoboCasa's pins (mujoco 3.3.1) conflict with the repo-wide requirements.txt
# (3.5.0), which is why the dev environment is a separate venv, not `make install`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SIM="$HERE"

# The pins are the single source of truth, shared with base/Dockerfile and the
# Makefile. Sourced, never restated: this script cannot fork from them.
# shellcheck source=pins.env
source "$SIM/pins.env"

# --- where things live ---------------------------------------------------------
# NO ABSOLUTE DEFAULTS. Every path is either derived from the repo's own location
# or must be set explicitly. A checked-in path from one developer's box resolves
# to nothing on anyone else's machine, and the failure surfaces deep inside a
# container as a missing model.xml rather than as "you have not configured this".

VENDOR="${ROBOCASA_VENDOR:-$REPO/third_party}"
VENV="$REPO/.venv-robocasa"
PY="$VENV/bin/python"

# The dataset the checkout should USE. RoboCasa resolves assets by package path,
# not by an environment variable, so the only way to point a checkout at a dataset
# elsewhere (a SquashFS mount, a shared copy) is to make that path BE the dataset
# -- see `link_asset_tree`. Defaults to the checkout's own tree, which is where
# `assets download` puts it.
ASSETS_IN_TREE="$VENDOR/robocasa/robocasa/models/assets"
ASSET_TREE="${ROBOCASA_ASSET_DIR:-$ASSETS_IN_TREE}"
MARKER="ROBOCASA_ASSET_VERSION"

# Where the packed .sqfs is kept. REQUIRED by pack/mount and nothing else -- it is
# site policy (a scratch filesystem, a shared cache), so there is no defensible
# default. Where a node mounts it is a convention, and it is the directory this
# script CREATES rather than one that must already exist.
ASSET_HOME="${ROBOCASA_ASSET_HOME:-}"
ASSET_SQFS="${ROBOCASA_ASSET_SQFS:-$ASSET_HOME/robocasa-assets-$ROBOCASA_ASSET_VERSION.sqfs}"
ASSET_MOUNT="${ROBOCASA_ASSET_MOUNT:-/mnt/robocasa-assets/$ROBOCASA_ASSET_VERSION}"

step() { echo; echo "=== $* ==="; }
die()  { echo "$*" >&2; exit 1; }

usage() {
    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
}

# verify_assets.py is pure stdlib, so USER MODE can run it with a system python3.
# Prefer the venv when there is one, so dev mode checks the interpreter it uses.
any_python() {
    if [ -x "$PY" ]; then echo "$PY"
    elif command -v python3 >/dev/null; then command -v python3
    else die "no python3 and no $PY -- cannot run the asset check"
    fi
}

require_asset_home() {
    [ -n "$ASSET_HOME" ] || die "\
ROBOCASA_ASSET_HOME is not set.
  It is where the packed robocasa-assets-*.sqfs is kept, and it is
  site-specific, so this repo ships no default.
    e.g.  ROBOCASA_ASSET_HOME=/path/to/assets sim/robocasa/robocasa.sh assets pack
  See sim/robocasa/README.md -> The asset dataset."
}

# A partial tree fails deep inside a container as a missing model.xml. The version
# marker is the cheap proof, and it is what verify_assets.py checks at container
# start too.
require_assets() {
    [ -f "$ASSET_TREE/$MARKER" ] || die "\
$ASSET_TREE is not a populated RoboCasa asset tree (no $MARKER).
  Run \`sim/robocasa/robocasa.sh assets download\` to fetch it, \`assets mount\` to
  mount an existing .sqfs, or set ROBOCASA_ASSET_DIR to a complete
  models/assets tree."
}

# ==============================================================================
# USER MODE
# ==============================================================================

# The shared simulator image. Pins come from pins.env as --build-arg; the
# Dockerfile declares those ARGs with NO defaults, so a bare `docker build` fails
# loudly rather than silently forking the image's pins from this repo's.
cmd_image() {
    step "simulator image $ROBOCASA_SIM_IMAGE"
    docker build -t "$ROBOCASA_SIM_IMAGE" \
        --build-arg ROBOSUITE_SHA="$ROBOSUITE_SHA" \
        --build-arg ROBOCASA_SHA="$ROBOCASA_SHA" \
        --build-arg ROBOCASA_ASSET_VERSION="$ROBOCASA_ASSET_VERSION" \
        "$SIM/base"
}

# The resolved paths, for humans and for `export ROBOCASA_ASSET_DIR=$(... env asset-dir)`.
cmd_env() {
    case "${1:-all}" in
        asset-dir) echo "$ASSET_TREE" ;;
        vendor)    echo "$VENDOR" ;;
        venv)      echo "$VENV" ;;
        image)     echo "$ROBOCASA_SIM_IMAGE" ;;
        all)
            echo "ROBOCASA_VENDOR=$VENDOR"
            echo "ROBOCASA_ASSET_DIR=$ASSET_TREE"
            echo "ROBOCASA_SIM_IMAGE=$ROBOCASA_SIM_IMAGE"
            echo "ROBOCASA_ASSET_VERSION=$ROBOCASA_ASSET_VERSION"
            echo "VENV=$VENV"
            ;;
        *) die "env: unknown key '$1' (asset-dir|vendor|venv|image|all)" ;;
    esac
}

# ==============================================================================
# THE ASSET DATASET
#
# 123,434 files / ~15 GB that RoboCasa only ever READS, so it ships as a
# separately versioned read-only dataset rather than image content. Baking it in
# cost a 30+ minute 37 GB layer export on every rebuild and duplicated nothing.
# ==============================================================================

# ~15 GB in six Box zips. `--type all` is not optional: the fixture registry YAMLs
# reference generative_textures paths directly, so a partial download fails at
# reset() rather than at download time.
assets_download() {
    local force="${1:-0}" have=""
    [ -f "$ASSETS_IN_TREE/$MARKER" ] && have="$(cat "$ASSETS_IN_TREE/$MARKER")"

    step "asset dataset $ROBOCASA_ASSET_VERSION"
    [ -x "$PY" ] || die "the downloader runs out of $VENV -- run \`sim/robocasa/robocasa.sh setup\` first"

    if [ "$force" = 0 ] && [ "$have" = "$ROBOCASA_ASSET_VERSION" ]; then
        echo "  already at $ROBOCASA_ASSET_VERSION -- skipping (--force to re-download)"
        return 0
    fi
    echo "  downloading into $ASSETS_IN_TREE (~15 GB, six zips)"
    # The downloader asks "Proceed? (y/n)" once. `printf` rather than `yes`: `yes`
    # is killed by SIGPIPE when python stops reading, and under `set -o pipefail`
    # that makes a SUCCESSFUL run report failure.
    printf 'y\n' | "$PY" -m robocasa.scripts.download_kitchen_assets --type all
    # NOTHING upstream writes this marker, and verify_assets.py treats a missing one
    # as fatal -- version skew is the failure mode with no natural symptom, so the
    # dataset has to say what it is. Written LAST, so its presence means "complete".
    printf '%s\n' "$ROBOCASA_ASSET_VERSION" > "$ASSETS_IN_TREE/$MARKER"
    echo "  wrote $MARKER=$ROBOCASA_ASSET_VERSION"
}

# Duplicate detection is on by default and matters here (RoboCasa repeats textures
# across objects: 4,357 duplicates, 15 GB -> 9.6 GB).
#
# The source must be the COMPLETE MERGED tree -- repo-tracked files plus downloads.
# 210 files under models/assets ship with the robocasa repo, and a bind mount
# OBSCURES whatever the image had at the mount point. Mounting individual heavy
# subdirectories is NOT safe either: fixtures/ mixes repo-tracked registry YAMLs
# with downloaded content.
assets_pack() {
    require_asset_home; require_assets
    step "packing $ASSET_TREE -> $ASSET_SQFS"
    mksquashfs "$ASSET_TREE" "$ASSET_SQFS" \
        -comp zstd -Xcompression-level 3 -b 256K -all-root -no-xattrs -noappend
    ( cd "$ASSET_HOME" && sha256sum "$(basename "$ASSET_SQFS")" > "$(basename "$ASSET_SQFS").sha256" )
}

# Checksum FIRST, on purpose: a silently wrong asset version changes the scenes
# with no error anywhere. Needs root, so this provisions a node, not a trial.
assets_mount() {
    require_asset_home
    step "mounting $ASSET_SQFS -> $ASSET_MOUNT"
    ( cd "$ASSET_HOME" && sha256sum -c "$(basename "$ASSET_SQFS").sha256" )
    sudo mkdir -p "$ASSET_MOUNT"
    sudo mount -t squashfs -o loop,ro,nodev,nosuid,noexec "$ASSET_SQFS" "$ASSET_MOUNT"
    echo "mounted. run with ROBOCASA_ASSET_DIR=$ASSET_MOUNT"
}

assets_umount() { sudo umount "$ASSET_MOUNT"; }

# The same guard every container runs at start: present, complete, current.
assets_verify() {
    ROBOCASA_ASSETS="$ASSET_TREE" ROBOCASA_ASSET_VERSION="$ROBOCASA_ASSET_VERSION" \
        "$(any_python)" "$SIM/base/verify_assets.py"
}

cmd_assets() {
    local sub="${1:-download}"; shift || true
    local force=0
    for arg in "$@"; do
        case "$arg" in
            --force|--force-assets) force=1 ;;
            *) die "assets $sub: unknown argument '$arg'" ;;
        esac
    done
    case "$sub" in
        download) assets_download "$force"; assets_verify ;;
        pack)     assets_pack ;;
        mount)    assets_mount ;;
        umount)   assets_umount ;;
        verify)   assets_verify ;;
        *) die "assets: unknown subcommand '$sub' (download|pack|mount|umount|verify)" ;;
    esac
}

# ==============================================================================
# DEV MODE
# ==============================================================================

setup_sources() {
    step "vendored sources at the pinned commits -> $VENDOR"
    mkdir -p "$VENDOR"
    local repo sha url
    for repo in robosuite robocasa; do
        case "$repo" in
            robosuite) sha=$ROBOSUITE_SHA; url=https://github.com/ARISE-Initiative/robosuite.git ;;
            robocasa)  sha=$ROBOCASA_SHA;  url=https://github.com/robocasa/robocasa.git ;;
        esac
        [ -d "$VENDOR/$repo/.git" ] || git clone "$url" "$VENDOR/$repo"
        # A shallow clone may not carry the pinned commit yet.
        git -C "$VENDOR/$repo" fetch --depth 1 origin "$sha" 2>/dev/null || \
            git -C "$VENDOR/$repo" fetch --unshallow 2>/dev/null || true
        git -C "$VENDOR/$repo" checkout --quiet "$sha"
        echo "  $repo @ $(git -C "$VENDOR/$repo" rev-parse --short HEAD)"
    done
}

setup_venv() {
    step "venv (uv; system python3 lacks ensurepip, and RoboCasa wants 3.11)"
    # Reused rather than recreated: `uv venv` refuses an existing one outright, and
    # this has to be re-runnable -- moving the vendored tree is exactly when you run
    # it again. Delete .venv-robocasa to force a clean rebuild.
    if [ -x "$PY" ]; then
        echo "  reusing $VENV ($("$PY" --version))"
    else
        uv venv --python 3.11 "$VENV"
    fi

    step "robosuite + robocasa (editable, from source)"
    # Editable installs record the source path, so this must re-run after the
    # vendored tree moves. Cheap when the deps are already satisfied.
    VIRTUAL_ENV="$VENV" uv pip install -e "$VENDOR/robosuite" 2>&1 | tail -3
    VIRTUAL_ENV="$VENV" uv pip install -e "$VENDOR/robocasa" 2>&1 | tail -3

    # Both setup scripts PROMPT before overwriting an existing macros_private.py, so
    # a re-run needs an answer on stdin or dies with EOFError. Overwriting with
    # defaults is the intent -- the point here is a reproducible environment.
    printf 'y\n' | "$PY" -m robocasa.scripts.setup_macros  >/dev/null
    printf 'y\n' | "$PY" -m robosuite.scripts.setup_macros >/dev/null
    echo "  macros_private.py written for both packages"
}

# Only relevant when the dataset lives OUTSIDE the checkout (a SquashFS mount, a
# shared copy); otherwise the checkout's own tree already is the dataset.
link_asset_tree() {
    step "the checkout's asset path"
    local a b
    a="$(readlink -f "$ASSET_TREE"     2>/dev/null || echo "$ASSET_TREE")"
    b="$(readlink -f "$ASSETS_IN_TREE" 2>/dev/null || echo "$ASSETS_IN_TREE")"
    if [ "$a" = "$b" ]; then
        echo "  the checkout's own tree IS the dataset; nothing to link"
    elif [ ! -e "$ASSET_TREE/$MARKER" ]; then
        echo "  WARNING: $ASSET_TREE has no $MARKER, so it is not a populated dataset."
        echo "  Host-side simulator tests will fail. Mount the dataset and re-run with"
        echo "  ROBOCASA_ASSET_DIR set, or unset it to download into the checkout."
    elif [ -L "$ASSETS_IN_TREE" ] || [ ! -e "$ASSETS_IN_TREE/$MARKER" ]; then
        # A symlink (a re-run) or the bare 210-file repo skeleton. Never a populated
        # real tree: replacing one of those would destroy a downloaded dataset.
        rm -rf "$ASSETS_IN_TREE"
        ln -sfn "$ASSET_TREE" "$ASSETS_IN_TREE"
        echo "  linked $ASSETS_IN_TREE -> $ASSET_TREE"
    else
        echo "  LEAVING ALONE: $ASSETS_IN_TREE is already a populated asset tree"
    fi
}

cmd_setup() {
    local do_assets=1 force=0
    for arg in "$@"; do
        case "$arg" in
            --no-assets)              do_assets=0 ;;
            --force-assets|--force)   force=1 ;;
            *) die "setup: unknown argument '$arg' (--no-assets|--force-assets)" ;;
        esac
    done

    setup_sources
    setup_venv
    if [ "$do_assets" = 1 ] && [ "$ASSET_TREE" = "$ASSETS_IN_TREE" ]; then
        assets_download "$force"
    elif [ "$do_assets" = 1 ]; then
        echo; echo "  ROBOCASA_ASSET_DIR points elsewhere; linking instead of downloading"
    fi
    link_asset_tree

    # Prove it, rather than report success and let it fail later on the seed that
    # samples the gap.
    step "verification"
    [ -e "$ASSET_TREE/$MARKER" ] && assets_verify
    "$PY" - <<'EOF'
import importlib
for m in ("numpy", "scipy", "mujoco", "robosuite", "robocasa", "gymnasium"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:12s} IMPORT FAILED: {type(e).__name__}: {e}")
EOF
    step "DONE"
    echo "  sources  $VENDOR/{robosuite,robocasa}"
    echo "  assets   $ASSET_TREE"
    echo "  venv     $VENV"
}

# rm -rf would empty a writable bind mount of a shared tree; refuse any mount at or under it.
refuse_if_mounted() {
    local d; d="$(readlink -f "$1" 2>/dev/null || true)"
    [ -n "$d" ] && [ -e "$d" ] || return 0
    awk -v p="$d" '$2 == p || index($2, p "/") == 1 { f = 1 } END { exit !f }' /proc/mounts \
        && die "$1 has a filesystem mounted at or under it; unmount it before clean"
    return 0
}

# By tag, never by id: an id may carry other tags, and -f on a tag only untags when
# a family image still derives from it (taskNN-clean removes those).
remove_image() {
    command -v docker >/dev/null || { echo "  docker absent, skipping $1"; return; }
    docker image inspect "$1" >/dev/null 2>&1 || { echo "  $1 absent"; return; }
    docker rmi -f "$1"
}

# Everything the layer put in the checkout, plus its image. A dataset elsewhere
# (ROBOCASA_ASSET_DIR, a .sqfs mount) is never touched: the in-tree symlink to it
# is unlinked, not followed.
cmd_clean() {
    refuse_if_mounted "$VENDOR/robosuite"
    refuse_if_mounted "$VENDOR/robocasa"
    step "removing $VENDOR/{robosuite,robocasa} and $VENV"
    rm -rf "$VENDOR/robosuite" "$VENDOR/robocasa" "$VENV"
    step "removing $ROBOCASA_SIM_IMAGE"
    remove_image "$ROBOCASA_SIM_IMAGE"
}

# ==============================================================================

cmd="${1:-help}"; shift || true
case "$cmd" in
    image)          cmd_image "$@" ;;
    assets)         cmd_assets "$@" ;;
    setup)          cmd_setup "$@" ;;
    env)            cmd_env "$@" ;;
    clean)          cmd_clean "$@" ;;
    help|-h|--help) usage ;;
    *) usage >&2; die "
unknown command: $cmd" ;;
esac
