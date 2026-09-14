# The perception layer

The two models the task01 **L2 and L3** harness serves: SAM3 for text-prompted
segmentation, Contact-GraspNet for grasp proposals. Ported from cap-x's perception
servers, at the commits cap-x pins (`pins.env`).

**L1 does not have this**, and that is the point — L1 is the control condition.

```
sim/perception/perception.sh setup     # fetch whatever is missing; a no-op once complete
sim/perception/perception.sh fetch     # clone both repos + download the gated SAM3 weights
sim/perception/perception.sh verify    # is third_party/perception complete?
```

`make sim-perception` runs `setup`, and
**`make task01` runs `setup` itself** before building L2 or L3, so fetching by hand is
optional — do it ahead of time, or let the build do it.

**There is no image here.** The models are a *stage* inside task01's own image
(`models-on`, selected by the `MODELS` build arg), fed from the tree below as a **named
build context** — which is how one Dockerfile pulls in several GB that live outside its
own context, without a separate image to build and keep in sync. `make task01` does it.
L1 selects `models-off`, which BuildKit then never builds.

`requirements.lock` fixes the Python 3.11/Linux x86_64 runtime and build
dependencies for both models, including transitives. Torch's version is expanded
from `pins.env`. The task01 generator stages the lock into L2/L3 build contexts;
installs use `--no-deps --no-build-isolation` to prevent dependency resolution
from changing it. PyOpenGL stays at 3.1.10 for MuJoCo EGL, overriding pyrender's
obsolete 3.1.0 requirement. When updating pins, rebuild one perception level and
check both models offline, then bump task01's version.

## The weights are gated

`facebook/sam3` requires an accepted access request on HuggingFace, so fetching needs a
token from an account that has one. It is read from the first of these that is set:

```bash
HF_TOKEN=hf_... make task01        # or sim/perception/perception.sh setup
HUGGING_FACE_HUB_TOKEN=hf_...
huggingface-cli login              # cached under $HF_HOME/token
```

With none of them, the build **stops before it builds anything** and says so. L1 is
unaffected: `TASK01_LEVELS=L1 make task01` needs no token.

Fetching is the **only** step that needs the token or the network. The cache is copied into
the image and `HF_HUB_OFFLINE=1` is set, so a run never reaches out — which it could not
anyway (`network_mode = "no-network"` for the verifier, allowlist for the agent).

Contact-GraspNet's checkpoint ships inside its own repo, so it needs neither.

## Where it lands

```
third_party/perception/         GITIGNORED, ~6 GB, created by `fetch`
  sam3/                         the repo at SAM3_SHA
  contact_graspnet_pytorch/     the repo at CGN_SHA, checkpoint included
  hf-cache/                     the HuggingFace cache holding facebook/sam3
```

The L2/L3 image copies all three to `/opt/perception`, **root:root 0700**. The agent reaches
the models through `/run/rlebench/perception.sock` and never as importable modules —
same split as the simulator, and for the same reason: a service with a contract rather
than a tree to monkeypatch.

## What it is not

It holds no environment, imports nothing from the metering daemon, and answers only from
the arrays in the request. It cannot advance an episode, so **nothing it does is
metered**. The agent pays for it in wall clock.
