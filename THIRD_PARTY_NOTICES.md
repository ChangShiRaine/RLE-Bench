# Third-party notices

RLE-Bench is licensed under the MIT License (see `LICENSE`). It vendors or
downloads the following third-party material; each keeps its own license.

## Vendored in this repository

| material | where | license |
|---|---|---|
| Franka Emika Panda model (MuJoCo Menagerie) | `assets/robots/franka_emika_panda/` | Apache-2.0 (`LICENSE` in that directory) |
| Universal Robots UR5e model (MuJoCo Menagerie) | `assets/robots/universal_robots_ur5e/` ([provenance](assets/robots/universal_robots_ur5e/VENDORED.txt)) | BSD-3-Clause, ROS Industrial Consortium (`LICENSE` there) |
| UFACTORY xArm7 model (MuJoCo Menagerie) | `assets/robots/ufactory_xarm7/` ([provenance](assets/robots/ufactory_xarm7/VENDORED.txt), modified) | BSD-3-Clause, UFACTORY Inc. (`LICENSE` there) |
| GELLO mechanical parts | `assets/robots/gello_mechanical/` ([provenance](assets/robots/gello_mechanical/VENDORED.txt)) | MIT, Philipp Wu ([LICENSE](assets/robots/gello_mechanical/LICENSE)) |
| MuJoCo cube XML and textures | `tasks/task03/tabletop/pocket/assets/` ([provenance](tasks/task03/tabletop/pocket/assets/VENDORED.md)) | Apache-2.0 ([LICENSE](tasks/task03/tabletop/pocket/assets/LICENSE)) |

Staged copies retain their source licenses. Task09's GELLO mesh directories
include `LICENSE`; its original procedural servo visuals use the repository's
MIT license, copied as `SERVO_LICENSE.txt`. The task03 cube payload retains
both `LICENSE` and `VENDORED.md` alongside its assets.

## Fetched at build time (never committed)

| material | fetched by | license |
|---|---|---|
| robosuite + RoboCasa (sources and asset dataset, pinned SHAs) | `sim/robocasa/robocasa.sh` into `third_party/` | MIT (robosuite), RoboCasa license |
| SAM3 (code + gated weights) | `sim/perception/perception.sh` into `third_party/perception/` | Meta AI license; the weight repository is gated on Hugging Face and requires accepting its terms |
| ContactGraspNet (PyTorch port + checkpoint) | `sim/perception/perception.sh` | see upstream repository license |
| motiontrack vendored sources (task04, pinned) | `sim/motiontrack/motiontrack.sh` into `third_party/motiontrack/` | upstream licenses at the pinned revisions |
| Python dependencies | `requirements*.txt`, `sim/*/requirements.txt`, image Dockerfiles | respective package licenses |

Nothing in this repository grants rights to the gated SAM3 weights; obtain
access from the upstream repository under its own terms.
