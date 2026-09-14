#!/bin/bash
# Harbor Oracle for task04. Runs inside the agent container: trains the
# reference PPO for ORACLE_MINUTES (60 for a scored run), staging the policy
# every 5 minutes in /logs/artifacts/policy.
set -eu

MINUTES="${ORACLE_MINUTES:-60}"

# The GPU comes from environment/docker-compose.yaml; fail at once if it did
# not arrive rather than minutes later inside torch.
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit('no CUDA device: the compose GPU reservation did not take effect')
print(f'training on {torch.cuda.get_device_name(0)}', flush=True)
"

mkdir -p /logs/artifacts/policy

python3 /solution/payload/train.py \
    --minutes "$MINUTES" \
    --num-envs 4096 \
    --seed 0 \
    --report-every 300 \
    --out /logs/artifacts/policy \
    --log /logs/artifacts/train_log.jsonl \
    --robot-dir "$MOTIONTRACK_ROBOT_DIR" \
    --motion "$MOTIONTRACK_MOTION"

echo "task04 reference solution staged."
