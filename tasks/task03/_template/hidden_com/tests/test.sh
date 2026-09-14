#!/bin/sh
set -eu
mkdir -p /logs/verifier
cd /
if ! /usr/local/bin/python -I -c 'import sys; sys.path.insert(0,"/opt/private"); from harness.tabletop.hidden_com.verify import main; main()' > /logs/verifier/reward.json; then
    printf '{"reward":0,"correct":0,"submitted":0,"control_steps":0}\n' > /logs/verifier/reward.json
fi
/usr/local/bin/python -I -c 'import sys; sys.path.insert(0,"/opt/private"); from harness.tabletop.hidden_com.video import export; export("/logs/verifier/media")' || true
chmod -R a+rX /logs/verifier
chmod a+rwx /logs/verifier
