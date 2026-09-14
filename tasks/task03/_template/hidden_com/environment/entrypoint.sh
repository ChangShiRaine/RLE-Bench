#!/bin/sh
set -eu
umask 077
mkdir -p /run/rlebench /var/lib/rlebench
chmod 755 /run/rlebench
chmod 700 /var/lib/rlebench
cd /
/usr/local/bin/python -I -c 'import sys; sys.path.insert(0,"/opt/private"); from harness.tabletop.hidden_com.daemon import main; main()' > /var/lib/rlebench/daemon.log 2>&1 &
daemon_pid=$!
i=0
while [ ! -S /run/rlebench/hidden-com.sock ]; do
    kill -0 "$daemon_pid" 2>/dev/null || exit 1
    i=$((i + 1))
    [ "$i" -lt 1800 ] || exit 1
    sleep 0.1
done
umask 022
cd /workspace
exec "$@"
