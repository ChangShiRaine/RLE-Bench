"""Restricted rescue API; no private puzzle state or counters returned to agents."""
import json
from pathlib import Path
from . import front_daemon
from .recovery import RecoveryPocketCube

def install():
    from harness.service import Service
    front_daemon.FrontPocketCube = RecoveryPocketCube
    front_daemon.install()
    dispatch = Service._dispatch

    def rescue_dispatch(self,op,msg):
        env = self._session.current_env
        if op == 'recover_drop':
            if set(msg) != {'op'}:
                raise ValueError('recover_drop takes no arguments.')
            if not self._session.evaluating or not self._session._live():
                raise ValueError('No active episode.')
            if self._session.status()['steps_remaining'] < 1:
                raise ValueError('Step budget exhausted.')
            if self._clock().get('seconds_remaining',1) <= 0:
                raise ValueError('Time budget exhausted.')
            env.recover_drop()
            action = [0.] * 14
            action[6] = action[13] = -1.
            reply = dispatch(self,'step',{'op':'step','actions':[action]})
            reply['recovered'] = True
        else:
            reply = dispatch(self,op,msg)
        env = self._session.current_env or env
        if isinstance(env,RecoveryPocketCube):
            counters = env.private_counters()
            if counters != getattr(self,'_private_rescue_counters',None):
                Path('/var/lib/rlebench/recovery.json').write_text(json.dumps(counters)+'\n')
                self._private_rescue_counters = counters
        return reply

    Service._dispatch = rescue_dispatch

def main():
    install()
    from harness.daemon_main import main as shared_main
    return shared_main()

if __name__ == '__main__':
    raise SystemExit(main())
