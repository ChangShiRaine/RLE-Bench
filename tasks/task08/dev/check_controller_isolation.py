"""Run as root inside the Task08 verifier image with PYTHONPATH=/tests."""
from pathlib import Path
import os
import shutil
import tempfile

import mujoco

from harness import config
from harness.base_design.controller_process import ControllerFault, ControllerProcess


def main():
    Path('/logs/verifier').mkdir(parents=True, exist_ok=True)
    for directory in ['/tests', '/logs/verifier']:
        os.chmod(directory, 0o700)
    secret = Path('/tests/controller-isolation-secret')
    secret.write_text('verifier-only')
    work = Path(tempfile.mkdtemp())
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="j"/><geom size=".1"/></body></worldbody>
      <actuator><motor name="motor" joint="j"/></actuator></mujoco>''')
    context = {'actuator_names': ['motor'], 'seed': 0}
    policy = work / 'controller.py'
    policy.write_text('''import os
import mujoco
class Policy:
 def reset(self, context, seed):
  assert os.geteuid() == 65534
  assert 'PYTHONPATH' not in os.environ
  for path in ['/tests/controller-isolation-secret', '/tests/harness/config.py',
               '/tests/harness/thresholds.py', '/tests/models/assets/franka_emika_panda/panda_nohand.xml',
               '/logs/verifier/reward.json', '/proc/%s/mem']:
   for mode in ['r', 'a']:
    try:
     with open(path, mode) as f:
      if mode == 'r': f.read(1)
      else: f.write('forged')
    except PermissionError: pass
    else: raise AssertionError('private file accessible: ' + path + ' mode=' + mode)
  try:
   import harness.base_design.scorer
  except (ModuleNotFoundError, PermissionError): pass
  else: raise AssertionError('harness accessible')
  try: os.fork()
  except OSError: pass
  else: raise AssertionError('fork allowed')
  self.model = mujoco.MjModel.from_binary_path(context['model_file'])
  self.model.body_mass[:] = 123
 def act(self, observation):
  print('controller diagnostics do not corrupt replies')
  return [0.25]
def make_controller(): return Policy()
''' % os.getpid())
    try:
        mass = model.body_mass.copy()
        with ControllerProcess(str(policy), model, context) as process:
            assert process.act({}).tolist() == [0.25]
            assert (model.body_mass == mass).all()
        assert secret.read_text() == 'verifier-only'
        config.PICK_CONTROLLER_REPLY_SECONDS = 0.3
        for body in ('return [float("nan")]', 'return [1, 2]', 'raise RuntimeError("bad")', 'while True: pass'):
            policy.write_text('class Policy:\n def reset(self, context, seed): pass\n def act(self, observation):\n  '+body+'\ndef make_controller(): return Policy()\n')
            with ControllerProcess(str(policy), model, context) as process:
                try:
                    process.act({})
                except ControllerFault:
                    pass
                else:
                    raise AssertionError(f'fault accepted: {body}')
        print('PASS: private files, parent memory, imports, subprocesses, local model mutation, invalid actions, and hang isolation')
    finally:
        secret.unlink()
        shutil.rmtree(work)


if __name__ == '__main__':
    main()
