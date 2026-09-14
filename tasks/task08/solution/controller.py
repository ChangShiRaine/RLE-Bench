"""Oracle shelf policy, appended to the standalone IK helpers by the stager."""
import math
from types import SimpleNamespace


class ShelfController:
    def reset(self, context, seed):
        self.model = mujoco.MjModel.from_binary_path(context['model_file'])
        for i in range(self.model.ngeom):
            if (self.model.geom(i).name or '').startswith('pod_'):
                self.model.geom_margin[i] = 0.01
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = context['qpos']
        self.data.qvel[:] = context['qvel']
        self.initial = np.asarray(context['ctrl'])
        self.base_adr = self.model.joint('base_free').qposadr[0]
        self.start_x = self.data.qpos[self.base_adr]
        self.target = np.asarray(context['target'])
        joints = context['arm_joint_names']
        actuators = context['arm_actuator_names']
        self.arm_ids = [self.model.actuator(n).id for n in actuators]
        self.arm_dofs = [self.model.joint(n).dofadr[0] for n in joints]
        self.wheel_ids = [self.model.actuator(n).id for n in context['wheel_actuator_names']]
        self.stow = np.array([self.data.qpos[self.model.joint(n).qposadr[0]] for n in joints])
        bounds = np.array([self.model.joint(n).range for n in joints])
        rng = np.random.default_rng(seed)
        starts = ((self.stow,) + tuple(np.asarray(q) for q in ORACLE_IK_SEEDS[context['arm']])
                  + tuple(rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(8)))
        arm = SimpleNamespace(name=context['arm'], joint_names=joints,
                              stow=self.stow, ee_site=context['ee_site'],
                              joint=lambda i: joints[i], actuator=lambda i: actuators[i],
                              controller_seeds=lambda: starts)
        # Search a small set of physical base poses. Only locally computed
        # collision-free paths are used; no private scoring state is available.
        face = context['shelf_face_x']
        best = None
        nominal = min(float(self.target[0]) - (0.60 if arm.name == 'xarm7' else 0.68), 0.05)
        positions = tuple(dict.fromkeys((nominal, 0.05, 0.0, -0.10, -0.18)))
        for final_x, final_y in [(x, y) for x in positions
                                for y in (0.0, float(self.target[1]))]:
            qpos = np.asarray(context['qpos']).copy()
            qpos[self.base_adr:self.base_adr + 2] = [final_x, final_y]
            self.planner = GoldenArmController(self.model, qpos, 0.015, arm)
            self.planner.ik_starts = starts
            self.planner._solve_segment = self._solve_segment
            approach = self.target.copy()
            approach[0] = face - 0.18
            self.data.qpos[:] = qpos
            # Reconfigure at the outside start, so the approach posture need
            # not connect to the folded stow beside the shelf.
            for start in starts:
                approach_q, approach_error = self.planner._ik(self.data, approach, (start,))
                if approach_error > 0.015 or self.planner._touches_shelf(self.data, approach_q):
                    continue
                points = self.planner._segment(approach, self.target, 0.025)
                reach_knots, error = self._solve_segment(self.data, points, approach_q)
                plan = ArmPlan((self.stow, approach_q), tuple(reach_knots), approach_error, error)
                if best is None or error < best[0]:
                    best = (error, final_x, final_y, self.planner, plan)
                if error <= 0.015:
                    break
            if best is not None and best[0] <= 0.015:
                break
        if best is None:
            raise RuntimeError('no collision-free shelf approach found')
        _, self.final_x, self.final_y, self.planner, self.plan = best
        self.approach_q = self.plan.transit_knots[-1]
        self.base_control = make_base_controller(self.model)

    def _solve_segment(self, data, points, seed):
        knots = []
        error = float('inf')
        for point in points:
            seed, error = self.planner._ik_connected(data, point, seed, self.planner.ik_starts)
            knots.append(seed)
            if not np.isfinite(error):
                break
        return knots or [seed], error

    def act(self, observation):
        t = observation['time']
        self.data.qpos[:] = observation['qpos']
        self.data.qvel[:] = observation['qvel']
        mujoco.mj_forward(self.model, self.data)
        ctrl = self.initial.copy()
        if t < 3.0:
            alpha = min(t / 3.0, 1.0)
            alpha = alpha * alpha * (3 - 2 * alpha)
            ctrl[self.arm_ids] = self.stow + alpha * (self.approach_q - self.stow)
        elif t < 15.0:
            ctrl[self.arm_ids] = self.approach_q
        else:
            knots = (self.approach_q,) + self.plan.reach_knots
            ctrl[self.arm_ids] = self.planner._interpolate(knots, (t - 15.0) / 5.0)
        desired_x = self.start_x if t < 3.0 else self.final_x
        desired_y = 0.0 if t < 3.0 else self.final_y
        base = self.data.body('base')
        rotation = base.xmat.reshape(3, 3)
        delta = rotation[:2, :2].T @ (np.array([desired_x, desired_y]) - base.xpos[:2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        twist = np.array([*np.clip(1.5 * delta, -0.12, 0.12), np.clip(-2 * yaw, -0.3, 0.3)])
        ctrl[self.wheel_ids] = self.base_control(self.data, twist)
        ctrl[self.arm_ids] += (self.data.qfrc_bias[self.arm_dofs]
                               / self.model.actuator_gainprm[self.arm_ids, 0])
        return ctrl


def make_controller():
    return ShelfController()
