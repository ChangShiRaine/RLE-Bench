"""Root service composition for the single-camera tilted-base variant."""

from . import daemon_main as shared
from .front import FrontPocketCube, CAMERAS, BASE_POSITIONS, BASE_ROTATIONS


def install():
    from harness.service import Service
    if getattr(Service, '_pocket_front_composed', False):
        return
    shared.PocketCube = FrontPocketCube
    shared.CAMERAS = CAMERAS
    shared.OBSERVATIONS = frozenset(
        [f'robot{i}_{field}' for i in range(2) for field in shared.ROBOT_FIELDS]
        + [f'front_{kind}' for kind in ('image', 'depth')])
    shared.install()
    dispatch = Service._dispatch

    def front_dispatch(self, op, msg):
        reply = dispatch(self, op, msg)
        if op == 'task_info':
            reply.update(action_reference_frame="each robot's fixed tilted base; use robot_base_poses",
                         robot_base_poses=[dict(pos=list(pos), quat_xyzw=rot.as_quat().tolist())
                                           for pos, rot in zip(BASE_POSITIONS, BASE_ROTATIONS)])
        return reply

    Service._dispatch = front_dispatch
    Service._pocket_front_composed = True


def main():
    install()
    from harness.daemon_main import main as shared_main
    return shared_main()


if __name__ == '__main__':
    raise SystemExit(main())
