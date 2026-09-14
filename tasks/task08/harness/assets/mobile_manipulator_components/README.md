# Mobile manipulator stock components

This directory contains individual components for the task08 from-scratch
design workspace. It deliberately contains no assembled chassis geometry.

- mecanum_wheel_left.xml and mecanum_wheel_right.xml are standalone,
  simulation-ready examples of the two wheel handednesses. Copy each example
  body into the robot and replace EXAMPLE with the required wheel suffix.
- battery.xml defines the supplied battery pack.
- payload.xml defines the 1 kg pickup weight used at each shelf target.
- arm_adapter.xml defines the supplied 180 x 180 x 10 mm, 0.875 kg
  universal adapter plate. The verifier inserts this exact part.
- aluminum_profiles.xml shows individual 2020, 2040, and 4040 stock
  aluminum-profile members. They are examples, not a preselected frame.
- component_catalog.json records quantities, dimensions, masses, profile
  linear densities, and the mecanum X-configuration mapping.

For an aluminum member of a different cut length, set its box half-length to
half the full cut length and set its mass to:

linear_density_kg_m * full_length_m

The assembled robot interface uses `base` with free joint `base_free`; wheel
suffixes `FL`, `FR`, `RL`, and `RR`; and a site named `arm_mount_site` directly
under `base`. The site is the underside of the supplied adapter and its +z
axis is the mounting normal. The verifier removes the submitted `arm_*`
subtree, inserts the stock adapter, and then mounts canonical Panda, UR5e, and
xArm7 models to the same site. The UR5e bolt pattern includes the catalogued
180-degree yaw; the other two use zero yaw.

`robot.xml` should retain the supplied Panda subtree so it loads in the public
workspace, but that arm and any submitted arm assets are not trusted for
scoring. Do not model the adapter as part of the chassis: it is provided and
inserted automatically.

The designer must choose the profile sizes, cut lengths, layout, joints,
mounts, and wheel locations. The XML files are reference models rather than
files intended to be included together unchanged.
