# Read-only c3po Gazebo diagnostic

`collect.py` reads advertised model/controller configuration and passively
observes clock and joint-state timing for10 seconds. It does not publish
commands, move the arm, switch/load controllers, write parameters, build
modules or alter the ROS/system environment. Run it while the existing
headless simulator is running, without starting another motion study.

The diagnostic records:

- Advertised `robot_description` from discovered read-only parameter services
  or transient-local String topics: exact model XML, arm-joint declared
  damping/friction/limits, child-link inertials, Gazebo XML/plugin blocks and
  ROS2-control hardware plugin names.
- Discovered controller-manager ListControllers responses and relevant
  manager/controller declared rates, joint/interfaces, gains and losses.
  Right-arm and joint-state-broadcaster queries take priority over other
  controller nodes. Graph candidates and selected topic/service names remain.
- A compact clock/joint-state CSV with ROS timestamps and host monotonic
  receive times together, plus observed rates, clock/header versus host
  advancement, repeated/backward stamps and partial errors.

**Advertised URDF and declared parameters do not prove effective Gazebo
physics.** The simulator can use cached or overridden SDF/physics settings;
missing damping/friction declarations do not prove zero losses. The script
does not follow plugin parameter-file references or query speculative simulator
tools. Timing is passive receiver evidence, not actuator torque or motion
qualification. It records no effort commands or measured actuator torque.

Use the same already-working c3po ROS/container shell as the prior Gazebo run.
Its existing simulation profile/domain1/CycloneDDS/loopback settings must remain
in place. An SSH shell alone may lack that environment. Use the active
ROS-capable `python3`, not a separate venv that lacks ROS. The request records
hostname, interpreter and only the allowlisted `ROS_DOMAIN_ID`,
`GATO_ROS_PROFILE`, `RMW_IMPLEMENTATION`; it does not dump the environment.

From the laptop/host that already has working c3po SSH access, copy the single
file (adjust the local checkout path if necessary):

```sh
scp /workspace/GATO-experiment-runner-simple/experiments/diagnostics/gazebo_readonly/collect.py c3po:/tmp/gato-gazebo-readonly.py
ssh -t c3po
```

Enter/source the existing ROS-capable environment exactly as for the prior
successful Gazebo invocation; do not start/reset the simulator or change DDS.
Then, inside that existing shell/container:

```sh
python3 -c 'import rclpy, controller_manager_msgs.srv, rcl_interfaces.srv, sensor_msgs.msg, rosgraph_msgs.msg'
python3 /tmp/gato-gazebo-readonly.py --seconds 10 --output /tmp/gato-gazebo-readonly-results
```

If the file was copied to the host but ROS runs inside the existing container,
make the same file visible there with the existing mount or `docker cp`; no
container recreation or package installation is needed. Copying a file does
not source the ROS environment. For the previously identified existing PAL
container, the optional host-side copy is:

```sh
docker cp /tmp/gato-gazebo-readonly.py tiago-pro-12-dev-felix:/tmp/gato-gazebo-readonly.py
```

Default collection is bounded by15 seconds of cumulative configuration queries
(1.5 s per read-service wait) plus10 seconds of passive timing. Allowed maxima
are25 s discovery and20 s sampling; no multi-minute scan occurs. Caps are six
description parameter candidates, two managers, ten prioritized controller
nodes,100 relevant parameter names/node and20,000 messages per timing stream.
Any omitted timing messages are explicitly counted. Alternative visible topic
names can be selected with `--clock-topic /name --joint-topic /name`; graph
selection/candidates are always retained.

Every invocation creates a fresh UTC timestamp directory and adjacent small
gzip archive, even when imports/discovery fail. Exit1 means incomplete data;
keep its archive and `errors.json`. Missing responses do not become invented
zero losses/rates. The printed archive is the collection artifact. If it is
inside a container, copy that exact printed path to the host, then from the
laptop:

```sh
scp c3po:/tmp/gato-gazebo-readonly-results/PRINTED_TIMESTAMP.tar.gz .
```

Replace `PRINTED_TIMESTAMP` with the exact printed timestamp including
microseconds. Return that archive for comparison with the saved native/model
and controller assumptions. No Git publication or experiment rerun is needed
to use this standalone diagnostic.
