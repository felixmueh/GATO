#!/usr/bin/env python3
"""Read advertised model/controller configuration and passive ROS timing; no motion."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import tarfile
import time
import xml.etree.ElementTree as ET


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def model_details(text):
    root = ET.fromstring(text)
    links = {v.get('name'): v for v in root.findall('link')}
    joints = []
    for joint in root.findall('joint'):
        if not joint.get('name', '').startswith(('arm_right_', 'arm_left_')):
            continue
        child = joint.find('child')
        link = links.get(child.get('link')) if child is not None else None
        inertial = link.find('inertial') if link is not None else None
        joints.append(dict(name=joint.get('name'), type=joint.get('type'),
            child_link=child.attrib if child is not None else None,
            dynamics=joint.find('dynamics').attrib if joint.find('dynamics') is not None else None,
            limits=joint.find('limit').attrib if joint.find('limit') is not None else None,
            inertial={v.tag: v.attrib for v in inertial} if inertial is not None else None))
    blocks = '\n'.join(ET.tostring(v, encoding='unicode') for v in root.findall('.//gazebo'))
    plugins = [v.text for v in root.findall('.//ros2_control/hardware/plugin')]
    return dict(meaning='Advertised URDF declarations; not verified effective Gazebo SDF/physics. Missing dynamics does not prove zero losses.',
                arm_joints=joints, ros2_control_hardware_plugins=plugins), blocks


def cadence(rows):
    if len(rows) < 2:
        return dict(messages=len(rows), available=False)
    host = [b[0]-a[0] for a, b in zip(rows, rows[1:])]
    stamps = [b[1]-a[1] for a, b in zip(rows, rows[1:])]
    elapsed = rows[-1][0]-rows[0][0]
    return dict(messages=len(rows), host_elapsed_s=elapsed,
        stamp_elapsed_s=rows[-1][1]-rows[0][1],
        stamp_per_host_ratio=(rows[-1][1]-rows[0][1])/elapsed if elapsed else None,
        received_hz=(len(rows)-1)/elapsed if elapsed else None,
        host_interval_median_s=statistics.median(host),
        host_interval_p95_s=sorted(host)[math.ceil(.95*len(host))-1],
        stamp_interval_median_s=statistics.median(stamps),
        repeated_stamps=sum(v == 0 for v in stamps), backward_stamps=sum(v < 0 for v in stamps))


def parameter_value(value):
    fields = {1: 'bool_value', 2: 'integer_value', 3: 'double_value', 4: 'string_value',
              5: 'byte_array_value', 6: 'bool_array_value', 7: 'integer_array_value',
              8: 'double_array_value', 9: 'string_array_value'}
    field = fields.get(value.type)
    if not field:
        return None
    result = getattr(value, field)
    return list(result) if value.type >= 5 else result


def collect(out, args):
    # Lazy imports allow --help and CPU parsing checks outside ROS.
    import rclpy
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
    from rcl_interfaces.srv import GetParameters, ListParameters
    from controller_manager_msgs.srv import ListControllers
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    errors, models, reads, controllers = [], [], [], []
    samples = {'clock': [], 'joint_states': []}
    dropped = {key: 0 for key in samples}
    active, latest_clock = False, None
    model_has_right_arm = False
    deadline = time.monotonic()+args.discovery_seconds
    node = stream = None
    try:
        rclpy.init(args=[])
        node = rclpy.create_node('gato_gazebo_readonly_'+str(time.time_ns()))
        stream = (out/'timing.csv').open('x', newline='')
        writer = csv.writer(stream)
        writer.writerow(['kind', 'host_monotonic_s', 'stamp_s', 'latest_clock_s'])
    except Exception:
        for operation in ([stream.close] if stream else []) + ([node.destroy_node] if node else []) + [rclpy.shutdown]:
            try:
                operation()
            except Exception:
                pass
        raise

    def observe(kind, stamp):
        nonlocal latest_clock
        if kind == 'clock':
            latest_clock = stamp
        if active:
            row = (time.monotonic(), stamp)
            if len(samples[kind]) < 20000:
                samples[kind].append(row); writer.writerow([kind, *row, latest_clock])
            else:
                dropped[kind] += 1

    def remember_model(text, source):
        nonlocal model_has_right_arm
        if any(m['source'] == source for m in models):
            return
        if text and len(text.encode()) <= 8*1024*1024:
            digest = hashlib.sha256(text.encode()).hexdigest()
            models.append(dict(source=source, sha256=digest))
            if not (out/'model.urdf').exists():
                (out/'model.urdf').write_text(text)
                try:
                    details, blocks = model_details(text)
                    write(out/'arm_model.json', details); (out/'gazebo_blocks.xml').write_text(blocks)
                    expected = {'arm_right_'+str(j)+'_joint' for j in range(1, 8)}
                    model_has_right_arm = expected <= {j['name'] for j in details['arm_joints']}
                except Exception as error:
                    errors.append(dict(stage='parse_advertised_model', error=str(error)))

    def call(service, kind, request):
        if time.monotonic() >= deadline:
            errors.append(dict(stage=service, error='Discovery budget exhausted')); return None
        client = node.create_client(kind, service)
        try:
            timeout = min(args.service_timeout, max(0., deadline-time.monotonic()))
            if not client.wait_for_service(timeout_sec=timeout):
                raise TimeoutError('Read-only service unavailable')
            future = client.call_async(request)
            until = min(deadline, time.monotonic()+args.service_timeout)
            while not future.done() and time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=min(.05, max(0., until-time.monotonic())))
            if not future.done():
                future.cancel(); raise TimeoutError('Read-only response timeout')
            return future.result()
        except Exception as error:
            errors.append(dict(stage=service, error=str(error))); return None
        finally:
            try:
                node.destroy_client(client)
            except Exception as error:
                errors.append(dict(stage=service+'/cleanup', error=str(error)))

    try:
        # Brief graph discovery, then inspect only model/manager/controller candidates.
        until = time.monotonic()+1.
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.05)
        topics = dict(node.get_topic_names_and_types())
        services = dict(node.get_service_names_and_types())
        nodes = node.get_node_names_and_namespaces()
        def select(requested, suffix, type_name):
            candidates = sorted(k for k, v in topics.items() if type_name in v and k.endswith(suffix))
            if requested:
                return requested, candidates
            return ('/'+suffix if '/'+suffix in candidates else candidates[0] if candidates else None), candidates
        clock, clocks = select(args.clock_topic, 'clock', 'rosgraph_msgs/msg/Clock')
        joints, joint_topics = select(args.joint_topic, 'joint_states', 'sensor_msgs/msg/JointState')
        if clock:
            node.create_subscription(Clock, clock, lambda m: observe('clock', m.clock.sec+m.clock.nanosec*1e-9), qos_profile_sensor_data)
        if joints:
            node.create_subscription(JointState, joints, lambda m: observe('joint_states', m.header.stamp.sec+m.header.stamp.nanosec*1e-9), qos_profile_sensor_data)
        descriptions = sorted(k for k, v in topics.items() if 'std_msgs/msg/String' in v and k.endswith('robot_description'))[:3]
        for topic in descriptions:
            node.create_subscription(String, topic, lambda m, t=topic: remember_model(m.data, 'topic:'+t),
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        managers = sorted(k for k, v in services.items() if 'controller_manager_msgs/srv/ListControllers' in v)[:2]
        get_services = sorted(k for k, v in services.items() if 'rcl_interfaces/srv/GetParameters' in v)
        relevant = {k: v for k, v in services.items() if any(w in k for w in ('controller_manager', 'robot_description', 'get_parameters', 'list_parameters'))}
        write(out/'graph.json', dict(nodes=nodes[:100], relevant_services=relevant,
            selected_clock=clock, clock_candidates=clocks, selected_joint_states=joints,
            joint_state_candidates=joint_topics, description_topics=descriptions, manager_services=managers))
        priority = sorted(get_services, key=lambda k: ('robot_state_publisher' not in k, 'gazebo' not in k, k))
        for service in priority[:6]:
            response = call(service, GetParameters, GetParameters.Request(names=['robot_description']))
            if response and response.values and response.values[0].type == 4:
                remember_model(response.values[0].string_value, 'parameter:'+service)
                if (out/'model.urdf').exists():
                    break
        parameter_nodes = []
        for service in managers:
            response = call(service, ListControllers, ListControllers.Request())
            if response is None:
                continue
            manager = service.rsplit('/list_controllers', 1)[0]
            parameter_nodes.append(manager)
            for controller in response.controller:
                controllers.append(dict(manager=manager, name=controller.name, type=controller.type,
                    state=controller.state, claimed_interfaces=list(controller.claimed_interfaces)))
                prefix = manager.rsplit('/', 1)[0]
                parameter_nodes.append(prefix+'/'+controller.name)
        manager_nodes = [s.rsplit('/list_controllers', 1)[0] for s in managers]
        priority = sorted(set(parameter_nodes), key=lambda p: (p not in manager_nodes,
            'right' not in p and 'joint_state' not in p, p))
        for prefix in priority[:10]:
            list_service, get_service = prefix+'/list_parameters', prefix+'/get_parameters'
            names = ['update_rate', 'publish_rate', 'state_publish_rate', 'use_sim_time', 'joints',
                     'command_interfaces', 'state_interfaces', 'interface_name']
            if list_service in services:
                response = call(list_service, ListParameters, ListParameters.Request(prefixes=[], depth=10))
                if response:
                    names += [k for k in response.result.names if any(w in k for w in ('update_rate', 'publish_rate', 'gains.', 'damping', 'friction'))]
            names = list(dict.fromkeys(names))[:100]
            if get_service in services:
                response = call(get_service, GetParameters, GetParameters.Request(names=names))
                if response:
                    reads.append(dict(service=get_service, values={k: parameter_value(v) for k, v in zip(names, response.values) if v.type}))
        active = True
        until = time.monotonic()+args.seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=min(.05, max(0., until-time.monotonic())))
    except Exception as error:
        errors.append(dict(stage='collection', error=str(error)))
    finally:
        for stage, operation in [('close_timing_file', stream.close), ('destroy_observer', node.destroy_node), ('shutdown_observer', rclpy.shutdown)]:
            try:
                operation()
            except Exception as error:
                errors.append(dict(stage=stage, error=str(error)))
        write(out/'controllers.json', dict(controllers=controllers, parameters=reads, advertised_models=models))
        write(out/'errors.json', errors)
        summary = dict(meaning='Passive receive/header timing. Advertised model/controller values are declarations, not verified effective Gazebo physics.',
            sampling_host_seconds_requested=args.seconds, streams={k: cadence(v) for k, v in samples.items()},
            omitted_after_20000_messages=dropped, errors=errors,
            advertised_right_arm_complete=model_has_right_arm,
            complete=bool(model_has_right_arm and controllers and
                          len(samples['clock']) >= 2 and len(samples['joint_states']) >= 2 and not errors))
        write(out/'summary.json', summary)
    return 0 if summary['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.cwd()/'.gazebo-readonly-results', help='Parent for a fresh UTC timestamp directory and archive')
    parser.add_argument('--seconds', type=float, default=10., help='Passive timing window,1–20s')
    parser.add_argument('--discovery-seconds', type=float, default=15., help='Total read-service discovery budget,1–25s')
    parser.add_argument('--service-timeout', type=float, default=1.5, help='Per read-service wait,0.1–2s within total budget')
    parser.add_argument('--clock-topic'); parser.add_argument('--joint-topic')
    args = parser.parse_args()
    for name, low, high in [('seconds', 1, 20), ('discovery_seconds', 1, 25), ('service_timeout', .1, 2)]:
        if not math.isfinite(getattr(args, name)) or not low <= getattr(args, name) <= high:
            parser.error(name+' outside bounded range')
    out = args.output.expanduser().resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'request.json', dict(seconds=args.seconds, discovery_seconds=args.discovery_seconds,
        service_timeout=args.service_timeout, clock_topic=args.clock_topic, joint_topic=args.joint_topic,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), python=sys.executable,
        hostname=socket.gethostname(),
        ros_environment={k: os.environ.get(k) for k in ('ROS_DOMAIN_ID', 'GATO_ROS_PROFILE', 'RMW_IMPLEMENTATION')}))
    status = 1
    try:
        status = collect(out, args)
    except Exception as error:
        write(out/'errors.json', [dict(stage='startup_or_finalize', error=str(error))])
    finally:
        archive = Path(str(out)+'.tar.gz')
        with tarfile.open(archive, 'x:gz', compresslevel=1) as bundle:
            bundle.add(out, arcname=out.name)
        print('Results: '+str(out)+'\nArchive: '+str(archive), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
