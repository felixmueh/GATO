#!/usr/bin/env python3
"""Read-only controller observer. Run with the simulator's PAL message types.

PAL and stock Humble ControllerState definitions can differ. Observing from the
PAL container avoids silently decoding that service with the wrong schema.
"""
import argparse
import json
from pathlib import Path
import time

import rclpy
from controller_manager_msgs.srv import ListControllers

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stop-file", type=Path, required=True)
parser.add_argument("--duration", type=float, default=100.0)
args = parser.parse_args()
rclpy.init()
node = rclpy.create_node("gato_fault_controller_observer")
client = node.create_client(ListControllers, "/controller_manager/list_controllers")
future = None
next_query = 0.0
deadline = time.monotonic() + args.duration
try:
    while time.monotonic() < deadline and not args.stop_file.exists():
        rclpy.spin_once(node, timeout_sec=0.02)
        now = time.monotonic()
        if future is not None and future.done():
            result = future.result()
            if result is not None:
                print(json.dumps({"wall": now, "controllers": [{"name": c.name, "state": c.state} for c in result.controller]}), flush=True)
            future = None
        if future is None and now >= next_query and client.service_is_ready():
            future = client.call_async(ListControllers.Request())
            next_query = now + 0.2
finally:
    node.destroy_node()
    rclpy.shutdown()
