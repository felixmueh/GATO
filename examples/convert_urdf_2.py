import xml.etree.ElementTree as ET
import numpy as np

def rpy_to_matrix(rpy):
    """Convert roll-pitch-yaw to rotation matrix."""
    r, p, y = rpy
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def matrix_to_rpy(R):
    """Convert rotation matrix to roll-pitch-yaw (XYZ order)."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw])

def parse_rpy(s):
    return np.array([float(x) for x in s.split()])

def format_rpy(v):
    return f"{v[0]:.9f} {v[1]:.9f} {v[2]:.9f}"

def convert_urdf_to_z_axis(urdf_in, urdf_out):
    tree = ET.parse(urdf_in)
    root = tree.getroot()

    # Keep track of cumulative transform for each link
    link_tf = {"base_link": np.eye(4)}

    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child  = joint.find("child").attrib["link"]

        origin_tag = joint.find("origin")
        rpy = parse_rpy(origin_tag.attrib.get("rpy", "0 0 0"))
        xyz = np.array([float(x) for x in origin_tag.attrib.get("xyz", "0 0 0").split()])

        axis_tag = joint.find("axis")
        if axis_tag is not None:
            axis = np.array([float(x) for x in axis_tag.attrib["xyz"].split()])
        else:
            axis = np.array([0,0,1])

        # Build the current joint transform (parent->child)
        R = rpy_to_matrix(rpy)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = xyz

        # Apply cumulative transform from parent
        T_parent = link_tf[parent]
        T_world_joint = T_parent @ T

        # If the axis is Y, rotate joint frame -90deg about X so its local z aligns with old y
        if np.allclose(axis, [0,1,0], atol=1e-8):
            R_fix = rpy_to_matrix([-np.pi/2, 0, 0])
            T_world_joint[:3,:3] = T_world_joint[:3,:3] @ R_fix
            axis_tag.attrib["xyz"] = "0 0 1"

        elif np.allclose(axis, [1,0,0], atol=1e-8):
            R_fix = np.eye(3)
            T_world_joint[:3,:3] = T_world_joint[:3,:3] @ R_fix
            axis_tag.attrib["xyz"] = "0 0 1"

        # Compute new origin (parent->joint) expressed in parent frame
        T_new = np.linalg.inv(T_parent) @ T_world_joint
        new_rpy = matrix_to_rpy(T_new[:3,:3])
        new_xyz = T_new[:3,3]

        # Update the XML
        origin_tag.attrib["rpy"] = format_rpy(new_rpy)
        origin_tag.attrib["xyz"] = f"{new_xyz[0]:.6f} {new_xyz[1]:.6f} {new_xyz[2]:.6f}"

        # Store new cumulative transform for this child link
        link_tf[child] = T_world_joint

    tree.write(urdf_out)
    print(f"✅ Converted URDF written to {urdf_out}")

if __name__ == "__main__":
    convert_urdf_to_z_axis("flexiv_description/flexiv_rizon4s_kinematics.urdf", "flexiv_description/flexiv_rizon4s_kinematics_zaxis2.urdf")