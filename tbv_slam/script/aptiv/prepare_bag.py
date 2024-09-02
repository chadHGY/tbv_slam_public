# %%
import rosbag
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import numpy.matlib as matlib
import rospy
import h5py
from scipy.special import expit
import os
from decimal import *
from math import sin, cos, atan2, sqrt
import geometry_msgs
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from pyproj import Transformer
import matplotlib.pyplot as plt

MATRIX_MATCH_TOLERANCE = 1e-4


def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix(
        [[1, 0, 0], [0, cos(rpy[0]), -sin(rpy[0])], [0, sin(rpy[0]), cos(rpy[0])]]
    )
    R_y = np.matrix(
        [[cos(rpy[1]), 0, sin(rpy[1])], [0, 1, 0], [-sin(rpy[1]), 0, cos(rpy[1])]]
    )
    R_z = np.matrix(
        [[cos(rpy[2]), -sin(rpy[2]), 0], [sin(rpy[2]), cos(rpy[2]), 0], [0, 0, 1]]
    )
    R_zyx = R_z * R_y * R_x
    return R_zyx


def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except ValueError:
        # w is non-real
        w = 0

    # Due to numerical precision the value passed to `sqrt` may be a negative of the order 1e-15.
    # To avoid this error we clip these values to a minimum value of 0.
    x = sqrt(max(1 + R_xx - R_yy - R_zz, 0)) / 2
    y = sqrt(max(1 + R_yy - R_xx - R_zz, 0)) / 2
    z = sqrt(max(1 + R_zz - R_yy - R_xx, 0)) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


def ProcessFrame(params, Tprev, stamp):  # write Fibonacci series up to n
    Tinc = build_se3_transform(params)
    Tupd = Tprev * Tinc

    t = geometry_msgs.msg.TransformStamped()
    t.header.frame_id = "/world"
    t.header.stamp = stamp
    t.child_frame_id = "/navtech"
    # t.transform.translation.x = Tupd[0, 3]
    # t.transform.translation.y = Tupd[1, 3]
    t.transform.translation.x = params[0]
    t.transform.translation.y = params[1]
    t.transform.translation.z = 0.0

    qupd = so3_to_quaternion(Tupd[0:3, 0:3])
    t.transform.rotation.x = qupd[1]
    t.transform.rotation.y = qupd[2]
    t.transform.rotation.z = qupd[3]
    t.transform.rotation.w = qupd[0]

    return Tupd, t


def create_2d_point_cloud(
    points: np.array, timestamp: rospy.rostime.Time
) -> PointCloud2:
    """
    Create a PointCloud2 message.
    :param points: Nx3 array of XYI points.
    :param timestamp: ROS timestamp
    :return: sensor_msgs/PointCloud2 message
    """
    # msg = PointCloud2()
    # msg.header.stamp = timestamp
    # msg.header.frame_id = "map"
    # msg.height = 1
    # msg.width = points.shape[0]
    # msg.fields = [
    #     PointField("x", 0, PointField.FLOAT32, 1),
    #     PointField("y", 4, PointField.FLOAT32, 1),
    #     PointField("z", 8, PointField.FLOAT32, 1),
    #     PointField("intensity", 12, PointField.FLOAT32, 1),
    # ]
    # msg.is_bigendian = False
    # msg.point_step = 16  # Update point_step to account for intensity field
    # msg.row_step = msg.point_step * points.shape[0]
    # msg.is_dense = True

    # Combine points and intensities into a single array
    # msg.data = np.asarray(points, np.float32).tobytes()
    # return msg

    # Create the header
    header = Header()
    header.stamp = timestamp
    header.frame_id = "map"  # Frame in which the point cloud is defined

    # Define the fields of the PointCloud2 message
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("intensity", 12, PointField.FLOAT32, 1),
    ]

    # Create the PointCloud2 message
    point_cloud_msg = pc2.create_cloud(header, fields, points)
    return point_cloud_msg


def latlon_to_xy(lat, lon, zone=32):
    # Create a Transformer object for transforming coordinates
    transformer = Transformer.from_crs(
        f"epsg:4326", f"epsg:{32600+zone}", always_xy=True
    )

    # Apply the transformation (note the order: lon, lat)
    x, y = transformer.transform(lon, lat)
    return x, y


def get_gps_trajectory(data_path, log_name) -> np.ndarray:
    with h5py.File(os.path.join(data_path, "Meta", log_name + ".h5"), "r") as meta_fh:
        host_position = meta_fh["T0"][:, 3, :2]
        meta_timestamps = meta_fh["timestamps"][:, 0]

    with h5py.File(os.path.join(data_path, "GPS", log_name + ".h5"), "r") as gps_fh:
        from pyquaternion import Quaternion

        applanix_pos = gps_fh["sensors"]["WUP_GoFast_Applanix"]["position"][:, :2]
        applanix_ts = gps_fh["sensors"]["WUP_GoFast_Applanix"]["timestamps"][:, 0]
        applanix_orient = gps_fh["sensors"]["WUP_GoFast_Applanix"]["orientation"][()]
        applanix_quat = [
            Quaternion(applanix_orient[i]) for i in range(applanix_orient.shape[0])
        ]

    # Mapping meta to applanix index
    meta_applanix_map = np.argmin(
        np.abs(meta_timestamps[:, None] - applanix_ts[None, :]), axis=-1
    )

    # Create the applanix position in T0 VCS
    appl_pos_xy = np.stack(latlon_to_xy(applanix_pos[:, 1], applanix_pos[:, 0]), axis=1)

    # y points to the right
    appl_pos_xy[:, 1] *= -1

    # Normalize to VCS of timestamp 0
    appl_pos_rel_xy = appl_pos_xy - appl_pos_xy[meta_applanix_map[0]]

    ## Initial orientation
    # Velocity based
    vel_0 = (
        appl_pos_rel_xy[meta_applanix_map[10]] - appl_pos_rel_xy[meta_applanix_map[0]]
    )
    angle = np.arctan2(vel_0[1], vel_0[0])
    rot_mat_t = np.array(
        [[np.cos(-angle), np.sin(-angle)], [-np.sin(-angle), np.cos(-angle)]]
    )

    # Applanix CS based
    # rot_mat_t = applanix_quat[meta_applanix_map[0]].rotation_matrix[:2,:2] @ np.array([[0,1], [-1,0]])
    appl_pos_rel_xy = appl_pos_rel_xy @ rot_mat_t
    return appl_pos_rel_xy


# %% Example point cloud data, intensity values, and timestamps
point_clouds = [
    (
        np.array([[1.0, 2.0, 0.5], [4.0, 5.0, 0.6], [7.0, 8.0, 0.7]]),
        rospy.Time.from_sec(1696940216.247674),
    ),
    (
        np.array([[10.0, 11.0, 0.8], [13.0, 14.0, 0.9]]),
        rospy.Time.from_sec(1696940216.29767),
    ),
    (
        np.array([[19.0, 20.0, 1.1], [22.0, 23.0, 1.2], [25.0, 26.0, 1.3]]),
        rospy.Time.from_sec(1696940216.34774),
    ),
]

# %% load data
data_path = "/workspaces/tbv_ws/datasets/aptiv/cluster/aiperception_cluster_data/online/TrainingTooling_Data/WUP_GoFast_SFW_Gen6_TT_v2_batch_pivot_fix_v15-6-0-0/End2End"

log_name = (
    "WUP_GoFast_Gen6-20231010T121645Z053__20231010T121656Z189_20231010T121708Z389"
)
point_cloud_path = f"{data_path}/output/semseg_stationary/{log_name}.h5"
output_dir = f"/workspaces/tbv_ws/datasets/2023_TBV_Radar_SLAM/radar_data_ORU/Aptiv/{log_name}/radar"
slam_estimation_path = f"{data_path}/output/slam_estimation/{log_name}.h5"


with h5py.File(slam_estimation_path, "r") as f:
    ego_positions = f["ego_positions"][:]
ego_positions = np.asarray(ego_positions)


with h5py.File(point_cloud_path, "r") as f:
    pts_stationary = f["pts_stationary"][:]
    timestamps = f["timestamps"][:]
timestamps[0]


# %%
# plot the trajectory
appl_pos_rel_xy = get_gps_trajectory(data_path, log_name)
plt.plot(appl_pos_rel_xy[:, 1], appl_pos_rel_xy[:, 0], label="Applanix")
plt.plot(ego_positions[:, 1], ego_positions[:, 0], label="Slam")
plt.legend()
plt.show()


# %%
pose_init, prev_pos = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]
Tpose = build_se3_transform(pose_init)

point_clouds = []
for frame_id, timestamp in enumerate(timestamps):
    ros_timestamp = rospy.Time.from_sec(timestamp)

    cur_pos = ego_positions[frame_id]
    if frame_id == 0:
        curr_inc = [0, 0, 0, 0, 0, 0]
    else:
        # x, y, yaw = (
        #     float(cur_pos[0] - prev_pos[0]),
        #     float(cur_pos[1] - prev_pos[1]),
        #     float(cur_pos[2] - prev_pos[2]),
        # )
        x, y, yaw = (
            float(cur_pos[0]),
            float(cur_pos[1]),
            # float(cur_pos[2]),
            float(cur_pos[2] - prev_pos[2]),
        )
        # make yaw from rad to deg
        # yaw = yaw * 180 / np.pi
        curr_inc = [
            Decimal(x),  # x
            Decimal(y),  # y
            0,
            0,
            0,
            Decimal(yaw),  # yaw
        ]

    # --- tf transform data ---
    Tpose, tf_transform = ProcessFrame(
        curr_inc, Tpose, ros_timestamp
    )  ## read input transormation and perform fwdkinematics
    prev_pos = cur_pos

    # --- point cloud data ---
    frame_points = pts_stationary[pts_stationary[:, 0] == frame_id, ...]
    points = np.hstack(
        (
            frame_points[:, 1:3],  # x, y
            np.zeros((frame_points.shape[0], 1)),  # z
            expit(frame_points[:, 3:4]),  # transform from logit to confidence
        )
    )
    # points = frame_points[:, 1:]  # (N, 3(x,y,I))
    # points[:, 2] = expit(points[:, 2])  # transform from logit to confidence
    point_clouds.append((points, tf_transform, ros_timestamp))


# %% Write to rosbag
os.makedirs(output_dir, exist_ok=True)
bag = rosbag.Bag(f"{output_dir}/{log_name}.bag", "w")
try:
    for points, tf_transform, timestamp in point_clouds:
        point_cloud_msg = create_2d_point_cloud(points, timestamp)
        bag.write(topic="/Aptiv/Pt_VCS", msg=point_cloud_msg, t=timestamp)
        tvek = TFMessage()
        tvek.transforms.append(tf_transform)
        bag.write(topic="/tf", msg=tvek, t=timestamp)
        odom = Odometry()
        odom.header.stamp = tf_transform.header.stamp
        odom.header.frame_id = "/world"
        odom.child_frame_id = "/navtech"
        odom.pose.pose.position = tf_transform.transform.translation
        odom.pose.pose.orientation = tf_transform.transform.rotation
        bag.write(topic="/gt", msg=odom, t=timestamp)

finally:
    bag.close()

# %% check gt data format
"""
import pandas as pd

df = pd.read_csv(
    "/workspaces/tbv_ws/datasets/2019_OxfordRadarRobert/2019-01-10-14-36-48-radar-oxford-10k-partial/gt/radar_odometry.csv"
)
# %%
df["x_accum"] = df["x"].cumsum()
df["y_accum"] = df["y"].cumsum()
df["yaw_accum"] = df["yaw"].cumsum()

# %%
from matplotlib import pyplot as plt

# set equal aspect ratio
plt.figure()
# plt.gca().set_aspect("equal", adjustable="box")
plt.plot(df["x_accum"], df["y_accum"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ground Truth")
plt.show()
"""
# %%
