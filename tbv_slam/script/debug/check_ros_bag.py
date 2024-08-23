# %%
import sys

# sys.path.insert(0, "/opt/ros/noetic/lib/python3/dist-packages")

from bagpy import bagreader
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from bagpy import bagreader

# from cv_bridge import CvBridge
import cv2
import numpy as np
import rosbag


bag_file_path = "/workspaces/tbv_ws/datasets/2023_TBV_Radar_SLAM/radar_data_ORU/oxford-eval-sequences/2019-01-10-11-46-21-radar-oxford-10k/radar/2019-01-10-11-46-21-radar-oxford-10k.bag"

# %%
# Open the bag file
# b = bagreader(bag_file_path)

# Get the list of topics
# print(b.topic_table)

# %%
# Open the bag file
bag = rosbag.Bag(bag_file_path, "r")

# topic, msg, t = next(bag.read_messages())
# Iterate through messages
# for topic, msg, t in bag.read_messages():
#     print(topic, msg, t)
topic, msg, t = next(bag.read_messages("/Navtech/Polar"))
cv_image = cv2.imdecode(
    np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR
)  # Adjust encoding as needed
cv2.imshow("Radar Image", cv_image)
cv2.waitKey(1)

# Close the bag file
bag.close()

# %%
