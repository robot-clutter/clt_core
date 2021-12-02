SURFACE_SIZE = 0.25  # size = half of dimension
# SURFACE_SIZE = 0.125  # size = half of dimension
CROP_TABLE = 193  # always square
# CROP_TABLE = 96

from clt_core.util.orientation import Quaternion, transform_points
from clt_core.util.math import min_max_scale, get_distance_of_two_bbox, OrnsteinUhlenbeckActionNoise, NormalNoise, \
    ConvexHull, LineSegment2D
from clt_core.util.memory import ReplayBuffer
from clt_core.util.cv_tools import PinholeCameraIntrinsics, PointCloud, Feature, get_circle_mask, get_circle_mask_2, rgb2gray, calc_m_per_pxl
import clt_core.util.pybullet as pybullet

from clt_core.core import MDP, Env, Robot, Camera, Transition, Agent, run_episode, train, eval, analyze_data, Push
from clt_core.env import UR5Bullet, BulletEnv
from clt_core.util.info import Logger, info, warn, error, natural_keys

from clt_core.conv_ae import ConvAe