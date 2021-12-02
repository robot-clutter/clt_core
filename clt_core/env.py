"""
Env
===

This module contains classes for defining an environment.
"""
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import cv2

from clt_core.util.robotics import Trajectory
from clt_core.util.orientation import Quaternion, Affine3
from clt_core.util.math import LineSegment2D, get_distance_of_two_bbox, min_max_scale
import clt_core.util.pybullet as bullet_util
import math
from clt_core.core import Env, Robot, Camera, Object
from clt_core.util.cv_tools import PinholeCameraIntrinsics, Feature

import matplotlib.pyplot as plt

import clt_assets

class Button:
    def __init__(self, title):
        self.id = p.addUserDebugParameter(title, 1, 0, 1)
        self.counter = p.readUserDebugParameter(self.id)
        self.counter_prev = self.counter

    def on(self):
        self.counter = p.readUserDebugParameter(self.id)
        if self.counter % 2 == 0:
            return True
        return False


class NamesButton(Button):
    def __init__(self, title):
        super(NamesButton, self).__init__(title)
        self.ids = []

    def show_names(self, objects):
        if self.on() and len(self.ids) == 0:
            for obj in objects:
                self.ids.append(p.addUserDebugText(text=obj.name, textPosition=[0, 0, 0], parentObjectUniqueId=obj.body_id))

        if not self.on():
            for i in self.ids:
                p.removeUserDebugItem(i)
            self.ids = []


class SceneGenerator:
    def __init__(self, params):
        self.params = params
        self.finger_size = 0.001
        self.surface_size = [0.25, 0.25]
        self.rng = np.random.RandomState()
        self.objects = []

    def reset(self):
        self.objects = []

    def seed(self, seed):
        self.rng.seed(seed)

    def generate_scene(self, hugging=False):
        target = Object(name='target')

        #   Randomize size
        a = self.params['target']['min_bounding_box'][2]
        b = self.params['target']['max_bounding_box'][2]
        if a > b:
            b = a
        target_height = self.rng.uniform(a, b)

        a = self.params['target']['min_bounding_box'][0]
        b = self.params['target']['max_bounding_box'][0]
        if a > b:
            b = a
        target_length = self.rng.uniform(a, b)

        a = self.params['target']['min_bounding_box'][1]
        b = min(target_length, self.params['target']['max_bounding_box'][1])
        if a > b:
            b = a
        target_width = self.rng.uniform(a, b)
        target.size = [target_length, target_width, target_height]

        # Randomize position
        theta = self.rng.uniform(0, 2 * math.pi)
        table_line_segments = [LineSegment2D(np.array([self.surface_size[0], -self.surface_size[1]]),
                                             np.array([self.surface_size[0], self.surface_size[1]])),
                               LineSegment2D(np.array([self.surface_size[0], self.surface_size[1]]),
                                             np.array([-self.surface_size[0], self.surface_size[1]])),
                               LineSegment2D(np.array([-self.surface_size[0], self.surface_size[1]]),
                                             np.array([-self.surface_size[0], -self.surface_size[1]])),
                               LineSegment2D(np.array([-self.surface_size[0], -self.surface_size[1]]),
                                             np.array([self.surface_size[0], -self.surface_size[1]]))]
        distance_table = np.linalg.norm(
            LineSegment2D(np.zeros(2), np.array([math.cos(theta), math.sin(theta)])).get_first_intersection_point(
                table_line_segments))

        max_distance = distance_table - math.sqrt(math.pow(target_length, 2) + math.pow(target_width, 2))
        distance = min(1, abs(self.rng.normal(0, 0.5))) * max_distance
        target.pos = np.array([distance * math.cos(theta), distance * math.sin(theta), target_height])

        if not self.params['target'].get('randomize_pos', True):
            target.pos = np.zeros(3)

        #   Randomize orientation
        theta = self.rng.uniform(0, 2 * math.pi)
        quat = Quaternion()
        quat.rot_z(theta)
        target.quat = quat

        self.objects.append(target)


        # Randomize obstacles
        # -------------------

        all_equal_height = self.rng.uniform(0, 1) < self.params['all_equal_height_prob']

        if all_equal_height:
            self.n_obstacles = self.params['nr_of_obstacles'][1]
        else:
            self.n_obstacles = self.params['nr_of_obstacles'][0] + self.rng.randint(self.params['nr_of_obstacles'][1] - self.params['nr_of_obstacles'][0] + 1)  # 5 to 25 obstacles

        for i in range(1, self.n_obstacles + 1):
            # Randomize type (box or cylinder)
            temp = self.rng.uniform(0, 1)
            if temp < self.params['obstacle']['probability_box']:
                type = 'box'
            else:
                type = 'cylinder'
                # Increase the friction of the cylinders to stabilize them
                # self.sim.model.geom_friction[geom_id][0] = 1.0
                # self.sim.model.geom_friction[geom_id][1] = .01
                # self.sim.model.geom_friction[geom_id][2] = .01
                # self.sim.model.geom_condim[geom_id] = 4

            #   Randomize size
            obstacle_length = self.rng.uniform(self.params['obstacle']['min_bounding_box'][0], self.params['obstacle']['max_bounding_box'][0])
            obstacle_width  = self.rng.uniform(self.params['obstacle']['min_bounding_box'][1], min(obstacle_length, self.params['obstacle']['max_bounding_box'][1]))

            pushable_threshold_coeff = self.params['obstacle']['pushable_threshold_coeff']
            if all_equal_height:
                obstacle_height = target_height
            else:
                self.pushable_threshold = target_height + pushable_threshold_coeff * 0.005
                min_h = max(self.params['obstacle']['min_bounding_box'][2], self.pushable_threshold)
                if min_h > self.params['obstacle']['max_bounding_box'][2]:
                    obstacle_height = self.params['obstacle']['max_bounding_box'][2]
                else:
                    obstacle_height = self.rng.uniform(min_h, self.params['obstacle']['max_bounding_box'][2])
            obstacle_height = self.rng.uniform(self.params['obstacle']['min_bounding_box'][2], self.params['obstacle']['max_bounding_box'][2])

            if type == 'box':
                x = obstacle_length
                y = obstacle_width
                z = obstacle_height
            else:
                x = obstacle_length
                y = obstacle_height
                z = 0.0

            # Randomize the positions
            distribution = self.params['obstacle'].get('distribution', 'exponential')
            if distribution == 'exponential':
                spread = 0.01
                if all_equal_height and hugging:  # If all equal spread the objects more in order to avoid flipping of objects during hugging
                    spread = 0.1
                r = self.rng.exponential(spread) + target_length + max(x, y)
                theta = self.rng.uniform(0, 2 * math.pi)
                pos = np.array([r * math.cos(theta) + target.pos[0], r * math.sin(theta) + target.pos[1], z])
            elif distribution == 'uniform':
                pos = np.array([self.rng.uniform(- 0.90 * self.surface_size[0], 0.90 * self.surface_size[0]),
                                self.rng.uniform(- 0.90 * self.surface_size[1], 0.90 * self.surface_size[1]), z])
            else:
                raise RuntimeError('Distribution should be exponential or uniform')


            theta = self.rng.uniform(0, 2 * math.pi)
            quat = Quaternion()
            quat.rot_z(theta)

            self.objects.append(Object(name='obs_' + str(i), pos=pos, quat=quat,
                                       size=[obstacle_length, obstacle_width, obstacle_height]))

    def plot(self, show=False):
        pose_scale = 0.01
        ax = self.objects[0].plot(pose_scale=pose_scale, color=[1, 0, 0, 1])
        for i in range(1, len(self.objects)):
            ax = self.objects[i].plot(pose_scale=pose_scale, color=[0, 0, 1, 1], ax=ax)
        if show:
            plt.show()
        return ax


class UR5Bullet(Robot):
    def __init__(self):
        self.num_joints = 6

        joint_names = ['ur5_shoulder_pan_joint', 'ur5_shoulder_lift_joint',
                       'ur5_elbow_joint', 'ur5_wrist_1_joint', 'ur5_wrist_2_joint',
                       'ur5_wrist_3_joint']

        self.camera_optical_frame = 'camera_color_optical_frame'
        self.ee_link_name = 'finger_tip'
        self.indices = bullet_util.get_joint_indices(joint_names, 0)


        self.joint_configs = {"home": [-2.8927932236757625, -1.7518461461930528, -0.8471216131631573,
                                       -2.1163833167682005, 1.5717067329577208, 0.2502483535771374],
                              "above_table": [-2.8964885089272934, -1.7541597533564786, -1.9212388653019141,
                                              -1.041716266062558, 1.5759665976832087, 0.24964880122853264]}

        self.reset_joint_position(self.joint_configs["home"])

        self.finger = [0.017, 0.017]

    def get_joint_position(self):
        joint_pos = []
        for i in range(self.num_joints):
            joint_pos.append(p.getJointState(0, self.indices[i])[0])
        return joint_pos

    def get_joint_velocity(self):
        joint_pos = []
        for i in range(self.num_joints):
            joint_pos.append(p.getJointState(0, self.indices[i])[1])
        return joint_pos

    def set_joint_position(self, joint_position):
        p.setJointMotorControlArray(bodyIndex=0, jointIndices=self.indices,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_position)

    def reset_joint_position(self, joint_position):
        for i in range(len(self.indices)):
            p.resetJointState(0, self.indices[i], joint_position[i])
        self.set_joint_position(joint_position)

    def get_task_pose(self):
        return bullet_util.get_link_pose(self.ee_link_name)

    def set_task_pose(self, pos, quat):
        link_index = bullet_util.get_link_indices([self.ee_link_name])[0]
        joints = p.calculateInverseKinematics(bodyIndex=0, endEffectorLinkIndex=link_index,
                                              targetPosition=(pos[0], pos[1], pos[2]), 
                                              targetOrientation=quat.as_vector("xyzw"))
        self.set_joint_position(joints)

    def reset_task_pose(self, pos, quat):
        link_index = bullet_util.get_link_indices([self.ee_link_name])[0]
        joints = p.calculateInverseKinematics(bodyIndex=0, endEffectorLinkIndex=link_index,
                                              targetPosition=(pos[0], pos[1], pos[2]),
                                              targetOrientation=quat.as_vector("xyzw"))
        self.reset_joint_position(joints)

    def set_joint_trajectory(self, final, duration):
        init = self.get_joint_position()
        trajectories = []

        for i in range(self.num_joints):
            trajectories.append(Trajectory([0, duration], [init[i], final[i]]))

        t = 0
        dt = 1/240  # This is the dt of pybullet
        while t < duration:
            command = []
            for i in range(self.num_joints):
              command.append(trajectories[i].pos(t))
            self.set_joint_position(command)
            t += dt
            self.step()


class CameraBullet(Camera):
    def __init__(self, pos, target_pos, up_vector,
                 pinhole_camera_intrinsics, name='sim_camera'):
        self.name = name

        self.pos = np.array(pos)
        self.target_pos = np.array(target_pos)
        self.up_vector = np.array(up_vector)

        # Compute view matrix
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=pos,
                                               cameraTargetPosition=target_pos,
                                               cameraUpVector=up_vector)

        self.z_near = 0.01
        self.z_far = 5.0
        self.width, self.height = pinhole_camera_intrinsics.width, pinhole_camera_intrinsics.height
        self.fx, self.fy = pinhole_camera_intrinsics.fx, pinhole_camera_intrinsics.fy
        self.cx, self.cy = pinhole_camera_intrinsics.cx, pinhole_camera_intrinsics.cy

        # Compute projection matrix
        fov_h = math.atan(self.height / 2 / self.fy) * 2 / math.pi * 180
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=fov_h, aspect=self.width / self.height,
                                                              nearVal=self.z_near, farVal=self.z_far)

    def get_pose(self):
        """
        Returns the camera pose w.r.t. world

        Returns
        -------
        np.array()
            4x4 matrix representing the camera pose w.r.t. world
        """
        return bullet_util.get_camera_pose(self.pos, self.target_pos, self.up_vector)

    def get_depth(self, depth_buffer):
        """
        Converts the depth buffer to depth map.

        Parameters
        ----------
        depth_buffer: np.array()
            The depth buffer as returned from opengl
        """
        depth = self.z_far * self.z_near / (self.z_far - (self.z_far - self.z_near) * depth_buffer)
        return depth

    def get_data(self):
        """
        Returns
        -------
        np.array(), np.array(), np.array()
            The rgb, depth and segmentation images
        """
        image = p.getCameraImage(self.width, self.height,
                                 self.view_matrix, self.projection_matrix,
                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return image[2], self.get_depth(image[3]), image[4]

    def get_intrinsics(self):
        """
        Returns the pinhole camera intrinsics
        """
        return PinholeCameraIntrinsics(width=self.width, height=self.height,
                                       fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)


class BulletEnv(Env):
    """
    Class implementing the clutter env in pyBullet.

    Parameters
    ----------
    name : str
        A string with the name of the environment. 
    params : dict
        A dictionary with parameters for the environment.
    """
    def __init__(self, robot, name='', params={}):
        super().__init__(name, params)
        self.render = params['render']
        if self.render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.params = params.copy()

        self.objects = []


        # Load table
        self.workspace_center_pos = np.array(params["workspace"]["pos"])
        self.workspace_center_quat = Quaternion(w=params["workspace"]["quat"]["w"],
                                                x=params["workspace"]["quat"]["x"],
                                                y=params["workspace"]["quat"]["y"],
                                                z=params["workspace"]["quat"]["z"])

        self.robot = None

        self.scene_generator = SceneGenerator(params['scene_generation'])

        # Set camera params
        pinhole_camera_intrinsics = PinholeCameraIntrinsics.from_params(params['camera']['intrinsics'])
        self.camera = CameraBullet(self.workspace2world(pos=params['camera']['pos'])[0],
                                   self.workspace2world(pos=params['camera']['target_pos'])[0],
                                   self.workspace2world(pos=params['camera']['up_vector'])[0], pinhole_camera_intrinsics)


        if self.render:
            self.button = Button("Pause")
            self.names_button = NamesButton("Show Names")
            self.slider = p.addUserDebugParameter("Delay sim (sec)", 0.0, 0.03, 0.0)
            self.exit_button = NamesButton("Exit")
        self.collision = False

        self.rng = np.random.RandomState()

    def load_robot_and_workspace(self):
        self.objects = []

        p.setAdditionalSearchPath(clt_assets.PATH)  # optionally

        # Load robot
        k = p.loadURDF("ur5e_rs_fingerlong.urdf")


        if self.params["workspace"]["walls"]:
            table_name = "table_walls.urdf"
        else:
            table_name = "table.urdf"

        table_id = p.loadURDF(table_name, basePosition=self.workspace_center_pos,
                              baseOrientation=self.workspace_center_quat.as_vector("xyzw"))

        # Todo: get table size w.r.t. local frame
        table_size = np.abs(np.asarray(p.getAABB(table_id)[1]) - np.asarray(p.getAABB(table_id)[0]))
        self.objects.append(Object(name='table', pos=self.workspace_center_pos,
                                   quat=self.workspace_center_quat.as_vector("xyzw"),
                                   size=(table_size[0], table_size[1]), body_id=table_id))

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", [0, 0, -0.7])
        self.objects.append(Object(name='plane', body_id=plane_id))

        self.robot = UR5Bullet()
        p.setJointMotorControlArray(bodyIndex=0, jointIndices=self.robot.indices, controlMode=p.POSITION_CONTROL)
        self.robot.reset_joint_position(self.robot.joint_configs["home"])

        pos, quat = self.workspace2world(pos=np.zeros(3), quat=Quaternion())
        scale = 0.3
        p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 0], [1, 0, 0])
        p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 1], [0, 1, 0])
        p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 2], [0, 0, 1])

    def workspace2world(self, pos=None, quat=None, inv=False):
        """
        Transforms a pose in workspace coordinates to world coordinates

        Parameters
        ----------
        pos: list
            The position in workspace coordinates

        quat: Quaternion
            The quaternion in workspace coordinates

        Returns
        -------

        list: position in worldcreate_scene coordinates
        Quaternion: quaternion in world coordinates
        """
        world_pos, world_quat = None, None
        tran = Affine3.from_vec_quat(self.workspace_center_pos, self.workspace_center_quat).matrix()

        if inv:
            tran = Affine3.from_matrix(np.linalg.inv(tran)).matrix()

        if pos is not None:
            world_pos = np.matmul(tran, np.append(pos, 1))[:3]
        if quat is not None:
            world_rot = np.matmul(tran[0:3, 0:3], quat.rotation_matrix())
            world_quat = Quaternion.from_rotation_matrix(world_rot)

        return world_pos, world_quat

    def seed(self, seed=None):
        self.scene_generator.seed(seed)
        self.rng.seed(seed)

    def remove_body(self, id):
        p.removeBody(id)
        object = next(x for x in self.objects if x.body_id == id)
        self.objects.remove(object)
        self.scene_generator.objects.remove(object)

    def reset_from_dict(self, saved_dict):
        ids = []
        for obj in self.objects:
            if obj.name != 'plane' and obj.name != 'table':
                ids.append(obj.body_id)

        for id in ids:
            self.remove_body(id)

        objs = []
        for d in saved_dict:
            if d['name'] in ['plane', 'table']:
                continue

            objs.append(Object.from_dict(d))
        self.scene_generator.objects = objs

        for obj in objs:
            if obj.name == 'target':
                color = [1, 0, 0, 1]
            else:
                color = [0, 0, 1, 1]
            col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj.size)
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=obj.size,
                                                  rgbaColor=color)

            base_position, base_orientation = self.workspace2world(pos=obj.pos, quat=obj.quat)
            base_orientation = base_orientation.as_vector("xyzw")
            mass = 1.0
            body_id = p.createMultiBody(mass, col_box_id, visual_shape_id,
                                        base_position, base_orientation)
            obj.body_id = body_id
            self.objects.append(obj)

        return self.get_obs()

    def reset(self):
        self.collision = False
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        self.load_robot_and_workspace()

        hugging = self.rng.random() < self.params['scene_generation']['hug']['probability']
        self.scene_generator.reset()
        self.scene_generator.generate_scene(hugging=hugging)
        for obj in self.scene_generator.objects:
            if obj.name == 'target':
                color = [1, 0, 0, 1]
            else:
                color = [0, 0, 1, 1]
            col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj.size)
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=obj.size,
                                                  rgbaColor=color)

            pos = obj.pos
            if obj.name != 'target':
                pos[2] += 0.03

            base_position, base_orientation = self.workspace2world(pos=obj.pos, quat=obj.quat)
            base_orientation = base_orientation.as_vector("xyzw")
            mass = 1.0
            body_id = p.createMultiBody(mass, col_box_id, visual_shape_id,
                                        base_position, base_orientation)
            obj.body_id = body_id
            self.objects.append(obj)

        t = 0
        while t < 3000:
            p.stepSimulation()
            t += 1

        # Update position and orientation
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

        if hugging:
            self.hug()

        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()

        self.clear_target_occlusion()

        return self.get_obs()

    def objects_still_moving(self):
        for obj in self.objects:
            if obj.name in ['table', 'plane']:
                continue

            vel, rot_vel = p.getBaseVelocity(bodyUniqueId=obj.body_id)
            norm_1 = np.linalg.norm(vel)
            norm_2 = np.linalg.norm(rot_vel)
            if norm_1 > 0.001 or norm_2 > 0.1:
                return True
        return False

    def step(self, action):
        if len(action) > 2:

            self.collision = False
            p1 = action[0]
            p2 = action[-1]
            p1_w, _ = self.workspace2world(p1)
            p2_w, _ = self.workspace2world(p2)

            tmp_1 = p1_w.copy()
            tmp_2 = p2_w.copy()
            tmp_1[2] = 0
            tmp_2[2] = 0
            y_direction = (tmp_2 - tmp_1) / np.linalg.norm(tmp_2 - tmp_1)
            x = np.cross(y_direction, np.array([0, 0, -1]))

            rot_mat = np.array([[x[0], y_direction[0], 0],
                                [x[1], y_direction[1], 0],
                                [x[2], y_direction[2], -1]])

            quat = Quaternion.from_rotation_matrix(rot_mat)

            # Inverse kinematics seems to not accurate when the target position is far from the current,
            # resulting to errors after reset. Call trajectory to compensate for these errors
            self.robot.reset_task_pose(p1_w + np.array([0, 0, 0.05]), quat)
            self.robot_set_task_pose_trajectory(p1_w + np.array([0, 0, 0.05]), quat, 0.2)
            self.robot_set_task_pose_trajectory(p1_w, quat, 0.5, stop_collision=True)

            if not self.collision:
                for i in range(1, len(action)):
                    p_w, _ = self.workspace2world(action[i])
                    self.robot_set_task_pose_trajectory(p_w, quat, None)
                # self.robot_set_task_pose_trajectory(p2_w + np.array([0, 0, 0.05]), quat, 1)

            self.robot.reset_joint_position(self.robot.joint_configs["home"])

            while self.objects_still_moving():
                time.sleep(0.001)
                self.sim_step()

            return self.get_obs()
        else:
            return self.step_linear(action)

    def step_linear(self, action):
        """
        Moves the environment one step forward in time.

        Parameters
        ----------

        action : tuple
            A tuple of two 3D np.arrays corresponding to the initial and final 3D point of the push with respect to
            inertia frame (workspace frame)

        Returns
        -------
        dict :
            A dictionary with the following keys: rgb, depth, seg, full_state. See get_obs() for more info.
        """
        self.collision = False
        p1 = action[0]
        p2 = action[1]
        p1_w, _ = self.workspace2world(p1)
        p2_w, _ = self.workspace2world(p2)

        y_direction = (p2_w - p1_w) / np.linalg.norm(p2_w - p1_w)
        x = np.cross(y_direction, np.array([0, 0, -1]))

        rot_mat = np.array([[x[0], y_direction[0], 0],
                            [x[1], y_direction[1], 0],
                            [x[2], y_direction[2], -1]])

        quat = Quaternion.from_rotation_matrix(rot_mat)

        # Inverse kinematics seems to not accurate when the target position is far from the current,
        # resulting to errors after reset. Call trajectory to compensate for these errors
        self.robot.reset_task_pose(p1_w + np.array([0, 0, 0.05]), quat)
        self.robot_set_task_pose_trajectory(p1_w + np.array([0, 0, 0.05]), quat, 0.2)
        self.robot_set_task_pose_trajectory(p1_w, quat, 0.5, stop_collision=True)

        if not self.collision:
            self.robot_set_task_pose_trajectory(p2_w, quat, None)
            # self.robot_set_task_pose_trajectory(p2_w + np.array([0, 0, 0.05]), quat, 1)

        self.robot.reset_joint_position(self.robot.joint_configs["home"])

        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()

        return self.get_obs()

    def get_obs(self):
        # Update visual observation
        rgb, depth, seg = self.camera.get_data()

        # Update position and orientation
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

        table = next(x for x in self.objects if x.name == 'table')
        full_state = {'objects': self.objects,
                      'finger': self.robot.finger,
                      'surface': [table.size[0], table.size[1]]}

        import copy
        return {'rgb': rgb, 'depth': depth, 'seg': seg, 'full_state': copy.deepcopy(full_state),
                'collision': self.collision}

    def clear_target_occlusion(self, eps=0.003):
        """
        Checks if an obstacle is above the target object and occludes it. Then
        it removes it from the arena.
        """
        target = next(x for x in self.objects if x.name == 'target')

        ids = []
        for obj in self.objects:
            if obj.name in ['table', 'plane', 'target']:
                continue

            if obj.pos[2] < 0:
                continue

            if target.is_above_me(obj) or obj.distance_from_plane([0, 0, 0], [0, 0, 1]) > 0.003:
                ids.append(obj.body_id)

        for id in ids:
            self.remove_body(id)

        p.stepSimulation()

    def hug(self):
        target = next(x for x in self.objects if x.name == 'target')
        ids = []
        force_magnitude = 20
        duration = 300
        t = 0
        while t < duration:
            for obj in self.objects:
                if obj.name in ['table', 'plane', 'target']:
                    continue

                if obj.pos[2] < 0:
                    continue

                pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
                error = self.workspace2world(target.pos)[0] - pos
                if np.linalg.norm(error) < self.params['scene_generation']['hug']['radius']:
                    force_direction = error / np.linalg.norm(error)
                    p.applyExternalForce(obj.body_id, -1, force_magnitude * force_direction, pos, p.WORLD_FRAME)

            p.stepSimulation()
            t += 1

        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

    def robot_set_task_pose_trajectory(self, pos, quat, duration, stop_collision=False):
        init_pos, init_quat = self.robot.get_task_pose()
        # Calculate duration adaptively if its None
        if duration is None:
            vel = 0.3
            duration = np.linalg.norm(init_pos - pos) / vel
        trajectories = []
        for i in range(3):
            trajectories.append(Trajectory([0, duration], [init_pos[i], pos[i]]))

        t = 0
        dt = 1/240  # This is the dt of pybullet
        while t < duration:
            command = []
            for i in range(3):
                command.append(trajectories[i].pos(t))
            self.robot.set_task_pose(command, init_quat)
            t += dt
            contact = self.sim_step()

            if stop_collision and contact:
                self.collision = True
                break

    def sim_step(self):
        if self.render:
            if self.exit_button.on():
                exit()

            while self.button.on():
                time.sleep(0.001)

            time.sleep(p.readUserDebugParameter(self.slider))
        p.stepSimulation()

        if self.render:
            self.names_button.show_names(self.objects)

        link_index = bullet_util.get_link_indices(['finger_body'])[0]

        contact = False
        for obj in self.objects:
            if obj.name == 'table' or obj.name == 'plane':
                continue

            contacts = p.getContactPoints(0, obj.body_id, link_index, -1)
            valid_contacts = []
            for c in contacts:
                normal_vector = c[7]
                normal_force = c[9]
                if np.dot(normal_vector, np.array([0, 0, 1])) > 0.9:
                    valid_contacts.append(c)

            if len(valid_contacts) > 0:
                contact = True
                break

        return contact

class RearragementSceneGenerator:
    def __init__(self, params):
        self.params = params
        self.finger_size = 0.001
        self.surface_size = [0.25, 0.25]
        self.rng = np.random.RandomState()
        self.objects = []

    def reset(self):
        self.objects = []

    def seed(self, seed):
        self.rng.seed(seed)

    def generate_scene(self):
        assert len(self.params['n_objects']['min']) == len(self.params['n_objects']['max'])
        self.n_obstacles = []
        for i in range(len(self.params['n_objects']['min'])):
            self.n_obstacles.append(self.rng.randint(self.params['n_objects']['min'][i],
                                                     self.params['n_objects']['max'][i]))

        colors = [[1, 0, 0, 1],
                  [0, 0, 1, 1],
                  [0, 1, 0, 1],
                  [1, 1, 0, 1],
                  [1, 0, 1, 1],
                  [0, 1, 1, 1]]

        for i in range(len(self.n_obstacles)):
            for j in range(self.n_obstacles[i]):
                #   Randomize size
                object_length = self.rng.uniform(self.params['object']['min_bounding_box'][0],
                                                 self.params['object']['max_bounding_box'][0])
                object_width  = self.rng.uniform(self.params['object']['min_bounding_box'][1],
                                                 min(object_length, self.params['object']['max_bounding_box'][1]))
                object_height = self.rng.uniform(self.params['object']['min_bounding_box'][2],
                                                   self.params['object']['max_bounding_box'][2])
                pos = np.array([self.rng.uniform(- 0.90 * self.surface_size[0], 0.90 * self.surface_size[0]),
                                self.rng.uniform(- 0.90 * self.surface_size[1], 0.90 * self.surface_size[1]),
                                object_height])

                theta = self.rng.uniform(0, 2 * math.pi)
                quat = Quaternion()
                quat.rot_z(theta)

                self.objects.append(Object(name='obj_' + str(i) + '_' + str(j), pos=pos, quat=quat,
                                           size=[object_length, object_width, object_height], class_=i, color=colors[i]))

    def plot(self, show=False):
        pose_scale = 0.01
        ax = self.objects[0].plot(pose_scale=pose_scale, color=[1, 0, 0, 1])
        for i in range(1, len(self.objects)):
            ax = self.objects[i].plot(pose_scale=pose_scale, color=[0, 0, 1, 1], ax=ax)
        if show:
            plt.show()
        return ax





class RearrangementEnv(Env):
    """
    Class implementing the clutter env in pyBullet.

    Parameters
    ----------
    name : str
        A string with the name of the environment.
    params : dict
        A dictionary with parameters for the environment.
    """
    def __init__(self, robot=UR5Bullet, name='', params={}):
        super().__init__(name, params)
        self.render = params['render']
        if self.render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(clt_assets.PATH)  # optionally

        # Load robot
        p.loadURDF("ur5e_rs_fingerlong.urdf")
        self.robot = robot()

        self.objects = []

        # Load table
        if params["workspace"]["walls"]:
            table_name = "table_walls.urdf"
        else:
            table_name = "table.urdf"
        self.workspace_center_pos = np.array(params["workspace"]["pos"])
        self.workspace_center_quat = Quaternion(w=params["workspace"]["quat"]["w"],
                                                x=params["workspace"]["quat"]["x"],
                                                y=params["workspace"]["quat"]["y"],
                                                z=params["workspace"]["quat"]["z"])
        table_id = p.loadURDF(table_name, basePosition=self.workspace_center_pos,
                              baseOrientation=self.workspace_center_quat.as_vector("xyzw"))

        # Todo: get table size w.r.t. local frame
        table_size = np.abs(np.asarray(p.getAABB(table_id)[1]) - np.asarray(p.getAABB(table_id)[0]))
        self.objects.append(Object(name='table', pos=self.workspace_center_pos,
                                   quat=self.workspace_center_quat.as_vector("xyzw"),
                                   size=(table_size[0], table_size[1]), body_id=table_id))

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", [0, 0, -0.7])
        self.objects.append(Object(name='plane', body_id=plane_id))

        self.scene_generator = RearragementSceneGenerator(params['scene_generation'])

        # Set camera params
        pinhole_camera_intrinsics = PinholeCameraIntrinsics.from_params(params['camera']['intrinsics'])
        self.camera = CameraBullet(self.workspace2world(pos=params['camera']['pos'])[0],
                                   self.workspace2world(pos=params['camera']['target_pos'])[0],
                                   self.workspace2world(pos=params['camera']['up_vector'])[0], pinhole_camera_intrinsics)

        pos, quat = self.workspace2world(pos=np.zeros(3), quat=Quaternion())
        scale = 0.3
        p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 0], [1, 0, 0])
        p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 1], [0, 1, 0])
        p.addUserDebugLine(pos, pos + scale * quat.rotation_matrix()[:, 2], [0, 0, 1])

        if self.render:
            self.button = Button("Pause")
            self.names_button = NamesButton("Show Names")
            self.slider = p.addUserDebugParameter("Delay sim (sec)", 0.0, 0.03, 0.0)
            self.exit_button = NamesButton("Exit")
        self.collision = False

        self.rng = np.random.RandomState()

    def workspace2world(self, pos=None, quat=None, inv=False):
        """
        Transforms a pose in workspace coordinates to world coordinates

        Parameters
        ----------
        pos: list
            The position in workspace coordinates

        quat: Quaternion
            The quaternion in workspace coordinates

        Returns
        -------

        list: position in worldcreate_scene coordinates
        Quaternion: quaternion in world coordinates
        """
        world_pos, world_quat = None, None
        tran = Affine3.from_vec_quat(self.workspace_center_pos, self.workspace_center_quat).matrix()

        if inv:
            tran = Affine3.from_matrix(np.linalg.inv(tran)).matrix()

        if pos is not None:
            world_pos = np.matmul(tran, np.append(pos, 1))[:3]
        if quat is not None:
            world_rot = np.matmul(tran[0:3, 0:3], quat.rotation_matrix())
            world_quat = Quaternion.from_rotation_matrix(world_rot)

        return world_pos, world_quat

    def seed(self, seed=None):
        self.scene_generator.seed(seed)
        self.rng.seed(seed)

    def remove_body(self, id):
        p.removeBody(id)
        object = next(x for x in self.objects if x.body_id == id)
        self.objects.remove(object)
        self.scene_generator.objects.remove(object)

    def reset_from_dict(self, saved_dict):
        ids = []
        for obj in self.objects:
            if obj.name != 'plane' and obj.name != 'table':
                ids.append(obj.body_id)

        for id in ids:
            self.remove_body(id)

        objs = []
        for d in saved_dict:
            if d['name'] in ['plane', 'table']:
                continue

            objs.append(Object.from_dict(d))
        self.scene_generator.objects = objs

        for obj in objs:
            if obj.name == 'target':
                color = [1, 0, 0, 1]
            else:
                color = [0, 0, 1, 1]
            col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj.size)
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=obj.size,
                                                  rgbaColor=color)

            base_position, base_orientation = self.workspace2world(pos=obj.pos, quat=obj.quat)
            base_orientation = base_orientation.as_vector("xyzw")
            mass = 1.0
            body_id = p.createMultiBody(mass, col_box_id, visual_shape_id,
                                        base_position, base_orientation)
            obj.body_id = body_id
            self.objects.append(obj)

        return self.get_obs()

    def reset(self):
        self.collision = False
        ids = []
        for obj in self.objects:
            if obj.name != 'plane' and obj.name != 'table':
                ids.append(obj.body_id)

        for id in ids:
            self.remove_body(id)

        hugging = self.rng.random() < self.params['scene_generation']['hug']['probability']
        self.scene_generator.reset()
        self.scene_generator.generate_scene()
        for obj in self.scene_generator.objects:
            col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj.size)
            visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=obj.size,
                                                  rgbaColor=obj.color)

            base_position, base_orientation = self.workspace2world(pos=obj.pos, quat=obj.quat)
            base_orientation = base_orientation.as_vector("xyzw")
            mass = 1.0
            body_id = p.createMultiBody(mass, col_box_id, visual_shape_id,
                                        base_position, base_orientation)
            obj.body_id = body_id
            self.objects.append(obj)

        t = 0
        while t < 3000:
            p.stepSimulation()
            t += 1

        # Update position and orientation
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

        if hugging:
            self.hug()

        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()

        # self.clear_target_occlusion()

        return self.get_obs()

    def objects_still_moving(self):
        for obj in self.objects:
            if obj.name in ['table', 'plane']:
                continue

            vel, rot_vel = p.getBaseVelocity(bodyUniqueId=obj.body_id)
            norm_1 = np.linalg.norm(vel)
            norm_2 = np.linalg.norm(rot_vel)
            if norm_1 > 0.001 or norm_2 > 0.1:
                return True
        return False

    def step(self, action):
        """
        Moves the environment one step forward in time.

        Parameters
        ----------

        action : tuple
            A tuple of two 3D np.arrays corresponding to the initial and final 3D point of the push with respect to
            inertia frame (workspace frame)

        Returns
        -------
        dict :
            A dictionary with the following keys: rgb, depth, seg, full_state. See get_obs() for more info.
        """
        self.collision = False
        p1 = action[0]
        p2 = action[1]
        p1_w, _ = self.workspace2world(p1)
        p2_w, _ = self.workspace2world(p2)

        y_direction = (p2_w - p1_w) / np.linalg.norm(p2_w - p1_w)
        x = np.cross(y_direction, np.array([0, 0, -1]))

        rot_mat = np.array([[x[0], y_direction[0], 0],
                            [x[1], y_direction[1], 0],
                            [x[2], y_direction[2], -1]])

        quat = Quaternion.from_rotation_matrix(rot_mat)

        # Inverse kinematics seems to not accurate when the target position is far from the current,
        # resulting to errors after reset. Call trajectory to compensate for these errors
        self.robot.reset_task_pose(p1_w + np.array([0, 0, 0.05]), quat)
        self.robot_set_task_pose_trajectory(p1_w + np.array([0, 0, 0.05]), quat, 0.2)
        self.robot_set_task_pose_trajectory(p1_w, quat, 0.5, stop_collision=True)

        if not self.collision:
            self.robot_set_task_pose_trajectory(p2_w, quat, None)
            # self.robot_set_task_pose_trajectory(p2_w + np.array([0, 0, 0.05]), quat, 1)

        self.robot.reset_joint_position(self.robot.joint_configs["home"])

        while self.objects_still_moving():
            time.sleep(0.001)
            self.sim_step()

        return self.get_obs()

    def get_obs(self):
        # Update visual observation
        rgb, depth, seg = self.camera.get_data()

        # Update position and orientation
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

        table = next(x for x in self.objects if x.name == 'table')
        full_state = {'objects': self.objects,
                      'finger': self.robot.finger,
                      'surface': [table.size[0], table.size[1]]}

        import copy
        return {'rgb': rgb, 'depth': depth, 'seg': seg, 'full_state': copy.deepcopy(full_state),
                'collision': self.collision}

    def clear_target_occlusion(self, eps=0.003):
        """
        Checks if an obstacle is above the target object and occludes it. Then
        it removes it from the arena.
        """
        target = next(x for x in self.objects if x.name == 'target')

        ids = []
        for obj in self.objects:
            if obj.name in ['table', 'plane', 'target']:
                continue

            if obj.pos[2] < 0:
                continue

            if target.is_above_me(obj) or obj.distance_from_plane([0, 0, 0], [0, 0, 1]) > 0.003:
                ids.append(obj.body_id)

        for id in ids:
            self.remove_body(id)

        p.stepSimulation()

    def hug(self):
        target = next(x for x in self.objects if x.name == 'target')
        ids = []
        force_magnitude = 20
        duration = 300
        t = 0
        while t < duration:
            for obj in self.objects:
                if obj.name in ['table', 'plane', 'target']:
                    continue

                if obj.pos[2] < 0:
                    continue

                pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
                error = self.workspace2world(target.pos)[0] - pos
                if np.linalg.norm(error) < self.params['scene_generation']['hug']['radius']:
                    force_direction = error / np.linalg.norm(error)
                    p.applyExternalForce(obj.body_id, -1, force_magnitude * force_direction, pos, p.WORLD_FRAME)

            p.stepSimulation()
            t += 1

        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

    def robot_set_task_pose_trajectory(self, pos, quat, duration, stop_collision=False):
        init_pos, init_quat = self.robot.get_task_pose()
        # Calculate duration adaptively if its None
        if duration is None:
            vel = 0.3
            duration = np.linalg.norm(init_pos - pos) / vel
        trajectories = []
        for i in range(3):
            trajectories.append(Trajectory([0, duration], [init_pos[i], pos[i]]))

        t = 0
        dt = 1/240  # This is the dt of pybullet
        while t < duration:
            command = []
            for i in range(3):
                command.append(trajectories[i].pos(t))
            self.robot.set_task_pose(command, init_quat)
            t += dt
            contact = self.sim_step()

            if stop_collision and contact:
                self.collision = True
                break

    def sim_step(self):
        if self.render:
            if self.exit_button.on():
                exit()

            while self.button.on():
                time.sleep(0.001)

            time.sleep(p.readUserDebugParameter(self.slider))
        p.stepSimulation()

        if self.render:
            self.names_button.show_names(self.objects)

        link_index = bullet_util.get_link_indices(['finger_body'])[0]

        contact = False
        for obj in self.objects:
            if obj.name == 'table' or obj.name == 'plane':
                continue

            contacts = p.getContactPoints(0, obj.body_id, link_index, -1)
            valid_contacts = []
            for c in contacts:
                normal_vector = c[7]
                normal_force = c[9]
                if np.dot(normal_vector, np.array([0, 0, 1])) > 0.9:
                    valid_contacts.append(c)

            if len(valid_contacts) > 0:
                contact = True
                break

        return contact


def get_distances_from_target(obs):
    objects = obs['full_state']['objects']

    # Get target pose from full state
    target = next(x for x in objects if x.name == 'target')
    target_pose = np.eye(4)
    target_pose[0:3, 0:3] = target.quat.rotation_matrix()
    target_pose[0:3, 3] = target.pos

    # Compute the distances of the obstacles from the target
    distances_from_target = []
    for obj in objects:
        if obj.name in ['target', 'table', 'plane'] or obj.pos[2] < 0:
            continue

        # Transform the objects w.r.t. target (reduce variability)
        obj_pose = np.eye(4)
        obj_pose[0:3, 0:3] = obj.quat.rotation_matrix()
        obj_pose[0:3, 3] = obj.pos

        distance = get_distance_of_two_bbox(target_pose, target.size, obj_pose, obj.size)
        distances_from_target.append(distance)
    return np.array(distances_from_target)


def empty_push(obs, next_obs, eps=0.005):
    """
    Checks if the objects have been moved
    """

    for prev_obj in obs['full_state']['objects']:
        if prev_obj.name in ['table', 'plane']:
            continue

        for obj in next_obs['full_state']['objects']:
            if prev_obj.body_id == obj.body_id:
                if np.linalg.norm(prev_obj.pos - obj.pos) > eps:
                    return False
    return True


def compute_free_space_map(push_target_map):
    # Compute contours
    ret, thresh = cv2.threshold(push_target_map, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Compute minimum distance for each point from contours
    contour_point = []
    for contour in contours:
        for pt in contour:
            contour_point.append(pt)

    cc = np.zeros((push_target_map.shape[0], push_target_map.shape[1], len(contour_point), 2))
    for i in range(len(contour_point)):
        cc[:, :, i, :] = np.squeeze(contour_point[i])

    ids = np.zeros((push_target_map.shape[0], push_target_map.shape[1], len(contour_point), 2))
    for i in range(push_target_map.shape[0]):
        for j in range(push_target_map.shape[1]):
            ids[i, j, :, :] = np.array([j, i])

    dist = np.min(np.linalg.norm(ids - cc, axis=3), axis=2)

    # Compute min distance for each point from table limits
    dists_surface = np.zeros((push_target_map.shape[0], push_target_map.shape[1]))
    for i in range(push_target_map.shape[0]):
        for j in range(push_target_map.shape[1]):
            dists_surface[i, j] = np.min(np.array([i, push_target_map.shape[0] - i, j, push_target_map.shape[1] - j]))

    map = np.minimum(dist, dists_surface)
    min_value = np.min(map)
    map[push_target_map > 0] = min_value
    map = min_max_scale(map, range=[np.min(map), np.max(map)], target_range=[0, 1])
    return map

def get_heightmap(obs):
    """
    Computes the heightmap based on the 'depth' and 'seg'. In this heightmap table pixels has value zero,
    objects > 0 and everything below the table <0.

    Parameters
    ----------
    obs : dict
        The dictionary with the visual and full state of the environment.

    Returns
    -------
    np.ndarray :
        Array with the heightmap.
    """
    rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
    objects = obs['full_state']['objects']

    # Compute heightmap
    table_id = next(x.body_id for x in objects if x.name == 'table')
    depthcopy = depth.copy()
    table_depth = np.max(depth[seg == table_id])
    depthcopy[seg == table_id] = table_depth
    heightmap = table_depth - depthcopy
    return heightmap


def unified_map(obs, crop=(192, 192), n_classes=5):
    rgb, depth, seg = obs['rgb'], obs['depth'], obs['seg']
    workspace_crop_area = crop

    heightmap = get_heightmap(obs)
    heightmap[heightmap < 0] = 0
    heightmap = Feature(heightmap).crop(workspace_crop_area[0], workspace_crop_area[1]).array()

    target_id = next(x.body_id for x in obs['full_state']['objects'] if x.name == 'target')
    mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
    mask[seg == target_id] = 255
    mask = Feature(mask).crop(workspace_crop_area[0], workspace_crop_area[1]).array()

    target_height = np.max(heightmap[mask == 255])
    eps = 0.005

    # print('target ', target_height)
    # import matplotlib.pyplot as plt
    # plt.imshow(heightmap)
    # plt.show()

    fused_map = np.zeros(heightmap.shape).astype(np.int32)
    if n_classes == 5:
        fused_map[heightmap > 0] = 128 # 4 - flat objects
        fused_map[heightmap > target_height - eps] = 192 # 3 - equal to target
        fused_map[heightmap > target_height + eps] = 255 # 2 - tall objects
        fused_map[mask > 0] = 64 # 1 - target
    elif n_classes == 3:
        fused_map[heightmap > 0] = 255  # 4 - flat objects
        fused_map[mask > 0] = 128  # 1 - target
    else:
        raise AttributeError('n_classes should be 5 or 3')

    return fused_map, heightmap
