"""Hand tracking from a web camera and retargeting to Psyonic hand.

Retargetting roughly follows the idea from
Handa, Ankur, et al. Dexpilot: Vision-based teleoperation of dexterous
robotic hand-arm system. 2020 IEEE International Conference on Robotics
and Automation (ICRA). IEEE, 2020.
"""

import os

import kinpy as kp
import nlopt
import numpy as np
import open3d as o3d
import pygame
import pybullet as p
from transforms3d.axangles import axangle2mat

import config
from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from omniisaacgymenvs.data_types import geometry_utils
from omniisaacgymenvs.data_types import se3
from utils import OneEuroFilter, imresize
from utils import *
from wrappers import ModelPipeline


def live_application(capture):
  """
  Launch an application that reads from a webcam and estimates hand pose at
  real-time.

  The captured hand must be the right hand, but will be flipped internally
  and rendered.

  Parameters
  ----------
  capture : object
    An object from `capture.py` to read capture stream from.
  """

  # Defines the constants for hand retargeting
  left_right = 'left'
  urdf_path = ('./urdf/psyonic/ability_hand_' +
               left_right + '.urdf')

  # The unit length in meters to normalize the absolute tracked
  # finger joint positions
  unit_length = 0.09473151311686484

  # Kinematics of the hand
  chain = kp.build_chain_from_urdf(
    open(os.path.expanduser(urdf_path), 'rb').read())
  joint_indices = {
    name: index for index, name in enumerate(
    chain.get_joint_parameter_names())}

  # Each fingertip is a tip for the kinematic chain it is associated with
  finger_tips = [
    'thumb_anchor_frame',
    'index_anchor_frame',
    'middle_anchor_frame',
    'ring_anchor_frame',
    'pinky_anchor_frame'
  ]
  # Each joint is on a different branch of the kinematics tree
  tip_joints = {
    'thumb_anchor_frame': ['thumb_q1', 'thumb_q2'],
    'index_anchor_frame': ['index_q1', 'index_q2'],
    'middle_anchor_frame': ['middle_q1', 'middle_q2'],
    'ring_anchor_frame': ['ring_q1', 'ring_q2'],
    'pinky_anchor_frame': ['pinky_q1', 'pinky_q2']
  }
  # Buids separate kinematics chain for each joint
  finger_chains = [kp.chain.SerialChain(
    chain, tip_frame, 'thumb_base_frame') for tip_frame in finger_tips]

  # Sets the mimic joints as the robot hand is underactuated
  mimic_config = {
    'index_q2': ('index_q1', 1.05851325, 0.72349796),
    'middle_q2': ('middle_q1', 1.05851325, 0.72349796),
    'ring_q2': ('ring_q1', 1.05851325, 0.72349796),
    'pinky_q2': ('pinky_q1', 1.05851325, 0.72349796),
  }
  mimic_joints = {}
  for name, source in mimic_config.items():
    mimic_joints[joint_indices[name]] = (
      joint_indices[source[0]], source[1], source[2])

  # The mapping matrix from finger tip to pairwise vectors
  # The columns from left to right correspond to: thumb, index, middle,
  # ring, and pinky.
  d_vector_d_tip_positions = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [-1, 1, 0, 0, 0],
    [-1, 0, 1, 0, 0],
    [-1, 0, 0, 1, 0],
    [-1, 0, 0, 0, 1],
    [0, -1, 1, 0, 0],
    [0, -1, 0, 1, 0],
    [0, -1, 0, 0, 1],
    [0, 0, -1, 1, 0],
    [0, 0, -1, 0, 1],
    [0, 0, 0, -1, 1]]
  )

  def vector_length(vector):
    return np.linalg.norm(vector)

  def normalize(vector):
    return vector / np.clip(vector_length(vector), 1e-10, 1e10)
    
  def human_points_to_vectors(points):
    """Maps from human finger tip positions to the pairwise vectors.

    Returns:
      np.array of 15x3.
    
    The human points are of 21x3, following MPIIHandJoints convention.

    labels = [
      'W', #0
      'T0', 'T1', 'T2', 'T3', #4
      'I0', 'I1', 'I2', 'I3', #8
      'M0', 'M1', 'M2', 'M3', #12
      'R0', 'R1', 'R2', 'R3', #16
      'L0', 'L1', 'L2', 'L3', #20
    ]
    """
    vectors = [
      points[4, :] - points[0, :],
      points[8, :] - points[0, :],
      points[12, :] - points[0, :],
      points[16, :] - points[0, :],
      points[20, :] - points[0, :],
      points[8, :] - points[4, :],
      points[12, :] - points[4, :],
      points[16, :] - points[4, :],
      points[20, :] - points[4, :],
      points[12, :] - points[8, :],
      points[16, :] - points[8, :],
      points[20, :] - points[8, :],
      points[16, :] - points[12, :],
      points[20, :] - points[12, :],
      points[20, :] - points[16, :],
    ]
    return vectors

  def robot_config_to_vectors(joint_positions):
    """Computes the pairwise vector from robot joint positions.

    Returns:
      np.array of 15x3.
    """
    tip_positions = []
    for index, finger_tip_frame in enumerate(finger_tips):
      finger_joints = [joint_positions[joint_indices[joint_name]]
                      for joint_name in tip_joints[finger_tip_frame]]
      tip_positions.append(finger_chains[index].forward_kinematics(
        finger_joints).pos / unit_length)

    return np.dot(d_vector_d_tip_positions, np.array(tip_positions))

  def augment_reduced_joints(reduced_joint_positions):
    """Augments to full hand DOFs from actually controlled DOFs.

    The full hand DOFs is 16 (6 for xyz+rpy for hand base frame, 10 for
    2 DOFs for each of 5 fingers.)
    """
    return np.array([0, 0, 0, 0, 0, 0,
                    reduced_joint_positions[0],
                    reduced_joint_positions[0] * 1.05851325 + 0.72349796,
                    reduced_joint_positions[1],
                    reduced_joint_positions[1] * 1.05851325 + 0.72349796,
                    reduced_joint_positions[2],
                    reduced_joint_positions[2] * 1.05851325 + 0.72349796,
                    reduced_joint_positions[3],
                    reduced_joint_positions[3] * 1.05851325 + 0.72349796,
                    reduced_joint_positions[4],
                    reduced_joint_positions[5]])

  def update_jacobian_for_mimic_joints(jacobian, mimic_joints):
    """Updates the jacobian to consider mimic joints.
    """
    for joint_index, source in mimic_joints.items():
      source_index, multiplier, _ = source
      jacobian[:, source_index] = (
        jacobian[:, joint_index] * multiplier + jacobian[:, source_index])
      jacobian[:, joint_index] = 0
    return jacobian

  def compute_jacobians(joint_positions):
    """Computes the jacobian for all fingertips.

    Returns:
      [np.array of 6x16]
    """
    jacobians = []
    for index, finger_tip_frame in enumerate(finger_tips):
      finger_joint_indices = [joint_indices[joint_name]
                              for joint_name in tip_joints[finger_tip_frame]]
      finger_joints = joint_positions[finger_joint_indices]
      jac = np.zeros((6, len(joint_positions)))
      jac[:, finger_joint_indices] = finger_chains[index].jacobian(
        finger_joints)
      update_jacobian_for_mimic_joints(jac, mimic_joints)
      jacobians.append(jac)
    return jacobians

  # Starts PyBullet GUI to visualize the retarged robot hand
  p.connect(p.GUI)
  p.setAdditionalSearchPath(
    os.path.expanduser('./urdf/psyonic'))
  p.setGravity(0,0,-10)
  robot_id = p.loadURDF(
    os.path.expanduser('./urdf/psyonic/ability_hand_' +
                       left_right + '.urdf'),
    [0,0,0], p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)
  p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=200.8,
                               cameraPitch=-57.4,
                               cameraTargetPosition=[0.09, -0.28, -0.54])

  # Gets the hand base frame in PyBullet
  xyzw = p.getBasePositionAndOrientation(robot_id)[1]
  world_t_base = se3.Transform(xyz=p.getBasePositionAndOrientation(
    robot_id)[0], rot=[xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

  # Gets the thumb base frame in PyBullet
  idx = 6
  xyzw = p.getLinkState(robot_id, idx)[5]
  base_t_thumb_base = world_t_base.inverse() * se3.Transform(
    xyz=p.getLinkState(robot_id, idx)[4],
    rot=[xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

  # Hand tracking visualization setup.
  # align different coordinate systems
  view_mat = axangle2mat([1, 0, 0], np.pi)
  window_size = 1080

  hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
  mesh.vertices = o3d.utility.Vector3dVector(
    np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
  mesh.compute_vertex_normals()

  viewer = o3d.visualization.Visualizer()
  viewer.create_window(
    width=window_size + 1, height=window_size + 1,
    window_name='Minimal Hand - output'
  )
  viewer.add_geometry(mesh)

  view_control = viewer.get_view_control()
  cam_params = view_control.convert_to_pinhole_camera_parameters()
  extrinsic = cam_params.extrinsic.copy()
  extrinsic[0:3, 3] = 0
  cam_params.extrinsic = extrinsic
  cam_params.intrinsic.set_intrinsics(
    window_size + 1, window_size + 1, config.CAM_FX, config.CAM_FY,
    window_size // 2, window_size // 2
  )
  view_control.convert_from_pinhole_camera_parameters(cam_params)
  view_control.set_constant_z_far(1000)

  render_option = viewer.get_render_option()
  render_option.load_from_json('./render_option.json')
  viewer.update_renderer()

  # Input video stream visualization setup.
  pygame.init()
  display = pygame.display.set_mode((window_size, window_size))
  pygame.display.set_caption('Minimal Hand - input')
  mesh_smoother = OneEuroFilter(4.0, 0.0)
  clock = pygame.time.Clock()

  # Hand tracking model initialization.
  model = ModelPipeline()

  # Initial retargeted finger joint configuration.
  # [index, middle, ring, pinky, thumb_q1, thumb_q2]
  theta = np.array([0.5, 0.5, 0.5, 0.5, -0.5, 0.5])

  # Sets up the optimizer by setting an algorithm and dimension.
  opt = nlopt.opt(nlopt.GN_DIRECT, 6)
  opt.set_lower_bounds(
    [0, 0, 0, 0, -2.0943951, 0])
  opt.set_upper_bounds(
    [2.0943951, 2.0943951, 2.0943951, 2.0943951, 0, 2.0943951])
  opt.set_ftol_rel(1e-3)
  opt.set_maxtime(1000)

  while True:
    frame_large = capture.read()
    if frame_large is None:
      continue
    if frame_large.shape[0] > frame_large.shape[1]:
      margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
      frame_large = frame_large[margin:-margin]
    else:
      margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
      frame_large = frame_large[:, margin:-margin]

    frame_large = np.flip(frame_large, axis=1).copy()
    frame = imresize(frame_large, (128, 128))

    # Tracks the hand
    loc_mpii, theta_mpii = model.process(frame)

    human_points_relative = np.array(loc_mpii)
    human_points_relative = (
      human_points_relative - human_points_relative[0, :])
    thumb_base_t_wrist = se3.Transform(
      xyz=[24.0476665e-3, 3.78124745e-3, 32.32964923e-3]).inverse()

    # Defines a hardcoded mapping between human and robot hand bases.
    # This should be inferred by tracking the human hand base frame.
    robot_t_human_base = se3.Transform(
      rot=np.radians([0, 0, 180])) * se3.Transform(
      xyz=(world_t_base * base_t_thumb_base * thumb_base_t_wrist).translation,
      rot=np.radians([-115, 0, 0])
    )
    # adds a scaling of 0.7 to incorporte the size difference between
    # human and robot hands.
    human_points = 0.7 * geometry_utils.transform_points(
      human_points_relative.transpose(), robot_t_human_base).transpose()
    human_vectors = human_points_to_vectors(human_points)

    last_theta = theta.copy()

    def weight_and_scale(vector, vector_index):
      # Defines the weight and scale parameters in the cost function.
      vector_norm = vector_length(human_vectors[vector_index])
      antipodal_set = set([5, 6, 7, 8])
      primary_set = set([9, 10, 11, 12, 13, 14])
      if vector_index in antipodal_set:
        return 5, vector_norm
      elif vector_index in primary_set:
        return 10, vector_norm
      else:
        return 1, vector_norm

    def metric(reduced_joint_positions, gradient):
      # Defines the cost and gradient functions.
      cost = 0
      num_dof = 16
      joint_positions = augment_reduced_joints(reduced_joint_positions)
      vectors = -np.array(robot_config_to_vectors(joint_positions))
      new_grad = np.zeros((num_dof, ))
      jacobians = compute_jacobians(joint_positions)
      for vector_index, vector in enumerate(vectors):
        weight, scale = weight_and_scale(vector, vector_index)
        diff = vector - scale * normalize(human_vectors[vector_index])
        cost += weight * np.linalg.norm(diff) ** 2
        d_vector_d_joint_position = np.zeros((3, num_dof))
        for tip_index in range(d_vector_d_tip_positions.shape[1]):
          d_vector_d_joint_position += d_vector_d_tip_positions[
            vector_index, tip_index] * jacobians[tip_index][:3, :]
        new_grad += weight * np.dot(diff, d_vector_d_joint_position)
      cost = (0.5 * cost + 2.5e-3 * np.linalg.norm(joint_positions) ** 2 +
              vector_length(reduced_joint_positions - last_theta))
      new_grad += 5e-3 * joint_positions
      if gradient.size > 0:
        gradient[:] = new_grad[[6, 8, 10, 12, 14, 15]]

      return cost

    # Solves the optimization problem to find the robot hand joint
    # configuration closest to the current human hand joint configuration.
    opt.set_min_objective(metric)
    theta = opt.optimize(np.array([1.0, 1.0, 1.0, 1.0, -1.0, 1.0]))

    # Visualizes the robot hand configuration in PyBullet.
    jps = np.zeros((22))
    jps[[7, 10, 13, 16, 19, 20]] = theta[[0, 1, 2, 3, 4, 5]]
    jps[[8, 11, 14, 17]] = jps[[7, 10, 13, 16]] * 1.05851325 + 0.72349796
    for i in range(22):
      p.resetJointState(robot_id, i, jps[i])

    # Visualizes the tracked human hand configuration.
    theta_mano = mpii_to_mano(theta_mpii)
    v = hand_mesh.set_abs_quat(theta_mano)
    v *= 2 # for better visualization
    v = v * 1000 + np.array([0, 0, 400])
    v = mesh_smoother.process(v)
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
    mesh.paint_uniform_color(config.HAND_COLOR)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    viewer.update_geometry(mesh)
    viewer.poll_events()

    # Visualizes the input frame
    display.blit(
      pygame.surfarray.make_surface(
        np.transpose(
          imresize(frame_large, (window_size, window_size)
        ), (1, 0, 2))
      ),
      (0, 0)
    )
    pygame.display.update()

    clock.tick(30)


if __name__ == '__main__':
  live_application(OpenCVCapture())
