from ntpath import join
from re import A
import gym
from gym import error, spaces, utils
from gym.utils import seeding


import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import pathlib
import time
from interruptingcow import timeout

from simple_xarm.resources.robot.Xarm import Xarm_robot

#How to find driving joints:
#for id in range(numJoints):
#   print(f"{id},{p.getJointInfo(self.xarm_id,id)[12].decode('UTF-8')},{p.getJointInfo(self.xarm_id,id)[1].decode('UTF-8')},{p.getJointInfo(self.xarm_id,id)[2]}")
#self.num_dof = number of joint of type 0 (revolute) type 4 is fixed
#self.drivingJoints = list of type 0 joints id

class XarmEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.step_counter = 0
        workDir = pathlib.Path(__file__).parent.resolve()
        self.resourcesDir = os.path.join(workDir, "../resources") 
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=60, cameraPitch=-10, cameraTargetPosition=[1,.5,1])
        self.action_space = gym.spaces.box.Box(low = np.array([-1]*4, dtype=np.float32), high = np.array([1]*4, dtype=np.float32))
        low_bound = np.array([-1]*7, dtype=np.float32)
        low_bound[3] = 0  #Gripper dist
        high_bound = np.array([1]*7, dtype=np.float32)
        high_bound[3] = 0.12 #Gripper dist
        self.observation_space = gym.spaces.box.Box(low = low_bound, high = high_bound)
        #
        #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.xarm_id = p.loadURDF(f"{self.resourcesDir}/urdf/xarm7_g/xarm7_with_gripper.urdf", [0, 0, 0.5], useFixedBase=True)
        self.driving_joints = [1,2,3,4,5,6,7,10,11,12,13,14,15]
        self.num_dof = len(self.driving_joints)
        self.joint_goal = [0]*self.num_dof
        self.current_timeStep = 0
        self.Xarm = Xarm_robot(self.xarm_id)
    
    def step(self, action):
        self.current_timeStep+=1
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        action = self.mapAction(action)
        new_pos,dgripper = self.getNewPos(action)
        self.Xarm.setPose(new_pos,self.initial_eef_q4,self.eef_id,grip_width=dgripper)
        p.stepSimulation()

        current_pos, current_obj,_,_,dist_fing = self.getObservation()

        reward = self.calculateReward()
       
        self.observation = np.concatenate((current_pos, dist_fing , current_obj), axis=None, dtype=np.float32)

        if self.current_timeStep > 1000:
            print("step     1000")
            self.done = True
            self.current_timeStep =0
            reward = -100/4*0.6
        elif current_obj[2] > 1: 
            print(" object z > 1")
            self.done = True
            self.current_timeStep =0
            reward = 1000/4*0.8
        elif reward < -1:     #Softbody crash
            print("Soft body crash")
            self.done = True
            self.current_timeStep =0
            reward = -100/4*0.8

        return self.observation, reward, self.done, dict()
        
    def calculateReward(self):
        current_pos, current_obj,left_finger,right_finger,dist_fing = self.getObservation()
        
        goal_z = 1

        distance_obj_ori_x = self.getDistance(current_obj[0],self.state_object[0])
        distance_obj_ori_y = self.getDistance(current_obj[1],self.state_object[1])
        reward_distance_obj_original =  -(distance_obj_ori_x + distance_obj_ori_y)/5

        # Calculate distance
        distance_obj_goal_x = self.getDistance(current_obj[0], current_pos[0])
        distance_obj_goal_y = self.getDistance(current_obj[1], current_pos[1])
        distance_obj_goal_z = self.getDistance(current_obj[2], current_pos[2])-0.2  #offset
        reward_distance_gripper_obj = (1-((distance_obj_goal_x+distance_obj_goal_y))- distance_obj_goal_z)
        
        distance_obj_goal = self.getDistance(goal_z, current_obj[2]) + 0.18
        reward_distance_obj_goal = (goal_z-distance_obj_goal)*8

        left_lower_bound = current_obj-(0.3,0.6,0.5)
        left_upper_bound = current_obj+(0.3,0.0,0.2)
        right_lower_bound = current_obj-(0.3,0.0,0.5)
        right_upper_bound = current_obj+(0.3,0.6,0.2)
        if (left_lower_bound < left_finger).all() and (left_finger< left_upper_bound).all() and (right_lower_bound < (right_finger)).all() and (right_finger < right_upper_bound).all():
            # print("***** grap posision *****")
            if self.current_timeStep % 10 == 0:
                if dist_fing<0.6:
                    print("************grab**********")
                    print("finger", dist_fing)
            reward_gripper = (1-dist_fing) #0=open
            if len(p.getContactPoints(self.object_id,self.xarm_id)) > 2:
                reward_gripper = reward_gripper*2
        else: 
            if dist_fing>.4:
                reward_gripper = 0.1
            else:reward_gripper = 0

        penalty_time = self.current_timeStep * 0.00001        
        reward = reward_distance_obj_goal + reward_gripper + reward_distance_gripper_obj/2
        # reward = reward_distance_gripper_obj + reward_distance_obj_goal + reward_gripper + reward_distance_obj_original
        Norm_reward = self.RewardNorm(reward)
        if self.current_timeStep % 10 == 0:
            print("reward_distance_gripper_obj",reward_distance_gripper_obj)
            print("reward_distance_obj_goal ",reward_distance_obj_goal)
            # print("reward_distance_obj_original ",reward_distance_obj_original)
            print("reward_gripper",reward_gripper)
            # print("penalty_time",-penalty_time)
            print("Total reward",reward)
            print("Norm reward",Norm_reward)
            # print("dist_fing", dist_fing)
            # print("right finger", right_finger)

        return Norm_reward

    def RewardNorm(self,reward):
        # motion_range = [-0.05,0.05]
        rew_range = [0,1]
        # gripper_range = [0.0,0.85]
        reward = np.interp(reward,[0,2.2],rew_range)
        return reward

    def getObservation(self):
        # current_pose = self.getLinkPose(self.eef_id)
        current_pose = self.Xarm.getLinkPose(self.eef_id)

        current_pos = np.array(current_pose[0])
        current_obj = p.getBasePositionAndOrientation(self.object_id)
        current_obj = np.array(current_obj[0])
        left_finger = p.getLinkState(self.xarm_id,11)[0]
        right_finger = p.getLinkState(self.xarm_id,14)[0]
        dist_fing = np.linalg.norm(tuple(map(lambda i, j: i - j, left_finger, right_finger)))
        return current_pos, current_obj,left_finger,right_finger, dist_fing

    def getNewPos(self,action):
        current_pos,_,_,_,_ = self.getObservation()
        dx,dy,dz,dgripper = action
        new_pos = [current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz]
        return new_pos,dgripper
    
    def getDistance(self, a, b):
        return abs(a-b)
    
    def mapAction(self,action):
        # motion_range = [-0.05,0.05]
        motion_range = [-0.3,0.3]
        gripper_range = [0.0,0.85]
        action[0] = np.interp(action[0],[-1,1],motion_range)
        action[1] = np.interp(action[1],[-1,1],motion_range)
        action[2] = np.interp(action[2],[-1,1],motion_range)
        if action[3]<0:
            action[3] = 0
        else: action[3] = 1

        # action[3] = np.interp(action[3],[-1,1],gripper_range)
        return action

    def setJoints(self,joint_goal,wait_finish=False):
        #print(f"num_dof: {self.num_dof},len_joint_goal={len(joint_goal)}")
        #joint_goal = [1.57,0,0,0,0,0,1.57,0]
        p.setJointMotorControlArray(bodyIndex=self.xarm_id,
                                    jointIndices=self.driving_joints, # 0 is base
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_goal,
                                    targetVelocities=[0 for i in range(self.num_dof)],
                                    forces=[500 for i in range(self.num_dof)],
                                    positionGains=[0.03 for i in range(self.num_dof)],
                                    velocityGains=[1 for i in range(self.num_dof)])
        current_joint_pos = self.Xarm.getJointStates()
        t0 = time.time()
        timeout = 3 #sec
        isFinish = 0
        if wait_finish:
            while time.time() - t0 < timeout:
                if math.dist(current_joint_pos,joint_goal) < 0.005:
                    isFinish = 1
                    break
                current_joint_pos = self.Xarm.getJointStates()
        return isFinish
    
    def reset(self):
        self.done=False
        self.step_counter = 0
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        # p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # ref to pybullet_data
        p.setGravity(0,0,-10)

        # self.xarm_id = p.loadURDF(f"{self.resourcesDir}/urdf/xarm7_g/xarm7_with_gripper.urdf", [0, 0, 0.5], useFixedBase=True)
        self.xarm_id = p.loadURDF(f"{self.resourcesDir}/urdf/xarm7_g/xarm7_with_gripper.urdf", [0, 0, 0], useFixedBase=True,globalScaling=5)
        p.resetBasePositionAndOrientation(self.xarm_id, [0, 0, 0], [0, 0, 0, 1])
        numJoints = p.getNumJoints(self.xarm_id)
        link_name_to_index = dict((p.getJointInfo(self.xarm_id,id)[12].decode('UTF-8'), id) for id in range(numJoints))
        joint_name_to_index = dict((p.getJointInfo(self.xarm_id,id)[1].decode('UTF-8'), id) for id in range(numJoints))
        # for id in range(numJoints):
            # print(f"{id},{p.getJointInfo(self.xarm_id,id)[12].decode('UTF-8')},{p.getJointInfo(self.xarm_id,id)[1].decode('UTF-8')},{p.getJointInfo(self.xarm_id,id)[2]}")
        xarmEndEffectorIndex = link_name_to_index['link_tcp'] 
        self.eef_id = xarmEndEffectorIndex
        # tableUid = p.loadURDF("table/table.urdf",basePosition=[0.5,0,-0.65])
        # trayUid = p.loadURDF("tray/traybox.urdf",basePosition=[0.65,0,0],globalScaling=0.6)
        self.state_object= [0.6, 0, 0.05]
        # self.object_id = p.loadURDF("cube.urdf", basePosition = self.state_object,globalScaling=0.05)
        self.object_id = p.loadSoftBody("tube.vtk", simFileName = "tube.vtk", basePosition = [2.5,-0.1,0.13],baseOrientation = [0,0,1,1], scale =0.5, mass = 1, 
        useBendingSprings = 1,springBendingStiffness = 1, useNeoHookean = 1, NeoHookeanMu = 500, NeoHookeanLambda = 1000, NeoHookeanDamping = 0.002, useSelfCollision = 1, frictionCoeff = 3, collisionMargin = 0.001)
        self.observation =  self.getObservation()
        eef_state = self.Xarm.getLinkPose(link_id=8)
        self.initial_eef_p3,self.initial_eef_q4 = eef_state[0],eef_state[1]
        self.Xarm.setInitPose()
        # self.Xarm.reset()
        # #reset the env to initial state
        

        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        
        #print(f"{p.getLinkState(self.xarm_id,0)}")
        p.changeDynamics(self.object_id, -1,mass=0.1, lateralFriction=100)
        #p.changeDynamics(self.xarm_id, 11, lateralFriction=0.9,spinningFriction=1.9, rollingFriction=1.9)
        #p.changeDynamics(self.xarm_id, 14, lateralFriction=0.9,spinningFriction=1.9, rollingFriction=.9)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        
        current_pose = self.Xarm.getLinkPose(self.eef_id)
        # current_pos = np.array(current_pose[0])
        
        self.observation = np.concatenate((current_pose[0], 0 , self.state_object), axis=None, dtype=np.float32)
        # self.observation = np.array([1]*7)
        # print("observe" ,self.observation)
        return self.observation
    

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[2.8,-0.1,0.2],
                                                            distance=1.5,
                                                            yaw=90,
                                                            pitch=-40,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    
    

    def close(self):
        p.disconnect()

    