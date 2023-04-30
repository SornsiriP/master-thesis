from cgitb import enable
from ntpath import join

from sklearn.preprocessing import scale
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

#How to find driving joints:
#for id in range(numJoints):
#   print(f"{id},{p.getJointInfo(self.xarm_id,id)[12].decode('UTF-8')},{p.getJointInfo(self.xarm_id,id)[1].decode('UTF-8')},{p.getJointInfo(self.xarm_id,id)[2]}")
#self.num_dof = number of joint of type 0 (revolute) type 4 is fixed
#self.drivingJoints = list of type 0 joints id

class XarmIK(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.step_counter = 0
        workDir = pathlib.Path(__file__).parent.resolve()
        self.resourcesDir = os.path.join(workDir, "../resources") 
        #p.connect(p.GUI)
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[2,0,0.5])
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))
        #
        #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.xarm_id = p.loadURDF(f"{self.resourcesDir}/urdf/xarm7_g/xarm7_with_gripper.urdf", [0, 0, 0], useFixedBase=True)
        self.driving_joints = [1,2,3,4,5,6,7,10,11,12,13,14,15]
        self.num_dof = len(self.driving_joints)
        self.joint_goal = [0]*self.num_dof
    
    def step(self):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        p.stepSimulation()

    
    def mapAction(self,action):
        motion_range = [-0.1,0.1]
        gripper_range = [0.0,0.85]
        action[0] = np.interp(action[0],[-1,1],motion_range)
        action[1] = np.interp(action[1],[-1,1],motion_range)
        action[2] = np.interp(action[2],[-1,1],motion_range)
        action[3] = np.interp(action[3],[-1,1],gripper_range)
        return action

    def getLinkPose(self,link_id=0):
        currentPose = p.getLinkState(self.xarm_id,link_id)
        return currentPose
    
    def setPose(self,pos3 = [0,0,0],rot4=[0,0,0,1],grip_width=0.0,wait_finish=False):
        jointGoal = [0]*self.num_dof
        jointPoses = p.calculateInverseKinematics(self.xarm_id,
                                                  self.eef_id,
                                                  pos3,
                                                  rot4,
                                                  solver=0,
                                                  maxNumIterations=500,
                                                  residualThreshold=.0001)
        jointGoal = list(jointPoses[:self.num_dof]) #last joint is gripper
        jointleft = p.getLinkState(self.xarm_id,11)[0]
        jointright = p.getLinkState(self.xarm_id,14)[0]
        # state_object= (0.6, 0, 0.05)
        print("jointleft",jointleft)
        print("jointright",jointright)
        print("distance",abs(jointleft[1]-jointright[1]) )
        
        # if (state_object-(0.01,0.05,0.01) < jointleft).all() and (jointleft< state_object+(0.01,-0.03,0.01)).all() and (state_object-(0.01,-0.03,0.01) < (jointright)).all() and (jointright < state_object+(0.01,0.05,0.01)).all():
        #     print("***** grap posision *****")
        # else: print("No")
        jointGoal[7:] = [grip_width]*6
        isFinish = self.setJoints(jointGoal,wait_finish=wait_finish)
        return isFinish
    
    def getJointStates(self):
        current_joint_pos = [p.getJointState(self.xarm_id,id)[0] for id in range(1,self.num_dof+1)]
        return current_joint_pos

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
        current_joint_pos = self.getJointStates()
        print(current_joint_pos)
        t0 = time.time()
        timeout = 3 #sec
        isFinish = 0
        if wait_finish:
            while time.time() - t0 < timeout:
                if math.dist(current_joint_pos,joint_goal) < 0.005:
                    isFinish = 1
                    break
                current_joint_pos = self.getJointStates()
                self.step()
        return isFinish
    
    def getObjectPose(self,object_id):
        currentPose = p.getBasePositionAndOrientation(object_id)
        return currentPose
        
    def reset(self):
        self.step_counter = 0
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        # p.resetSimulation()
        p.setPhysicsEngineParameter(enableConeFriction=1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # ref to pybullet_data
        p.setGravity(0,0,-10)
        
        # reset the robot to initial state
        self.xarm_id = p.loadURDF(f"{self.resourcesDir}/urdf/xarm7_g/xarm7_with_gripper.urdf", [0, 0, 0], useFixedBase=True,globalScaling=10)
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
        p.resetBasePositionAndOrientation(self.xarm_id, [0, 0, 0], [0, 0, 0, 1])
        numJoints = p.getNumJoints(self.xarm_id)
        link_name_to_index = dict((p.getJointInfo(self.xarm_id,id)[12].decode('UTF-8'), id) for id in range(numJoints))
        joint_name_to_index = dict((p.getJointInfo(self.xarm_id,id)[1].decode('UTF-8'), id) for id in range(numJoints))
        print(link_name_to_index)
        for id in range(numJoints):
            print(f"{id},{p.getJointInfo(self.xarm_id,id)[12].decode('UTF-8')},{p.getJointInfo(self.xarm_id,id)[1].decode('UTF-8')},{p.getJointInfo(self.xarm_id,id)[2]}")
        
        xarmEndEffectorIndex = link_name_to_index["link_tcp"] 
        self.eef_id = xarmEndEffectorIndex
        eef_state = self.getLinkPose(link_id=8)
        self.initial_eef_p3,self.initial_eef_q4 = eef_state[0],eef_state[1]
        #restposes for null space
        rp = [0]*self.num_dof
        for i in range(self.num_dof):
            p.resetJointState(self.xarm_id, i, rp[i])

        #reset the env to initial state
        plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        # tableUid = p.loadURDF("table/table.urdf",basePosition=[0.5,0,-0.65],globalScaling=5)
        # trayUid = p.loadURDF("tray/traybox.urdf",basePosition=[0.65,0,0],globalScaling=0.8)
        state_object= [0.6, 0, 0.05]
        # self.object_id = p.loadURDF("cube.urdf", basePosition=state_object,globalScaling=0.05)
        # self.object_id = p.loadSoftBody("cylinder.vtk", simFileName = "cylinder.vtk", basePosition = [2.5,-0.1,0.2], scale =0.25, mass = 1, useNeoHookean = 1, NeoHookeanMu = 800, NeoHookeanLambda = 1200, NeoHookeanDamping = 0.001, useSelfCollision = 1, frictionCoeff = .9, collisionMargin = 0.001)
        # self.object_id = p.loadSoftBody("RopeNew.obj", simFileName = "tet.vtk", basePosition = [5,-0.1,1],baseOrientation = [0,0,0,1], scale =1, mass = 1, 
        # useBendingSprings = 1,springBendingStiffness = 1, useNeoHookean = 1, NeoHookeanMu = 500, NeoHookeanLambda = 1000, NeoHookeanDamping = 0.002, useSelfCollision = 1, frictionCoeff = 3, collisionMargin = 0.001)
        # p.changeVisualShape(self.object_id,-1, rgbaColor =[0,0,0,1])        # self.object_id = self.add_noodle(pos = [2,-3,1], orientation=[0.5,0,0,1])
        self.object_id= self.add_noodle(pos = [7,-3,1], orientation=[0,0,.5,1])
        # base_position, base_orientation = self.random_start() 
        # print(base_position,"*****",base_orientation)
        # self.object_id = self.add_noodle(pos = base_position, orientation=base_orientation)
        # self.add_noodle(pos = [3,3,.3], orientation=[0,0,0,1])

        # p.createSoftBodyAnchor(self.object_id  ,1,-1,-1)
        # p.createSoftBodyAnchor(self.object_id  ,500,-1,-1)
        # p.createSoftBodyAnchor(self.object_id  ,0,0,0)

        print(f"{p.getLinkState(self.xarm_id,0)}")
        p.changeDynamics(plane_id,-1, lateralFriction=.1)
        p.changeDynamics(self.xarm_id, 11, lateralFriction=10,spinningFriction=.5, rollingFriction=.5)
        p.changeDynamics(self.xarm_id, 14, lateralFriction=10,spinningFriction=.5, rollingFriction=.5)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        self.observation = [-1]*7
        return self.observation
    
    def random_start(self):
        pos_x = np.random.uniform(2, 8)
        pos_y = np.random.uniform(-3, 3)
        pos_z = 1
        ori_x = np.random.uniform(0, .5)
        ori_y = np.random.uniform(0, .5)
        ori_z = np.random.uniform(0, .1)
        ori_w = np.random.uniform(0, .1)
        position = [pos_x,pos_y,pos_z]
        orientation = [ori_x,ori_y,ori_z,ori_w]
        return position,orientation


    def add_noodle(self, pos, orientation):
        filename = "800.vtk"
        id = p.loadSoftBody("RopeNew.obj",
            simFileName=filename,
            basePosition=pos,
            # baseOrientation = [1,0,0,1],    #Upward
            baseOrientation = orientation,  
            scale= 1, 
            mass = 2, 
            collisionMargin=0.005,
            # useMassSpring=0,
            useBendingSprings=1,
            springBendingStiffness = 0.001,
            useNeoHookean=1,
            # NeoHookeanMu = 1100, 
            # NeoHookeanLambda = 900,
            NeoHookeanMu = 2000,         #stiffness/elastic modulus/shear deformation  /the more it resists shear deformation
            NeoHookeanLambda = 1500,     #High: make shape more stable (compressibility)/volumetric deformation.
            NeoHookeanDamping = 0.1,
            # NeoHookeanDamping = 0.5,
            # springElasticStiffness=0.1,
            # springDampingStiffness=1000,
            # springDampingAllDirections=0,
            frictionCoeff=5,
            # useFaceContact=True,
            useSelfCollision = 1,
            # repulsionStiffness=500
            )
        p.changeVisualShape(id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
        p.changeVisualShape(id,-1, rgbaColor =[1,.8,.3,1])  
        return id

    # def remove_anchor(self):
            # print("remove")
            # p.createSoftBodyAnchor(self.object_id  ,1,0,0)
    

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
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

    