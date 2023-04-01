import numpy as np
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import random
import math
import time
import pathlib
import os
from abc import ABC, abstractmethod

class Xarm_robot():

    def __init__(self,xarm_id):  
        self.driving_joints = [1,2,3,4,5,6,7,10,11,12,13,14,15]
        self.num_dof = len(self.driving_joints)
        self.joint_goal = [0]*self.num_dof
        
        self.temp = 1
        self.xarm_id = xarm_id
        # self.reset()
    
    def getLinkPose(self,link_id=0):
        currentPose = p.getLinkState(self.xarm_id,link_id)
        return currentPose
    
    def setPose(self,pos3 = [0,0,0],rot4=[0,0,0,1],eef_id = 0,grip_width=0.0,wait_finish=False):
        jointGoal = [0]*self.num_dof
        jointPoses = p.calculateInverseKinematics(self.xarm_id,
                                                  eef_id,
                                                  pos3,
                                                  rot4,
                                                #   solve0r=0,
                                                  maxNumIterations=500,
                                                  residualThreshold=.0001)
        jointGoal = list(jointPoses[:self.num_dof]) #last joint is gripper
        jointGoal[7:] = [grip_width]*6
        isFinish = self.setJoints(jointGoal,wait_finish=wait_finish)
        return isFinish
    
    def getJointStates(self):
        current_joint_pos = [p.getJointState(self.xarm_id,id)[0] for id in range(1,self.num_dof+1)]
        return current_joint_pos

    def setJoints(self,joint_goal,wait_finish=False):
        p.setJointMotorControlArray(bodyIndex=self.xarm_id,
                                    jointIndices=self.driving_joints, # 0 is base
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_goal,
                                    targetVelocities=[0 for i in range(self.num_dof)],
                                    forces=[500 for i in range(self.num_dof)],
                                    positionGains=[0.03 for i in range(self.num_dof)],
                                    velocityGains=[1 for i in range(self.num_dof)])
        current_joint_pos = self.getJointStates()
        t0 = time.time()
        timeout = 3 #sec
        isFinish = 0
        if wait_finish:
            while time.time() - t0 < timeout:
                if math.dist(current_joint_pos,joint_goal) < 0.005:
                    isFinish = 1
                    break
                current_joint_pos = self.getJointStates()
        return isFinish
    
    def setInitPose(self):
        # rp = [-0.001729720910844686, 0.2, 3.909082663468943e-06, 1.9761466873919924, -0.0033038848423014005, 1.5194642477590865, 5.489017156143467e-05, 0.0, 0.0, -2.5124309669405676e-16, 0.0009580626467569612, 2.0146295612691145e-05, -1.2045288941379286e-06] #home position
        # self.setJoints(rp)
        rp = [0.0007184558568159241, 0.16688883895190476, 0.0012846674003256628, 1.105374640469449, -0.0002070227924870695, 0.9394200246237291, 0.001847820801415448, 0.0, 0.0, -2.1007098223492635e-07, 0.0, 0.0, -2.712988540579329e-13]
        # rp = [-0.021673216419159883, 0.42168129303767743, -0.01597722406041248, 1.018271818123641, 0.013858068419425315, 0.5954027097592304, -0.04150813811542951, 0.0, 0.0, 0.0014712502380994752, 0.0, 0.0, -0.13811146793529266]
        for i,id in enumerate (self.driving_joints):
           p.resetJointState(self.xarm_id, id, rp[i])

    def reset(self):
        workDir = pathlib.Path(__file__).parent.resolve()
        self.xarm_id = p.loadURDF(f"{workDir}/../urdf/xarm7_g/xarm7_with_gripper.urdf", [0, 0, 0], useFixedBase=True)
        p.resetBasePositionAndOrientation(self.xarm_id, [0, 0, 0], [0, 0, 0, 1])
        numJoints = p.getNumJoints(self.xarm_id)
        link_name_to_index = dict((p.getJointInfo(self.xarm_id,id)[12].decode('UTF-8'), id) for id in range(numJoints))
        joint_name_to_index = dict((p.getJointInfo(self.xarm_id,id)[1].decode('UTF-8'), id) for id in range(numJoints))
        xarmEndEffectorIndex = link_name_to_index["link_tcp"] 
        self.eef_id = xarmEndEffectorIndex
        eef_state = self.getLinkPose(link_id=8)
        self.initial_eef_p3,self.initial_eef_q4 = eef_state[0],eef_state[1]
        rp = [-0.001729720910844686, 0.2, 3.909082663468943e-06, 1.9761466873919924, -0.0033038848423014005, 1.5194642477590865, 5.489017156143467e-05, 0.0, 0.0, -2.5124309669405676e-16, 0.0009580626467569612, 2.0146295612691145e-05, -1.2045288941379286e-06] #home position
        # self.setJoints(rp)
        for i,id in enumerate (self.driving_joints):
           p.resetJointState(self.xarm_id, id, rp[i])
        current_pose = self.getLinkPose(self.eef_id)

    def getObjectPose(self,object_id):
        currentPose = p.getBasePositionAndOrientation(object_id)
        return currentPose