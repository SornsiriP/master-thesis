from os import wait3
import gym
from simple_xarm.envs.xarm_ik_env import XarmIK
import pybullet as p
import numpy as np
import time
import csv
# from simple_xarm.envs.xarm_env import XarmEnv

env = XarmIK()
observation = env.reset()

def write_deform_csv():
  array_2d = p.getMeshData(env.object_id)[1]
  with open('deformation.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)

    writer.writerows(array_2d)
    print("*****Writing deform to csv******")
    # f.close()

def write_original_csv():
  array_2d = p.getMeshData(env.object_id)[1]
  with open('original.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)

    writer.writerows(array_2d)
    print("*****Writing original to csv******")


for _ in range(5000):

  events = p.getKeyboardEvents()

  rKey = ord('r')
  # print(events)
  if rKey in events and events[rKey] & p.KEY_IS_DOWN:
    env.reset()  

  obj_pos = list(env.getObjectPose(env.object_id)[0])
  # print(f"obj_pos: {obj_pos}")
  pre_pick_offset = [0,0,.3]
  pre_pick_pos = [obj_pos[0]+pre_pick_offset[0],obj_pos[1]+pre_pick_offset[1],obj_pos[2]+pre_pick_offset[2]]
  # pick_pos = [obj_pos[0],obj_pos[1],obj_pos[2]-0.07]
  pick_pos = [obj_pos[0],obj_pos[1],obj_pos[2]-.59]
  highest = [obj_pos[0],obj_pos[1],obj_pos[2]+pre_pick_offset[2]+1]
  # array_2d = p.getMeshData(env.object_id)[1]
  # print("Mesh data"   , p.getMeshData(env.object_id)[1])
  # print("Max in array",np.max(array_2d,axis=0))
  env.setPose(pre_pick_pos,env.initial_eef_q4,grip_width=0,wait_finish=True) #open

  # print("open")
  # env.setPose(pick_pos,env.initial_eef_q4,grip_width=0,wait_finish=True) #open
  # print("close")
  # env.setPose(pick_pos,env.initial_eef_q4,grip_width=1,wait_finish=True) #close
  # print("half")
  # env.setPose(pick_pos,env.initial_eef_q4,grip_width=0.5,wait_finish=True) #half
  
  print("2")
  write_original_csv()
  env.setPose(pick_pos,env.initial_eef_q4,grip_width=0.,wait_finish=True)
  # env.remove_anchor()
  print("3")
  # print("Get joint",env.getJointStates())
  env.setPose(pick_pos,env.initial_eef_q4,grip_width=0.9,wait_finish=True)
  print("4")
  env.setPose(pre_pick_pos,env.initial_eef_q4,grip_width=0.85,wait_finish=True)
  env.setPose(highest,env.initial_eef_q4,grip_width=0.85,wait_finish=True)
  
  for _ in range(300):
    env.step()
  
  # write_deform_csv()

env.close()