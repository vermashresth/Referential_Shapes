import pybullet as p
import pybullet_data
import os
import time
GRAVITY = -9.8
dt = 1e-3
iters = 2000
import pybullet_data
import matplotlib.pyplot as plt

physicsClient = p.connect(p.GUI)
#p.setAdditionalSearchPath('./')
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
#p.setRealTimeSimulation(True)
p.setGravity(0, 9.8, GRAVITY)
p.setTimeStep(dt)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [1, -1, 0.5]
cubeStartOrientation = p.getQuaternionFromEuler([3.14/2, 0, 0])
botId = p.loadURDF("my_sphere_green_small.urdf", cubeStartPos, cubeStartOrientation)
#textureId = p.loadTexture('red.png')
#p.changeVisualShape(botId, -1, textureUniqueId=textureId)
#disable the default velocity motors
#and set some position control with small force to emulate joint friction/return to a rest pose
#jointFrictionForce = 1
#for joint in range(p.getNumJoints(botId)):
#  p.setJointMotorControl2(botId, joint, p.POSITION_CONTROL, force=jointFrictionForce)

#for i in range(10000):
#     p.setJointMotorControl2(botId, 1, p.TORQUE_CONTROL, force=1098.0)
#     p.stepSimulation()
#import ipdb
#ipdb.set_trace()

viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[1.5, 1.5, 2],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[-1, -1, 1])
projectionMatrix = p.computeProjectionMatrixFOV(
    fov=90.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=30.1)

import time
p.setRealTimeSimulation(1)
st = 150
while (st>0):
  #p.stepSimulation()
  #p.setJointMotorControl2(botId, 1, p.TORQUE_CONTROL, force=1098.0)
  p.setGravity(0, 0, GRAVITY)
  st-=1
  print(st)
  time.sleep(1 / 2000.)
while(1):
  time.sleep(1 / 240.)
  width, height, rgbImg, depthImg, segImg = p.getCameraImage(
    width=64, 
    height=64,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix)

