import pybullet as p
import pybullet_data
import numpy as np


def add_noodle(pos, orientation):
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
        # springElasticStiffness=0.1,
        # springDampingStiffness=1000,
        # springDampingAllDirections=0,
        frictionCoeff=2,
        # useFaceContact=True,
        useSelfCollision = 1,
        # repulsionStiffness=500
        )
    p.changeVisualShape(id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
    p.changeVisualShape(id,-1, rgbaColor =[1,.8,.3,1])  
    return id

physicsClient = p.connect(p.GUI)
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
plane_id = p.loadURDF("simple_xarm/resources/urdf/whiteplane.urdf", [0, 0, -.01], useFixedBase=True)

p.setGravity(0, 0, -9)

current_obj=np.array([1,0,.1])
object_id = add_noodle(pos = [-3,0,0], orientation=[0.22,0.2,0,0])

left_lower_bound = current_obj-[3,1,0.1]
left_upper_bound = current_obj+[3,0.0,0.3]
right_lower_bound = current_obj-[3,0.0,0.1]
right_upper_bound = current_obj+[3,1,0.3]


# p.createSoftBodyAnchor(object_id  ,1,-1,-1)
# p.createSoftBodyAnchor(object_id  ,500,-1,-1)
while(True):
    p.stepSimulation()
    p.addUserDebugLine(left_lower_bound,right_upper_bound, lifeTime = 0.1)
    # p.addUserDebugLine(right_lower_bound,right_upper_bound, lifeTime = 0.1)
