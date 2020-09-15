import pybullet as p
import pybullet_data
import os
import time
GRAVITY = -9.8
dt = 1e-3
iters = 2000
import pybullet_data

import numpy as np
import matplotlib.pyplot as plt

N_CELLS = 3

WIDTH = N_CELLS * 10
HEIGHT = WIDTH

CELL_WIDTH = WIDTH / N_CELLS
CELL_HEIGHT = HEIGHT / N_CELLS
N_CHANNELS = 3

BIG_RADIUS = CELL_WIDTH * .75 / 2
SMALL_RADIUS = CELL_WIDTH * .5 / 2

SHAPE_CIRCLE = 0
SHAPE_SQUARE = 1
SHAPE_TRIANGLE = 2

SHAPE_SPHERE = 0
SHAPE_CUBE = 1
SHAPE_CAPSULE = 2
N_SHAPES = SHAPE_CAPSULE + 1

SIZE_SMALL = 0
SIZE_BIG = 1
N_SIZES = SIZE_BIG + 1

COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
N_COLORS = COLOR_BLUE + 1
physicsClient = p.connect(p.DIRECT)
#p.setAdditionalSearchPath('./')
p.setAdditionalSearchPath(pybullet_data.getDataPath())

class Image:
    def __init__(self, shapes, colors, sizes, data, metadata):
        self.shapes = shapes
        self.colors = colors
        self.sizes = sizes
        self.data = data
        self.metadata = metadata

class Figure:
    def __init__(self, shape, color, size, r, c):
        self.shape = shape
        self.color = color
        self.size = size
        self.r = r
        self.c = c

    def __repr__(self):
        return 'Figure at ({},{}) | shape {} | color {} | size {}'.format(
            self.r,
            self.c,
            self.shape,
            self.color,
            self.size)

def get_image(figures):
    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)

    shapes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
    colors = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
    sizes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]

    for fig in figures:
        # Random shape, color, size each time if they are -1
        shape = fig.shape if fig.shape >= 0 else np.random.randint(N_SHAPES)
        color = fig.color if fig.color >= 0 else np.random.randint(N_COLORS)
        size = fig.size if fig.size >= 0 else np.random.randint(N_SIZES)

        # Random location if they're -1
        r = fig.r if fig.r >= 0 else np.random.randint(N_CELLS)
        c = fig.c if fig.c >= 0 else np.random.randint(N_CELLS)

        if fig.r >= 0 and fig.c >= 0:
            assert shapes[r][c] is None
        else:
            while not shapes[r][c] is None:
                if fig.r >= 0:
                    c = np.random.randint(N_CELLS)
                elif fig.c >= 0:
                    r = np.random.randint(N_CELLS)
                else:
                    c = np.random.randint(N_CELLS)
                    r = np.random.randint(N_CELLS)

        shapes[r][c] = shape
        colors[r][c] = color
        sizes[r][c] = size

        data = draw_bullet(shapes[r][c],
            colors[r][c],
            sizes[r][c],
            c,
            r,
            data)

    metadata = {'shapes':shapes, 'colors':colors, 'sizes':sizes}

    flat_shapes = [item for sublist in metadata['shapes'] for item in sublist]
    assert len(list(filter(lambda x: not x is None, flat_shapes))) == len(figures)

    return Image(shapes, colors, sizes, data, metadata)

def draw_bullet(shape, color, size, r, c, data):


    p.resetSimulation()
    #p.setRealTimeSimulation(True)
    p.setGravity(0, 9.8, GRAVITY)
    p.setTimeStep(dt)
    planeId = p.loadURDF("plane.urdf")


    sizes = ['small', 'big']
    colors = ['red', 'green', 'blue']
    shapes = ['sphere', 'cube', 'capsule']


    sizename = sizes[size]
    shapename = shapes[shape]
    colorname = colors[color]

    body_name = 'shapes/bullet_data/my_{}_{}_{}.urdf'.format(shapename, colorname, sizename)
    if size == 0:
        factor = 0.5
    else:
        factor = 1
    if shape == 2:
        body_name = 'shapes/bullet_data/my_{}_{}.urdf'.format(shapename, sizename)
    z_centre = 0.5*factor

    cubeStartPos = [(r-1), (c-1), z_centre]
    cubeStartOrientation = p.getQuaternionFromEuler([3.14/2, 0, 0])

    botId = p.loadURDF(body_name, cubeStartPos, cubeStartOrientation)
    if shape == 2:
        textureId = p.loadTexture('shapes/bullet_data/{}.png'.format(colorname))
        p.changeVisualShape(botId, -1, textureUniqueId=textureId)

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
    st = 1500

    p.setGravity(0, 0, GRAVITY)

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=30,
        height=30,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)
    data = np.array(rgbImg)
    return data
