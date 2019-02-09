#!/usr/bin/env python2

import cairo
import itertools
import numpy as np

N_QUERY_INSTS = 1#64

N_TRAIN_TINY    = 1
N_TRAIN_SMALL = 10
N_TRAIN_MED     = 100
N_TRAIN_LARGE = 1000
N_TRAIN_ALL     = N_TRAIN_MED

SHAPE_CIRCLE = 0
SHAPE_SQUARE = 1
SHAPE_TRIANGLE = 2
N_SHAPES = SHAPE_TRIANGLE + 1

SIZE_SMALL = 0
SIZE_BIG = 1
N_SIZES = SIZE_BIG + 1

COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
N_COLORS = COLOR_BLUE + 1

WIDTH = 30
HEIGHT = 30

N_CELLS = 3

CELL_WIDTH = WIDTH / N_CELLS
CELL_HEIGHT = HEIGHT / N_CELLS

BIG_RADIUS = CELL_WIDTH * .75 / 2
SMALL_RADIUS = CELL_WIDTH * .5 / 2


def draw(shape, color, size, left, top, ctx):
    center_x = (left + .5) * CELL_WIDTH
    center_y = (top + .5) * CELL_HEIGHT

    radius = SMALL_RADIUS if size == SIZE_SMALL else BIG_RADIUS
    radius *= (.9 + np.random.random() * .2)

    if color == COLOR_RED:
        rgb = np.asarray([1., 0., 0.])
    elif color == COLOR_GREEN:
        rgb = np.asarray([0., 1., 0.])
    else:
        rgb = np.asarray([0., 0., 1.])
    rgb += (np.random.random(size=(3,)) * .4 - .2)
    rgb = np.clip(rgb, 0., 1.)

    #rgb = np.asarray([1., 1., 1.])

    if shape == SHAPE_CIRCLE:
        ctx.arc(center_x, center_y, radius, 0, 2*np.pi)
    elif shape == SHAPE_SQUARE:
        ctx.new_path()
        ctx.move_to(center_x - radius, center_y - radius)
        ctx.line_to(center_x + radius, center_y - radius)
        ctx.line_to(center_x + radius, center_y + radius)
        ctx.line_to(center_x - radius, center_y + radius)
    else:
        ctx.new_path()
        ctx.move_to(center_x - radius, center_y + radius)
        ctx.line_to(center_x, center_y - radius)
        ctx.line_to(center_x + radius, center_y + radius)
    ctx.set_source_rgb(*rgb)
    ctx.fill()

class Image:
    def __init__(self, shapes, colors, sizes, data, cheat_data = None):
        self.shapes = shapes
        self.colors = colors
        self.sizes = sizes
        self.data = data
        self.cheat_data = cheat_data

def sample_image():
    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
    cheat_data = np.zeros((6, 3, 3))
    surf = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(0., 0., 0.)
    ctx.paint()

    shapes = [[None for c in range(3)] for r in range(3)]
    colors = [[None for c in range(3)] for r in range(3)]
    sizes = [[None for c in range(3)] for r in range(3)]

    for r in range(3):
        for c in range(3):
            if np.random.random() < 0.2:
                continue
            shape = np.random.randint(N_SHAPES)
            color = np.random.randint(N_COLORS)
            size = np.random.randint(N_SIZES)
            draw(shape, color, size, c, r, ctx)
            shapes[r][c] = shape
            colors[r][c] = color
            sizes[r][c] = size
            cheat_data[shape][r][c] = 1
            cheat_data[N_SHAPES + color][r][c] = 1

    #surf.write_to_png("_sample.png")
    return Image(shapes, colors, sizes, data, cheat_data)


def get_image(shape, color, n=1, nOtherShapes=0, shouldOthersBeSame=False):
    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
    surf = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(0., 0., 0.)
    ctx.paint()

    shapes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
    colors = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
    sizes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]

    if n == 1:
        # Random location
        r = np.random.randint(N_CELLS)
        c = np.random.randint(N_CELLS)

        shapes[r][c] = shape
        colors[r][c] = color
        sizes[r][c] = np.random.randint(N_SIZES)

        draw(shapes[r][c],
            colors[r][c],
            sizes[r][c],
            c,
            r,
            ctx)

    else:
        assert False, 'NYI n>1'

    return Image(shapes, colors, sizes, data)

if __name__ == "__main__":

    # From Serhii's original experiment
    train_size = 18626
    val_size = 2069
    test_size = 10126

    train_data = []
    val_data = []
    test_data = []



    train_data.append(get_image(SHAPE_TRIANGLE, COLOR_GREEN))

    train_data_tiny = train_data[:N_TRAIN_TINY * N_QUERY_INSTS]
    # train_data_small = train_data[:N_TRAIN_SMALL * N_QUERY_INSTS]
    # train_data_med = train_data[:N_TRAIN_MED * N_QUERY_INSTS]
    # train_data_large = train_data

    sets = {
        "train.tiny": train_data_tiny,
        # "train.small": train_data_small,
        # "train.med": train_data_med,
        # "train.large": train_data_large,
        # "val": val_data,
        # "test": test_data
    }

    for set_name, set_data in sets.items():
        set_inputs = np.asarray([image.data[:,:,0:3] for image in set_data])
        np.save("images/%s.input" % set_name, set_inputs)
