import numpy as np
import cairo

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
N_SHAPES = SHAPE_TRIANGLE + 1

SIZE_SMALL = 0
SIZE_BIG = 1
N_SIZES = SIZE_BIG + 1

COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
N_COLORS = COLOR_BLUE + 1


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

# def get_image(shape=-1, color=-1, size=-1):
#     return get_image([Figure(shape, color, size, r=-1, c=-1)])

def get_image(figures):
    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
    surf = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(0., 0., 0.)
    ctx.paint()

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

        draw(shapes[r][c],
            colors[r][c],
            sizes[r][c],
            c,
            r,
            ctx)

    metadata = {'shapes':shapes, 'colors':colors, 'sizes':sizes}

    flat_shapes = [item for sublist in metadata['shapes'] for item in sublist]
    assert len(list(filter(lambda x: not x is None, flat_shapes))) == len(figures)

    return Image(shapes, colors, sizes, data, metadata)