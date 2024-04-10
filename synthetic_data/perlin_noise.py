import numpy as np
import cv2


def min_max_scaler(t):
    return (t - t.min()) / (t.max() - t.min())


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def perlin_2d(shape, gen_shape=(256, 256), gen_res=(4, 4), interpolation='bicubic'):
    if interpolation == 'bicubic':
        interpolation_type = cv2.INTER_CUBIC
    elif interpolation == 'linear':
        interpolation_type = cv2.INTER_LINEAR
    else:
        interpolation_type = cv2.INTER_NEAREST

    perlin_array = generate_perlin_noise_2d(shape=gen_shape, res=gen_res)
    return cv2.resize(perlin_array, dsize=shape, interpolation=interpolation_type)


class Perlin2D:
    def __init__(self, gen_shape=(256, 256), gen_res=(4, 4), interpolation='bicubic'):
        self.gen_shape = gen_shape
        self.gen_res = gen_res
        self.interpolation = interpolation

    def generate(self, shape):
        return perlin_2d(
            shape=shape,
            gen_shape=self.gen_shape,
            gen_res=self.gen_res,
            interpolation=self.interpolation
        )

    def __call__(self, image):
        shape = image.shape[1::-1]
        return min_max_scaler(self.generate(shape))


class Perlin2DMixture:
    def __init__(self, gen_shapes=[(256, 256)], gen_reses=[(4, 4)], gen_weights=None, interpolation='bicubic'):
        assert len(gen_shapes) == len(gen_reses)
        self.gen_shapes = gen_shapes
        self.gen_reses = gen_reses
        if gen_weights is None:
            gen_weights = [1] * len(self.gen_shapes)
        assert len(gen_weights) == len(gen_shapes)
        self.gen_weights = gen_weights
        self.interpolation = interpolation

    def generate(self, shape):
        arrays_list = []
        for gen_shape, gen_res, gen_weight in zip(self.gen_shapes, self.gen_reses, self.gen_weights):
            arrays_list.append(
                perlin_2d(
                    shape=shape,
                    gen_shape=gen_shape,
                    gen_res=gen_res,
                    interpolation=self.interpolation
                ) * gen_weight
            )
        return np.stack(arrays_list).mean(axis=0)

    def __call__(self, image):
        shape = image.shape[1::-1]
        return min_max_scaler(self.generate(shape))
