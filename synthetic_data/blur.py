import numpy as np
import cv2
import kornia

def min_max_scaler(t):
    return (t - t.min()) / (t.max() - t.min())

class GaussianBlur:
    def __init__(self, k_size=(19, 19), sigma=(10, 10)):
        self.blur = kornia.filters.GaussianBlur2d(k_size, sigma)

    def __call__(self, image):
        image = kornia.image_to_tensor(image, keepdim=False)
        blured_image = self.blur(image.float())
        return kornia.tensor_to_image(blured_image)


class GaussianBlurPyramid:
    def __init__(self, k_size=(19, 19), sigma=(10, 10), steps=5):
        self.steps = steps
        sigma = np.array(sigma)
        self.blur_sigmas = []
        for i in range(1, steps + 1):
            new_sigma = np.sqrt((sigma * (i / steps)) ** 2 - (sigma * ((i - 1) / steps)) ** 2)
            self.blur_sigmas.append(tuple(new_sigma))
        self.blurs = list(map(lambda x: GaussianBlur(k_size, x), self.blur_sigmas))

    def __call__(self, image):
        pyramid = [image.copy()]
        for blur in self.blurs:
            pyramid.append(blur(pyramid[-1]))
        return pyramid

class Blender:
    def __init__(self, k_size=(19, 19), sigma=(10, 10), steps=5, blending_k_size=None, blending_sigma=None):
        self.steps = steps
        self.generator = GaussianBlurPyramid(k_size, sigma, steps)
        if blending_k_size is None:
            blending_k_size = k_size
        if blending_sigma is None:
            blending_sigma = tuple(np.array(sigma) / steps)
        self.blending_blur = GaussianBlur(blending_k_size, blending_sigma)

    def blend(self, image1, image2, blending_map):
        return image1 * (1 - blending_map)[..., None] + image2 * blending_map[..., None]

    def __call__(self, image, blur_map):
        pyramid = self.generator(image)
        image = np.zeros_like(pyramid[0])
        blend = np.zeros_like(blur_map)
        for i, (prev_stage, new_stage) in enumerate(zip(pyramid[:-1], pyramid[1::]), 1):
            blending_map = np.where((blur_map <= i / self.steps) * (blur_map >= (i - 1) / self.steps), 1, 0)
            blur_part = self.blend(prev_stage, new_stage, np.clip((blur_map - i / self.steps) * self.steps, 0, 1))
            blending_map = self.blending_blur(blending_map)
            image += blur_part * blending_map[..., None]
            blend += blending_map
        return np.clip(image / blend[..., None], 0, 1)
