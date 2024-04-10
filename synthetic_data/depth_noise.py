import numpy as np
import cv2


def min_max_scaler(t):
    return (t - t.min()) / (t.max() - t.min())

def depth_2d(depth_map, target_dist=0.5):
    blur_map = (1 - depth_map / target_dist)
    return min_max_scaler(np.abs(target_dist * blur_map))


class Depth2D:
    def __init__(self, focal_gamma=1, target_dist=0.5, focal_mode='mix'):
        self.focal_gamma = focal_gamma
        self.target_dist = target_dist
        assert focal_mode in ('pow', 'inv', 'mix')
        self.focal_mode = focal_mode

    def __call__(self, depth_map, target_dist=None, focal_gamma=None):
        blur_map = depth_2d(
            depth_map=depth_map,
            target_dist=self.target_dist if target_dist is None else target_dist,
        )
        if focal_gamma is None:
            focal_gamma = self.focal_gamma
        if self.focal_mode == 'mix':
            focal_mode = ('pow', 'inv')[focal_gamma > 1]
        if focal_mode == 'pow':
            blur_map = blur_map ** focal_gamma
        else:
            blur_map = 1 - (1 - blur_map) ** (1 / focal_gamma)
        return blur_map
