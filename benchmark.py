
from skued.image_analysis import angular_average
import numpy as np

im = np.random.random(size = (4096, 4096))

if __name__ == '__main__':
    for _ in range(10):
        angular_average(im, center = (2048, 2048))