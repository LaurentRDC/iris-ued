import os.path
from skimage.io import imread

test_picture = imread(os.path.join(os.path.dirname(__file__), 'test_diff_picture.tif'))