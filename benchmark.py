
from iris.raw import RawDataset
from iris.processing import process
import tempfile
from os.path import join

TEST_PATH = 'C:\\Diffraction data\\2017.06.29.21.59.FeMgO_12_12_1_7-5mJ_long20scans'

if __name__ == '__main__':
    process(RawDataset(TEST_PATH), destination = join(tempfile.gettempdir(), 'test.hdf5'),
            beamblock_rect = (0,0,0,0), processes = 4, callback = print)