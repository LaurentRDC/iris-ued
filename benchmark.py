
from iris.raw import RawDataset
from iris.processing import process

TEST_PATH = 'C:\\Diffraction data\\2017.03.03.10.13.WVO2_1p5pc_18mj'

if __name__ == '__main__':
    r = RawDataset(TEST_PATH)
    process(r, destination = 'C:\\Diffraction data\\test.hdf5', 
            beamblock_rect = (0,0,0,0), processes = 4, sample_type = 'single_crystal')