
from iris.raw import RawDataset

TEST_PATH = 'C:\\Diffraction data\\2017.03.03.10.13.WVO2_1p5pc_18mj'

if __name__ == '__main__':
    r = RawDataset(TEST_PATH)
    r.process('test.hdf5', center = (861, 949), radius = 100, 
               beamblock_rect = (721, 1055, 0, 1091), sample_type = 'powder')