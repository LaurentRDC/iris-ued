
from iris.raw import TEST_PATH, RawDataset

if __name__ == '__main__':
    r = RawDataset(TEST_PATH)
    r.process('test.hdf5', center = (861, 949), radius = 100, 
               beamblock_rect = (721, 1055, 0, 1091), sample_type = 'powder')