
from iris.raw import TEST_PATH, RawDataset

if __name__ == '__main__':
    r = RawDataset(TEST_PATH)
    r.process('test.hdf5', center = (1100, 1100), radius = 100, beamblock_rect = (0,0,0,0), sample_type = 'powder')