
from iris import McGillRawDataset, DiffractionDataset
from os.path import join

TEST_PATH = 'C:\\Diffraction data\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'

if __name__ == '__main__':
    raw = McGillRawDataset(TEST_PATH)

    with DiffractionDataset.from_raw(raw, 'test.hdf5', callback = print, processes = 4,
                                     exclude_scans = list(range(30, 120)), mode = 'w'):
        pass