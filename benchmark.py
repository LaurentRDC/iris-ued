
from iris import MerlinRawDataset, DiffractionDataset
from npstreams import average
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg'
import matplotlib.pyplot as plt

TEST_PATH = 'D:\\Merlin data\\test-sequential'

if __name__ == '__main__':
    raw = MerlinRawDataset(TEST_PATH)

    im = average(raw.raw_data(0.0, scan) for scan in raw.scans)
    plt.figure()
    plt.imshow(im)
    plt.show()
    