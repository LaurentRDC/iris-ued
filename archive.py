from iris import pack, open_raw

if __name__ == "__main__":
    with open_raw(
        "D:\\Diffraction data\\SnSe\\SnSe Batch 3 Sample E3\\2018-11-16 - SnSe Batch 2 Sample E3"
    ) as raw:
        print(raw)
        pack(raw, "C:\\Users\\Laurent\\Desktop\\archive.hdf5")
