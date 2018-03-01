

class FSURawDataset(AbstractRawDataset):

    @classmethod
    def parse_notes(cls, path):
        """ 
        Parse the metadata in a notes.txt file 
        """
        metadata = dict()
        unparsed = dict()
        with open(path) as file:
            for line in file:
                line = line.strip().lower()
                with suppress(ValueError): # not enough values to unpack
                    key, val = line.split(':', maxsplit = 1)
                    unparsed[key] = val
        
        metadata['acquisition_date'] = str(unparsed['date'] + unparsed['time'])
        metadata['temperature'] = float(unparsed['temperature'].strip('k'))
        metadata['exposure'] = float(unparsed['exposure time'].strip('ms')) * 1e-3 #from ms to seconds
        metadata['nscans'] = tuple(range(1, int(unparsed['number of stages']) + 1))
        metadata['energy'] = float(unparsed['cathode voltage'].strip('kv'))
        metadata['notes'] = str(unparsed.get('user notes', ''))
        return metadata

    def __init__(self, directory):
        if isdir(directory):
            self.raw_directory = directory
        else:
            raise ValueError('The path {} is not a directory'.format(directory))
        
        for k, v in self.parse_notes(join(self.raw_directory, 'notes.txt')):
            setattr(self, k, v)
    
    @property
    def _image_list(self):
        """ All images in the raw folder. """
        return (f for f in listdir(self.raw_directory) 
                  if isfile(join(self.raw_directory, f)) and f.endswith(('.tif', '.tiff')))

    @cached_property
    def _stage_positions(self):
        # stage position given in mm
        # filename format is tr_000000xx.tif
        return list(map(lambda fname: fname[3:9], self._image_list))

    @cached_property
    def time_points(self):
        # TODO: time-zero stage position
        return 4*3.33*self._stage_positions
    
    def raw_data_filename(timedelay, scan = 1):
        stage_pos = int(timedelay / (4 * 3.33))
        return join(self.raw_directory, 'tr_{}p0.tif'.format(stage_pos))