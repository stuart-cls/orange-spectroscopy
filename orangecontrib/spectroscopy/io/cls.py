import h5py
from Orange.data import FileFormat

from orangecontrib.spectroscopy.io.util import SpectralFileFormat


class HDF5Reader_SGM(FileFormat, SpectralFileFormat):
    """ A very case specific reader for interpolated hyperspectral mapping HDF5
    files from the SGM beamline at the CLS"""
    EXTENSIONS = ('.h5',)
    DESCRIPTION = 'HDF5 file @SGM/CLS'

    def read_spectra(self):
        with h5py.File(self.filename, 'r') as h5:
