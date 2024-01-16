import h5py
import numpy as np
from Orange.data import FileFormat, ContinuousVariable

from orangecontrib.spectroscopy.io.util import SpectralFileFormat, _spectra_from_image


class HDF5Reader_SGM(FileFormat, SpectralFileFormat):
    """ A very case specific reader for interpolated hyperspectral mapping HDF5
    files from the SGM beamline at the CLS"""
    EXTENSIONS = ('.h5',)
    DESCRIPTION = 'HDF5 file @SGM/CLS'

    def read_spectra(self):
        with h5py.File(self.filename, 'r') as h5:
            NXentries = [str(x) for x in h5['/'].keys() if 'NXentry' in str(h5[x].attrs.get('NX_class'))]
            NXdata = [entry + "/" + str(x) for entry in NXentries for x in h5['/' + entry].keys()
                      if 'NXdata' in str(h5[entry + "/" + x].attrs.get('NX_class'))]
            axes = [[str(nm) for nm in h5[nxdata].keys() for s in h5[nxdata].attrs.get('axes') if str(s) in str(nm) or
                     str(nm) in str(s)] for nxdata in NXdata]
            indep_shape = [v.shape for i, d in enumerate(NXdata) for k, v in h5[d].items() if k in axes[i][0]]
            data = [
                {k: np.squeeze(v[()]) for k, v in h5[d].items() if v.shape[0] == indep_shape[i][0] and k not in axes[i]}
                for i, d in
                enumerate(NXdata)]

            # for i, d in enumerate(NXdata):
            # starting with single entry (0)
            i = 0
            d = NXdata[0]
            x_locs = h5[d][axes[i][0]]
            y_locs = h5[d][axes[i][1]]
            en = h5[d]['en']
            sdd3 = data[i]['sdd3']
            X = np.transpose(sdd3, (1, 0, 2))
            features = np.arange(X.shape[-1])

            features, spectra, meta_table = _spectra_from_image(X, features, x_locs, y_locs)
            meta_table = meta_table.add_column(ContinuousVariable("en"), np.ones(spectra.shape[0]) * en, to_metas=True)

            return features, spectra, meta_table