import numpy as np

from dispernet import App
#with h5py.File('fileName.h5', 'w') as fw:
#	fw.create_dataset('f', data=freq)
#	fw.create_dataset('c', data=velo)
#	fw.create_dataset('amp', data=spectrum)

#App('/data/fordongsh/longbeach_ccfj_snr','./curveFile',np.arange(0.5,6,0.05))
App('/data/fordongsh/longbeach_ccfj_snr_subgroup','./curveFile',np.arange(0.5,6,0.05))
