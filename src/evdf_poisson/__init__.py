"""Poisson maximum-likelihood fitting of Solar Orbiter SWA-EAS electron
velocity distribution functions into core, halo and strahl components.

Typical use (repository ``src/`` on sys.path)::

    from evdf_poisson import loadEpoch, detectStrahlFromPad, seedFromData, fitEpoch
    grid, psd, counts, epochNs, scDensity = loadEpoch(h5Path, epochIndex)
    strahl = detectStrahlFromPad(h5Path, epochIndex, breakEnergyEv=breakEnergyEv)
    seeds = seedFromData(psd, counts, grid, strahlDetection=strahl,
                         scDensity=scDensity, breakEnergyEv=breakEnergyEv)
    params, modelPsd, diagnostics = fitEpoch(psd, counts, grid, seeds)

or, for whole files, ``runner.runFiles`` / ``python -m evdf_poisson.runner``.
"""

import os

# One BLAS/OpenMP thread per process: the fits are many small least-squares
# problems, so threaded BLAS only oversubscribes the machine once the batch
# runner forks/spawns several workers. Must be set before numpy is imported,
# hence at the very top of the package.
for _threadVariable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_threadVariable, '1')

from .epoch_io import loadEpoch, countEpochs, readEpochTimes
from .seeds import seedFromData
from .pad_detection import detectStrahlFromPad
from .objectives import fitEpoch, OBJECTIVES
from .metrics import (poissonDeviancePerDof, reducedChiSquare,
                      componentMomentsFromParams, sliceDataMeans)
from .runner import fitOneEpoch, runFiles

__all__ = ['loadEpoch', 'countEpochs', 'readEpochTimes', 'seedFromData',
           'detectStrahlFromPad', 'fitEpoch', 'OBJECTIVES',
           'poissonDeviancePerDof', 'reducedChiSquare',
           'componentMomentsFromParams', 'sliceDataMeans',
           'fitOneEpoch', 'runFiles']
