"""EAS look-direction geometry: sensor-head frame -> spacecraft reference frame (SRF).

Each EAS head measures 16 elevation x 32 azimuth pixels at 64 energy steps. The
L2 CDF files carry the pixel centres and widths, the energy table and the fixed
rotation matrix ``EASx_TO_SRF`` of every head. This module extracts them once
into a small ``.npz`` (maintainers, needs cdflib) and rebuilds from that file,
with numpy only, everything the tutorials need: the SRF unit vector of every
pixel, its solid angle, and the pitch angle for a given magnetic field.

Conventions (identical to ``preprocess_data_for_fit_hdf5``):

* the head-frame unit vector of a pixel is
  ``(cos(el) cos(az), cos(el) sin(az), -sin(el))`` and points along the particle
  VELOCITY; the direction the pixel looks at is its negative;
* SRF vectors are ``head_to_srf @ head_vector``;
* the 1024 pixels of a preprocessed epoch are ordered EAS1 (0-511) then EAS2
  (512-1023), each head flattened as ``elevation * 32 + azimuth``.

CLI (repository root, maintainers):

    python src/eas_geometry.py --eas1-cdf <L2 EAS1 CDF> --eas2-cdf <L2 EAS2 CDF> \\
        --out examples/example_data/eas_head_geometry.npz
"""
import argparse
import os

import numpy as np

HEAD_NAMES = ('EAS1', 'EAS2')
ELEVATIONS_PER_HEAD = 16
AZIMUTHS_PER_HEAD = 32
PIXELS_PER_HEAD = ELEVATIONS_PER_HEAD * AZIMUTHS_PER_HEAD
PIXEL_ORDER = 'EAS1 pixels 0-511 then EAS2 pixels 512-1023; each head flattened as elevation*32 + azimuth'
HEAD_SLICES = {'EAS1': slice(0, PIXELS_PER_HEAD), 'EAS2': slice(PIXELS_PER_HEAD, 2 * PIXELS_PER_HEAD)}

GEOMETRY_KEYS = ('elevation_deg', 'elevation_delta_upper_deg', 'elevation_delta_lower_deg',
                 'azimuth_deg', 'azimuth_delta_upper_deg', 'azimuth_delta_lower_deg',
                 'energy_eV', 'energy_delta_upper_eV', 'energy_delta_lower_eV', 'to_srf')


def extractHeadGeometry(cdfPath, headName):
    """Static geometry of one head from an L2 nm3d-psd CDF (cdflib imported lazily)."""
    import cdflib
    cdf = cdflib.CDF(cdfPath)
    energyPrefix = f'SWA_{headName}_ENERGY'
    return {
        'elevation_deg': np.asarray(cdf.varget('SWA_EAS_ELEVATION'), dtype=float),
        'elevation_delta_upper_deg': np.asarray(cdf.varget('SWA_EAS_ELEVATION_delta_upper'), dtype=float),
        'elevation_delta_lower_deg': np.asarray(cdf.varget('SWA_EAS_ELEVATION_delta_lower'), dtype=float),
        'azimuth_deg': np.asarray(cdf.varget('SWA_EAS_AZIMUTH'), dtype=float),
        'azimuth_delta_upper_deg': np.asarray(cdf.varget('SWA_EAS_AZIMUTH_delta_upper'), dtype=float),
        'azimuth_delta_lower_deg': np.asarray(cdf.varget('SWA_EAS_AZIMUTH_delta_lower'), dtype=float),
        'energy_eV': np.asarray(cdf.varget(energyPrefix), dtype=float),
        'energy_delta_upper_eV': np.asarray(cdf.varget(f'{energyPrefix}_delta_upper'), dtype=float),
        'energy_delta_lower_eV': np.asarray(cdf.varget(f'{energyPrefix}_delta_lower'), dtype=float),
        'to_srf': np.asarray(cdf.varget(f'{headName}_TO_SRF'), dtype=float),
    }


def saveHeadGeometry(npzPath, headGeometries, sourceFiles):
    """Write the per-head tables into one npz; keys are ``<HEAD>_<quantity>``."""
    payload = {'pixel_order': np.array(PIXEL_ORDER), 'source_files': np.array(sourceFiles)}
    for headName in HEAD_NAMES:
        for key in GEOMETRY_KEYS:
            payload[f'{headName}_{key}'] = headGeometries[headName][key]
    np.savez(npzPath, **payload)


def loadHeadGeometry(npzPath):
    """{'EAS1': {...}, 'EAS2': {...}, 'pixel_order': str, 'source_files': [...]}, numpy only."""
    with np.load(npzPath) as archive:
        geometry = {headName: {key: archive[f'{headName}_{key}'] for key in GEOMETRY_KEYS}
                    for headName in HEAD_NAMES}
        geometry['pixel_order'] = str(archive['pixel_order'])
        geometry['source_files'] = [str(name) for name in archive['source_files']]
    return geometry


def headUnitVectors(elevationDeg, azimuthDeg):
    """(16, 32, 3) unit vectors along the particle velocity in the head frame."""
    elevation = np.deg2rad(elevationDeg)[:, None]
    azimuth = np.deg2rad(azimuthDeg)[None, :]
    unitX = np.cos(elevation) * np.cos(azimuth)
    unitY = np.cos(elevation) * np.sin(azimuth)
    unitZ = -np.sin(elevation) * np.ones_like(azimuth)
    return np.stack([unitX, unitY, unitZ], axis=-1)


def pixelUnitVectorsSrf(geometry):
    """(1024, 3) SRF unit vectors along the particle velocity, in preprocessed pixel order."""
    blocks = []
    for headName in HEAD_NAMES:
        head = geometry[headName]
        headVectors = headUnitVectors(head['elevation_deg'], head['azimuth_deg'])
        blocks.append((headVectors @ head['to_srf'].T).reshape(-1, 3))
    return np.concatenate(blocks, axis=0)


def pixelSolidAngles(geometry):
    """(1024,) solid angle of every pixel in steradian, same order as pixelUnitVectorsSrf."""
    blocks = []
    for headName in HEAD_NAMES:
        head = geometry[headName]
        elevation = np.deg2rad(head['elevation_deg'])
        elevationWidth = np.deg2rad(head['elevation_delta_upper_deg'] + head['elevation_delta_lower_deg'])
        azimuthWidth = np.deg2rad(head['azimuth_delta_upper_deg'] + head['azimuth_delta_lower_deg'])
        blocks.append((np.cos(elevation) * elevationWidth)[:, None] * azimuthWidth[None, :])
    return np.concatenate([block.reshape(-1) for block in blocks])


def pixelPitchAngles(unitVectors, bVector):
    """Pitch angle in degrees of velocity unit vectors (N, 3) for a magnetic field vector (3,)."""
    bUnit = np.asarray(bVector, dtype=float) / np.linalg.norm(bVector)
    cosine = np.clip(unitVectors @ bUnit, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def wrapTo180(angleDeg):
    """Wrap angles into (-180, 180], keeping +180 rather than -180."""
    wrapped = (np.asarray(angleDeg, dtype=float) + 180.0) % 360.0 - 180.0
    return np.where(wrapped == -180.0, 180.0, wrapped)


def easAzimuthElevation(vectors):
    """EAS-convention azimuth and elevation (degrees) of velocity vectors (N, 3) in SRF.

    Azimuth is 0 for electrons moving along -X_SRF (away from the Sun) and
    increases clockwise (the +Y_SRF direction is at +90); elevation is the
    latitude of the velocity above the X-Y plane. This is the field-of-view
    convention of Owen et al. (2020, A&A 642, A16, Fig. 3).
    """
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 0)
    elevation = np.degrees(np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)))
    standardAzimuth = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    azimuth = wrapTo180(-standardAzimuth - 180.0)
    return azimuth, elevation


def pixelLookAngles(unitVectors):
    """SRF longitude and latitude (degrees) of the LOOK direction, i.e. minus the velocity."""
    look = -np.asarray(unitVectors)
    longitude = np.degrees(np.arctan2(look[:, 1], look[:, 0]))
    latitude = np.degrees(np.arcsin(np.clip(look[:, 2], -1.0, 1.0)))
    return longitude, latitude


def directionAngles(vector):
    """SRF longitude and latitude (degrees) of the direction of one vector (3,)."""
    unit = np.asarray(vector, dtype=float) / np.linalg.norm(vector)
    return (np.degrees(np.arctan2(unit[1], unit[0])),
            np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eas1-cdf', required=True)
    parser.add_argument('--eas2-cdf', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    headGeometries = {'EAS1': extractHeadGeometry(args.eas1_cdf, 'EAS1'),
                      'EAS2': extractHeadGeometry(args.eas2_cdf, 'EAS2')}
    saveHeadGeometry(args.out, headGeometries, [os.path.basename(args.eas1_cdf), os.path.basename(args.eas2_cdf)])
    geometry = loadHeadGeometry(args.out)
    unitVectors = pixelUnitVectorsSrf(geometry)
    print(f'wrote {args.out}: {unitVectors.shape[0]} pixels, '
          f'solid angle sum {pixelSolidAngles(geometry).sum():.3f} sr, '
          f'size {os.path.getsize(args.out) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
