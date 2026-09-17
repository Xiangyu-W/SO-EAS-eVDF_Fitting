"""Download the two-day example data set (2023-04-04 and 2023-04-05) of the
S3Park 2026 hands-on session from the GitHub release ``s3park-2026``.

Standard library only. The release carries eight six-hour preprocessed files
(about 600 MB each), their crossing break-energy CSVs, the reference fit as a
tarball and a ``SHA256SUMS`` file. This script reads ``SHA256SUMS`` first, so it
never needs to know the file names, downloads what you ask for, verifies every
file and skips files that are already complete.

    python workshop/download_example_days.py                      # both days + reference fit (~4.9 GB)
    python workshop/download_example_days.py --day 2023-04-04     # one day (~2.3 GB)
    python workshop/download_example_days.py --day 2023-04-04 --chunks 00   # one six-hour file
    python workshop/download_example_days.py --reference-only     # just the reference fit tarball

Then, from the repository root:

    PYTHONPATH=src python -m evdf_poisson.runner --data-dir data/example_days_2023-04-04_05 \\
        --out results/fitting/my_two_days --workers 8
"""
import argparse
import hashlib
import os
import sys
import tarfile
import time
import urllib.request

DEFAULT_BASE_URL = 'https://github.com/Xiangyu-W/SO-EAS-eVDF_Fitting/releases/download/s3park-2026/'
DEFAULT_DEST = os.path.join('data', 'example_days_2023-04-04_05')
CHECKSUM_FILE = 'SHA256SUMS'
REFERENCE_TARBALL = 'reference_fit_2023-04-04_05.tar.gz'
CHUNK_HOURS = ('00', '06', '12', '18')
RETRIES = 3
BLOCK_BYTES = 1 << 20


def fetch(url, destination, expectedSha=None):
    """Stream url to destination with a progress line; verify the sha256 when given."""
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(destination + '.part', 'wb') as target:
                total = int(response.headers.get('Content-Length') or 0)
                done = 0
                started = time.time()
                while True:
                    block = response.read(BLOCK_BYTES)
                    if not block:
                        break
                    target.write(block)
                    done += len(block)
                    if total:
                        rate = done / max(time.time() - started, 1e-3) / 1e6
                        sys.stdout.write(f'\r  {os.path.basename(destination)}: {done / 1e6:7.0f} / {total / 1e6:.0f} MB  ({rate:.1f} MB/s)')
                        sys.stdout.flush()
            sys.stdout.write('\n')
            if expectedSha and sha256Of(destination + '.part') != expectedSha:
                raise IOError('checksum mismatch')
            os.replace(destination + '.part', destination)
            return
        except Exception as error:   # network hiccups: retry, then give up loudly
            sys.stdout.write(f'\n  attempt {attempt}/{RETRIES} failed: {error}\n')
            time.sleep(5 * attempt)
    raise SystemExit(f'could not download {url}')


def sha256Of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(BLOCK_BYTES), b''):
            digest.update(block)
    return digest.hexdigest()


def readChecksums(path):
    checksums = {}
    with open(path) as source:
        for line in source:
            parts = line.split()
            if len(parts) == 2:
                checksums[parts[1].lstrip('*')] = parts[0]
    return checksums


def selectFiles(checksums, days, chunks, wantReference, referenceOnly):
    selected = []
    for name in sorted(checksums):
        if name == REFERENCE_TARBALL:
            if wantReference or referenceOnly:
                selected.append(name)
            continue
        if referenceOnly or not name.startswith('VDF_eas_forFitting_'):
            continue
        stamp = name.split('_')[3]                       # YYYYMMDDhhmmss of the file start
        day = f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}'
        if days and day not in days:
            continue
        if chunks and stamp[8:10] not in chunks:
            continue
        selected.append(name)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--day', action='append', choices=['2023-04-04', '2023-04-05'], help='repeatable; default both')
    parser.add_argument('--chunks', nargs='*', choices=CHUNK_HOURS, help='start hours of the six-hour files to take')
    parser.add_argument('--dest', default=DEFAULT_DEST)
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL)
    parser.add_argument('--no-reference', action='store_true', help='skip the reference fit tarball')
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    checksumPath = os.path.join(args.dest, CHECKSUM_FILE)
    fetch(args.base_url + CHECKSUM_FILE, checksumPath)
    checksums = readChecksums(checksumPath)
    files = selectFiles(checksums, args.day, args.chunks, not args.no_reference, args.reference_only)
    print(f'{len(files)} file(s) selected -> {args.dest}')
    for name in files:
        destination = os.path.join(args.dest, name)
        if os.path.exists(destination) and sha256Of(destination) == checksums[name]:
            print(f'  {name}: already complete')
            continue
        fetch(args.base_url + name, destination, checksums[name])
    if REFERENCE_TARBALL in files:
        with tarfile.open(os.path.join(args.dest, REFERENCE_TARBALL)) as archive:
            archive.extractall(os.path.join(args.dest, 'reference_fit'))
        print(f"  reference fit extracted to {os.path.join(args.dest, 'reference_fit')}")
    print('\nnext: PYTHONPATH=src python -m evdf_poisson.runner --data-dir', args.dest,
          '--out results/fitting/my_two_days --workers 8')


if __name__ == '__main__':
    main()
