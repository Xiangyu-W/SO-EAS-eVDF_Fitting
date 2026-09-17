"""Download SO/SWA EAS nm3d CDF files from SOAR: L2 PSD and L1 counts.

SOAR names these files like:

    solo_L2_swa-eas2-nm3d-psd_20221228T000004-20221228T060004_V03.cdf
    solo_L1_swa-eas2-nm3d_20230401T000008-20230401T060008_V03.cdf

Both products are handled by the same code path, because they share the failure
mode that motivates it: SOAR republishes a window under a higher version with the
period boundaries shifted by a few seconds, so a dedup keyed on the exact period
string keeps both copies, and V01 stores the energy on a different dimension than
later versions, which breaks the downstream concat. Overlap clustering identifies
"the same window" regardless of the shifted boundaries, and only the newest version
survives -- for downloads and for reuse of files already on disk.

Run directly to test a short time range.
"""

import os
import re
import time
import pandas as pd
from requests.exceptions import RequestException

# SOAR product descriptors for the two EAS sensors, per level.
EAS1_PSD_PRODUCT = 'swa-eas1-nm3d-psd'
EAS2_PSD_PRODUCT = 'swa-eas2-nm3d-psd'
EAS1_COUNT_PRODUCT = 'swa-eas1-nm3d'
EAS2_COUNT_PRODUCT = 'swa-eas2-nm3d'
L2_PSD_LEVEL = 2
L1_COUNT_LEVEL = 1

# Minimum search window so SOAR returns at least one 6-hour file.
DEFAULT_MIN_SEARCH_HOURS = 6

# Coverage shorter than this is not worth reporting as a hole: consecutive files
# are separated by a ~10 s publishing gap, and EAS itself drops short stretches.
DEFAULT_COVERAGE_TOLERANCE = pd.Timedelta(minutes=10)

# SOAR's TAP server intermittently returns HTTP 408 on multi-day queries, so
# searches are split per day and retried a few times before giving up.
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 5

# Filename patterns: full-precision time period and trailing version.
TIME_PERIOD_PATTERN = re.compile(r'(\d{8}T\d{6}-\d{8}T\d{6})')
VERSION_PATTERN = re.compile(r'V(\d+)\.cdf$')
FILE_PERIOD_FORMAT = '%Y%m%dT%H%M%S'


def parseFilePeriod(fileName):
    """Return (startTime, endTime, version) from a SOAR PSD filename, or None."""
    timeMatch = TIME_PERIOD_PATTERN.search(fileName)
    versionMatch = VERSION_PATTERN.search(fileName)
    if not timeMatch or not versionMatch:
        return None

    periodStart, periodEnd = timeMatch.group(1).split('-')
    return (pd.to_datetime(periodStart, format=FILE_PERIOD_FORMAT),
            pd.to_datetime(periodEnd, format=FILE_PERIOD_FORMAT),
            int(versionMatch.group(1)))


def groupOverlappingPeriods(fileEntries):
    """Cluster entries whose [start, end] windows overlap in time.

    SOAR republishes the same 6-hour window under a new version with boundaries
    shifted by a few seconds, so matching on the exact period string treats V01
    and V03 of one window as two distinct periods. Overlap is what actually
    identifies "the same time period". Files of a single version never overlap
    each other (consecutive windows are separated by a ~10 s gap), so a cluster
    only ever collects different versions of the same window.

    Returns a list of index lists into fileEntries.
    """
    orderedIndices = sorted(range(len(fileEntries)),
                            key=lambda index: fileEntries[index]['start'])

    clusters = []
    currentCluster = []
    currentClusterEnd = None
    for index in orderedIndices:
        entry = fileEntries[index]
        if currentCluster and entry['start'] < currentClusterEnd:
            currentCluster.append(index)
            currentClusterEnd = max(currentClusterEnd, entry['end'])
        else:
            if currentCluster:
                clusters.append(currentCluster)
            currentCluster = [index]
            currentClusterEnd = entry['end']
    if currentCluster:
        clusters.append(currentCluster)

    return clusters


def selectLatestVersionAcrossResults(soarTables):
    """Keep only the newest version of every overlapping time window.

    Runs across all daily query results at once, which removes two kinds of
    duplicate the per-day filter cannot see: the same file returned by two
    adjacent daily queries (windows straddle midnight), and an old-version file
    from one day overlapping a new-version file first seen on the next.

    All files of the winning version inside a cluster are kept, so a genuine
    pair of same-version overlapping windows is never silently halved.

    Returns a list of filtered tables, with fully emptied ones removed.
    """
    fileEntries = []
    for tableIndex, soarTable in enumerate(soarTables):
        for rowIndex, filename in enumerate(soarTable['Filename']):
            filenameStr = str(filename)
            parsedPeriod = parseFilePeriod(filenameStr)
            if parsedPeriod is None:
                print(f"Skipping unrecognised PSD filename: {filenameStr}")
                continue
            startTime, endTime, version = parsedPeriod
            fileEntries.append({
                'tableIndex': tableIndex,
                'rowIndex': rowIndex,
                'filename': filenameStr,
                'start': startTime,
                'end': endTime,
                'version': version,
            })

    if not fileEntries:
        print("No versioned PSD files matched the expected naming.")
        return []

    keptFilenames = set()
    keptRowsByTable = {}
    for cluster in groupOverlappingPeriods(fileEntries):
        latestVersion = max(fileEntries[index]['version'] for index in cluster)

        for index in cluster:
            entry = fileEntries[index]
            if entry['version'] != latestVersion:
                print(f"  Superseded by V{latestVersion:02d}, skipping: {entry['filename']}")
                continue
            if entry['filename'] in keptFilenames:
                continue  # same file returned by two adjacent daily queries
            keptFilenames.add(entry['filename'])
            keptRowsByTable.setdefault(entry['tableIndex'], []).append(entry['rowIndex'])

    filteredTables = [soarTables[tableIndex][sorted(rowIndices)]
                      for tableIndex, rowIndices in sorted(keptRowsByTable.items())]

    print(f"Version/overlap filter: {len(fileEntries)} candidate file(s) "
          f"reduced to {len(keptFilenames)} download(s)")
    return filteredTables


def adjustTimeRange(timeRange, minHours=DEFAULT_MIN_SEARCH_HOURS):
    """Pad a short time range symmetrically so at least one file is locatable."""
    startTime = pd.to_datetime(timeRange[0])
    endTime = pd.to_datetime(timeRange[-1])
    timeSpan = endTime - startTime

    if timeSpan < pd.Timedelta(hours=minHours):
        padding = (pd.Timedelta(hours=minHours) - timeSpan) / 2
        startTime = startTime - padding
        endTime = endTime + padding

    return startTime.strftime('%Y-%m-%d %H:%M'), endTime.strftime('%Y-%m-%d %H:%M')


def extractSoarTable(searchResult):
    """Return the SOAR result table from a Fido search, or None if it has none."""
    if 'soar' not in searchResult.keys():
        return None
    return searchResult['soar']


def splitIntoDailyRanges(timeRange):
    """Break a time range into consecutive per-day (start, end) pairs.

    Each SOAR metadata query then spans at most one day, which keeps the
    server-side query fast enough to avoid the HTTP 408 timeouts seen on
    multi-day queries. Interior boundaries are day-aligned; the first and last
    pair honour the exact requested start and end.
    """
    startTime = pd.to_datetime(timeRange[0])
    endTime = pd.to_datetime(timeRange[-1])

    dayBoundaries = [startTime]
    nextMidnight = startTime.normalize() + pd.Timedelta(days=1)
    while nextMidnight < endTime:
        dayBoundaries.append(nextMidnight)
        nextMidnight += pd.Timedelta(days=1)
    dayBoundaries.append(endTime)

    return [(dayBoundaries[i], dayBoundaries[i + 1])
            for i in range(len(dayBoundaries) - 1)]


def searchSoarWithRetry(*queryAttrs, maxRetries=DEFAULT_MAX_RETRIES,
                        retryDelaySeconds=DEFAULT_RETRY_DELAY_SECONDS):
    """Run Fido.search, retrying transient SOAR failures such as HTTP 408.

    Returns the search result, or re-raises the last error once retries are
    exhausted.
    """
    from sunpy.net import Fido

    for attempt in range(1, maxRetries + 1):
        try:
            return Fido.search(*queryAttrs)
        except RequestException as error:
            if attempt == maxRetries:
                print(f"  SOAR search failed after {maxRetries} attempts: {error}")
                raise
            print(f"  SOAR search attempt {attempt}/{maxRetries} failed "
                  f"({error}); retrying in {retryDelaySeconds}s...")
            time.sleep(retryDelaySeconds)


def getDefaultDataDir(subDirectory):
    """Absolute path of data/<subDirectory>, resolved from this file's location."""
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(scriptDir, '..', 'data', subDirectory))


def sortByFilePeriod(filePaths):
    """Order file paths by the start time encoded in their name.

    readCDFs concatenates in list order, so a fetch that returns paths in
    completion order would build a non-monotonic time axis.
    """
    def sortKey(filePath):
        parsedPeriod = parseFilePeriod(os.path.basename(filePath))
        return parsedPeriod[0] if parsedPeriod else pd.Timestamp.max

    return sorted(filePaths, key=sortKey)


def downloadSoarEasNm3d(timeRange, downloadDir, level, eas1Product, eas2Product,
                        productLabel, maxRetries=DEFAULT_MAX_RETRIES,
                        retryDelaySeconds=DEFAULT_RETRY_DELAY_SECONDS):
    """Search SOAR per day and download the EAS1/EAS2 files of one product for timeRange.

    The range is split into per-day queries, each retried on transient failures,
    to dodge SOAR's multi-day 408 timeouts. All located files are fetched in a
    single Fido.fetch call per sensor, so parfive downloads them concurrently.
    Searches stay sequential on purpose: firing them in parallel would recreate
    the load spike that triggers the 408s in the first place.

    Files already present in downloadDir are not re-downloaded (parfive skips
    them) but are still returned, so this is cheap to call on a partly-filled
    directory.

    Returns (eas1FilePaths, eas2FilePaths), time-ordered.
    """
    from sunpy.net import Fido, attrs as a
    import sunpy_soar  # noqa: F401  (registers the SOAR client with Fido)

    os.makedirs(downloadDir, exist_ok=True)

    dailyRanges = splitIntoDailyRanges(timeRange)
    print(f"Split request into {len(dailyRanges)} daily SOAR queries")

    eas1Tables = []
    eas2Tables = []
    for dayStart, dayEnd in dailyRanges:
        searchStart, searchEnd = adjustTimeRange([dayStart, dayEnd])
        print(f"Searching SOAR for {productLabel} between {searchStart} and {searchEnd}")

        resultEas1 = searchSoarWithRetry(
            a.Instrument('SWA'), a.Time(searchStart, searchEnd),
            a.Level(level), a.soar.Product(eas1Product),
            maxRetries=maxRetries, retryDelaySeconds=retryDelaySeconds)
        resultEas2 = searchSoarWithRetry(
            a.Instrument('SWA'), a.Time(searchStart, searchEnd),
            a.Level(level), a.soar.Product(eas2Product),
            maxRetries=maxRetries, retryDelaySeconds=retryDelaySeconds)

        soarTableEas1 = extractSoarTable(resultEas1)
        soarTableEas2 = extractSoarTable(resultEas2)
        if soarTableEas1 is not None and len(soarTableEas1) > 0:
            eas1Tables.append(soarTableEas1)
        if soarTableEas2 is not None and len(soarTableEas2) > 0:
            eas2Tables.append(soarTableEas2)

    # Filter after all searches: duplicates also straddle daily query boundaries.
    print("Resolving EAS1 versions across every daily result")
    eas1Results = selectLatestVersionAcrossResults(eas1Tables)
    print("Resolving EAS2 versions across every daily result")
    eas2Results = selectLatestVersionAcrossResults(eas2Tables)

    eas1FileCount = sum(len(result) for result in eas1Results)
    eas2FileCount = sum(len(result) for result in eas2Results)
    print(f"Found {eas1FileCount} EAS1 and {eas2FileCount} EAS2 {productLabel} files "
          f"across {len(dailyRanges)} day(s)")

    if not eas1Results or not eas2Results:
        print("Warning: no EAS1 or EAS2 results found for the requested range")
        return [], []

    # One fetch per sensor: parfive downloads every day's files concurrently.
    eas1FilePaths = Fido.fetch(*eas1Results, path=os.path.join(downloadDir, '{file}'))
    eas2FilePaths = Fido.fetch(*eas2Results, path=os.path.join(downloadDir, '{file}'))

    return sortByFilePeriod(eas1FilePaths), sortByFilePeriod(eas2FilePaths)


def downloadSoL2PsdNm3d(timeRange, downloadDir=None,
                        maxRetries=DEFAULT_MAX_RETRIES,
                        retryDelaySeconds=DEFAULT_RETRY_DELAY_SECONDS):
    """Download the EAS1/EAS2 L2 PSD files covering timeRange."""
    if downloadDir is None:
        downloadDir = getDefaultDataDir('vdfs')

    return downloadSoarEasNm3d(timeRange, downloadDir, L2_PSD_LEVEL,
                               EAS1_PSD_PRODUCT, EAS2_PSD_PRODUCT, 'L2 PSD',
                               maxRetries=maxRetries,
                               retryDelaySeconds=retryDelaySeconds)


def downloadSoL1CountNm3d(timeRange, downloadDir=None,
                          maxRetries=DEFAULT_MAX_RETRIES,
                          retryDelaySeconds=DEFAULT_RETRY_DELAY_SECONDS):
    """Download the EAS1/EAS2 L1 count files covering timeRange.

    Counts carry the Poisson uncertainty of the L2 PSD, so they must span the
    same window and the same version as the PSD files they annotate.
    """
    if downloadDir is None:
        downloadDir = getDefaultDataDir('L1_Count')

    return downloadSoarEasNm3d(timeRange, downloadDir, L1_COUNT_LEVEL,
                               EAS1_COUNT_PRODUCT, EAS2_COUNT_PRODUCT, 'L1 count',
                               maxRetries=maxRetries,
                               retryDelaySeconds=retryDelaySeconds)


def discoverLocalEasNm3d(timeRange, downloadDir, eas1Product, eas2Product, productLabel):
    """Find already-downloaded EAS1/EAS2 files of one product overlapping timeRange.

    Mirrors downloadSoarEasNm3d but reads from disk instead of querying SOAR, and
    applies the same overlap-cluster version filter: files whose windows overlap are
    the same period republished, so only the newest version of each cluster is kept.

    V01 stores the energy on a different dimension than later versions, so a mixed-
    version set breaks the downstream concat even when the windows do not overlap.
    If more than one version survives, only the highest is kept and the dropped files
    are logged rather than silently mixed.

    Returns (eas1FilePaths, eas2FilePaths), time-ordered.
    """
    if not os.path.isdir(downloadDir):
        print(f"Warning: local {productLabel} directory does not exist: {downloadDir}")
        return [], []

    rangeStart = pd.to_datetime(timeRange[0])
    rangeEnd = pd.to_datetime(timeRange[-1])
    availableFiles = os.listdir(downloadDir)

    def collectSensor(product):
        # The trailing underscore keeps 'swa-eas1-nm3d' from matching
        # 'swa-eas1-nm3d-psd'; early L1 files spell the product in upper case.
        fileToken = f'{product}_'.lower()
        fileEntries = []
        for fileName in availableFiles:
            if fileToken not in fileName.lower() or not fileName.endswith('.cdf'):
                continue
            parsedPeriod = parseFilePeriod(fileName)
            if parsedPeriod is None:
                continue
            fileStart, fileEnd, version = parsedPeriod
            if fileEnd < rangeStart or fileStart > rangeEnd:
                continue  # no overlap with the requested range
            fileEntries.append({'filename': fileName, 'start': fileStart,
                                'end': fileEnd, 'version': version})

        keptEntries = []
        for cluster in groupOverlappingPeriods(fileEntries):
            latestVersion = max(fileEntries[index]['version'] for index in cluster)
            for index in cluster:
                entry = fileEntries[index]
                if entry['version'] == latestVersion:
                    keptEntries.append(entry)
                else:
                    print(f"  Superseded by V{latestVersion:02d}, skipping: {entry['filename']}")

        # Enforce a single version across the whole sensor set.
        versionsPresent = {entry['version'] for entry in keptEntries}
        if len(versionsPresent) > 1:
            keepVersion = max(versionsPresent)
            dropped = sorted(entry['filename'] for entry in keptEntries
                             if entry['version'] != keepVersion)
            print(f"Warning: {product} local files mix versions {sorted(versionsPresent)}; "
                  f"keeping only V{keepVersion:02d} and dropping {len(dropped)} older-version "
                  f"file(s) to avoid an energy-dimension concat failure:")
            for name in dropped:
                print(f"  dropped {name}")
            keptEntries = [entry for entry in keptEntries if entry['version'] == keepVersion]

        return [os.path.join(downloadDir, entry['filename'])
                for entry in sorted(keptEntries, key=lambda entry: entry['start'])]

    eas1FilePaths = collectSensor(eas1Product)
    eas2FilePaths = collectSensor(eas2Product)
    print(f"Found {len(eas1FilePaths)} EAS1 and {len(eas2FilePaths)} EAS2 local {productLabel} "
          f"files overlapping {rangeStart} to {rangeEnd} in {downloadDir}")
    return eas1FilePaths, eas2FilePaths


def discoverLocalL2Psd(timeRange, downloadDir=None):
    """Find already-downloaded EAS1/EAS2 L2 PSD files overlapping timeRange."""
    if downloadDir is None:
        downloadDir = getDefaultDataDir('vdfs')

    return discoverLocalEasNm3d(timeRange, downloadDir,
                                EAS1_PSD_PRODUCT, EAS2_PSD_PRODUCT, 'L2 PSD')


def discoverLocalL1Count(timeRange, downloadDir=None):
    """Find already-downloaded EAS1/EAS2 L1 count files overlapping timeRange."""
    if downloadDir is None:
        downloadDir = getDefaultDataDir('L1_Count')

    return discoverLocalEasNm3d(timeRange, downloadDir,
                                EAS1_COUNT_PRODUCT, EAS2_COUNT_PRODUCT, 'L1 count')


def findUncoveredIntervals(timeRange, filePaths, tolerance=DEFAULT_COVERAGE_TOLERANCE):
    """Return the [start, end] stretches of timeRange no file in filePaths covers.

    Used to decide whether a local file set can stand in for a SOAR query: a set
    that silently covers only part of the window would truncate the run, because
    the pipeline intersects the PSD and count epochs.
    """
    rangeStart = pd.to_datetime(timeRange[0])
    rangeEnd = pd.to_datetime(timeRange[-1])

    coveredPeriods = sorted(
        (parsed[0], parsed[1]) for parsed in
        (parseFilePeriod(os.path.basename(filePath)) for filePath in filePaths)
        if parsed is not None)

    uncoveredIntervals = []
    cursor = rangeStart
    for periodStart, periodEnd in coveredPeriods:
        if periodStart > cursor + tolerance:
            uncoveredIntervals.append((cursor, min(periodStart, rangeEnd)))
        cursor = max(cursor, periodEnd)
        if cursor >= rangeEnd:
            break

    if cursor < rangeEnd - tolerance:
        uncoveredIntervals.append((cursor, rangeEnd))

    return [(start, end) for start, end in uncoveredIntervals if end - start > tolerance]


if __name__ == "__main__":
    # Short test window around the example file the user provided.
    testTimeRange = ['2022-12-28 00:00:00', '2022-12-28 06:00:00']

    eas1Files, eas2Files = downloadSoL2PsdNm3d(testTimeRange)

    print("\nEAS1 L2 PSD files:")
    for path in eas1Files:
        print(f"  {path}")
    print("\nEAS2 L2 PSD files:")
    for path in eas2Files:
        print(f"  {path}")
