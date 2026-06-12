from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from tqdm import tqdm

# Resolve paths from the script's real location, independent of the cwd
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "break_energy"
DEFAULT_MATPLOTLIB_CONFIG_DIR = DEFAULT_OUTPUT_DIR / ".matplotlib"
DEFAULT_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(DEFAULT_MATPLOTLIB_CONFIG_DIR)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))  # so data_io and utils import when run from anywhere

from data_io import convert_units, load_data
from utils import vel_to_eV

DEFAULT_PICKLE_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_INPUT_PATTERNS = ("VDF_eas_forFitting_*.pkl", "VDF_eas_forFitting_*.h5")
DEFAULT_CONE_HALF_WIDTH_DEG = 20.0
DEFAULT_WORKERS = min(32, os.cpu_count() or 1)
DEFAULT_FEATURES = ("v_perp", "log_psd", "deriv", "second_deriv")
DEFAULT_COMBINED_OUTPUT_NAME = "combined_cone_feature_split"

CONE_CENTER_DEG = 90.0
ENERGY_MAX_EV = 1000.0
LOW_ENERGY_BASE_EV = 12.0
MIN_COUNT = 1
MIN_PROFILE_LENGTH = 4
MIN_SEGMENT_SIZE = 3 # minimum Savitzky-Golay filter window length
TRANSITION_WINDOW_FRACTION = 0.5 # initial window length as a fraction of the profile length

# For smaller datasets (datalength<SMALL_PROFILE_LIMIT), don't let window get too large. Set it to be  min(window_length, SMALL_PROFILE_MAX_WINDOW)
SMALL_PROFILE_LIMIT = 100 
SMALL_PROFILE_MAX_WINDOW = 51 
LARGE_PROFILE_WINDOW = 101 # For larger datalength, use a fixed window length that is large enough
MIN_SAVGOL_WINDOW = 3

SUPPORTED_FEATURES = {
    "v_perp",
    "log_psd",
    "deriv",
    "second_deriv",
}

RESULT_COLUMNS = [
    "epoch_id",
    "timestamp",
    "break_v",
    "break_E_eV",
    "n_points",
    "PA_bins",
    "status",
    "features",
    "cone_half_width_deg",
    "input_file",
    "break_index",
    "error_message",
]
# epoch_id restarts from 0 in each input file, so the combined output needs its own global index.
COMBINED_EPOCH_COLUMN = "combined_epoch_id"


def allocateGridIndex(values: np.ndarray, gridBins: np.ndarray, rightClosed: bool = False) -> np.ndarray:
    return np.digitize(values, gridBins, right=rightClosed) - 1


def flattenPitchAngleVector(
    pitchAngleVector: np.ndarray,
    pitchAngleBins: np.ndarray,
    uncertaintyPitchAngleVector: np.ndarray,
) -> pd.DataFrame:
    pitchAngleCenters = np.diff(pitchAngleBins) / 2 + pitchAngleBins[:-1]
    pitchAngleBinIndices = allocateGridIndex(pitchAngleVector[:, :, 0], pitchAngleBins, rightClosed=True)
    validPitchAngleMask = (
        (pitchAngleBinIndices >= 0)
        & (pitchAngleBinIndices < len(pitchAngleCenters))
        & np.isfinite(pitchAngleVector[:, :, 0])
    )

    energyIndices, pixelIndices = np.nonzero(validPitchAngleMask)
    if len(energyIndices) == 0:
        return pd.DataFrame(
            columns=["pitch_angle", "PSD", "v_par", "v_perp", "count", "unc_psd", "energy_channel", "PA_bin"]
        )

    flattenedValues = pitchAngleVector[energyIndices, pixelIndices, :]
    uncertaintyValues = uncertaintyPitchAngleVector[energyIndices, pixelIndices]
    return pd.DataFrame(
        {
            "pitch_angle": flattenedValues[:, 0],
            "PSD": flattenedValues[:, 1],
            "v_par": flattenedValues[:, 2],
            "v_perp": flattenedValues[:, 3],
            "count": flattenedValues[:, 4],
            "unc_psd": uncertaintyValues,
            "energy_channel": energyIndices,
            "PA_bin": pitchAngleBinIndices[energyIndices, pixelIndices],
        }
    )


def getScpotValue(spacecraftPotential: object) -> float:
    potentialValue = getattr(spacecraftPotential, "values", spacecraftPotential)
    return float(np.asarray(potentialValue).ravel()[0])


def getEnergyChannelLimits(solarWindEnergy: np.ndarray, spacecraftPotential: float) -> tuple[int, int] | None:
    upperEnergyIndices = np.where(solarWindEnergy < ENERGY_MAX_EV)[0]
    lowerEnergyIndices = np.where(solarWindEnergy > (LOW_ENERGY_BASE_EV - spacecraftPotential))[0]
    if len(upperEnergyIndices) == 0 or len(lowerEnergyIndices) == 0:
        return None
    return int(upperEnergyIndices.min()), int(lowerEnergyIndices.max())


def getTransitionPreservingWindow(dataLength: int) -> int:
    windowLength = max(int(TRANSITION_WINDOW_FRACTION * dataLength), MIN_SAVGOL_WINDOW)
    if windowLength % 2 == 0:
        windowLength += 1 # Ensure window length is odd for savgol_filter

    if dataLength < SMALL_PROFILE_LIMIT:
        return min(windowLength, SMALL_PROFILE_MAX_WINDOW)
    return LARGE_PROFILE_WINDOW


def createConePixelProfile(
    epochIndex: int,
    pitchAngleVectors: np.ndarray,
    uncertaintyPitchAngleVectors: np.ndarray,
    pitchAngleBins: np.ndarray,
    solarWindEnergyValues: np.ndarray,
    spacecraftPotentials: np.ndarray,
    coneHalfWidthDeg: float,
) -> tuple[pd.DataFrame, str]:
    pitchAngleVector = pitchAngleVectors[epochIndex, :, :, :]
    if np.isnan(pitchAngleVector[:, :, 0]).all():
        return pd.DataFrame(), "all_nan"

    uncertaintyPitchAngleVector = uncertaintyPitchAngleVectors[epochIndex, :, :]
    solarWindEnergy = (
        solarWindEnergyValues[epochIndex]
        if solarWindEnergyValues.ndim == 2
        else solarWindEnergyValues
    )
    spacecraftPotential = getScpotValue(spacecraftPotentials[epochIndex])
    energyLimits = getEnergyChannelLimits(solarWindEnergy, spacecraftPotential)
    if energyLimits is None:
        return pd.DataFrame(), "no_energy_channel"

    energyStart, energyEnd = energyLimits
    pitchAngleDataFrame = flattenPitchAngleVector(
        pitchAngleVector,
        pitchAngleBins,
        uncertaintyPitchAngleVector,
    )
    selectedPixels = pitchAngleDataFrame[
        (pitchAngleDataFrame["count"] >= MIN_COUNT)
        & (np.abs(pitchAngleDataFrame["pitch_angle"] - CONE_CENTER_DEG) <= coneHalfWidthDeg)
        & pitchAngleDataFrame["energy_channel"].between(energyStart, energyEnd)
        & (pitchAngleDataFrame["PSD"] > 0)
        & (pitchAngleDataFrame["v_perp"] > 0)
    ].copy()
    if selectedPixels.empty:
        return pd.DataFrame(), "no_cone_pixels"

    selectedPixels = selectedPixels.sort_values("v_perp").reset_index(drop=True)
    selectedPixels["profile_psd"] = selectedPixels["PSD"]
    selectedPixels["log_psd"] = np.log10(selectedPixels["PSD"])
    return selectedPixels[
        [
            "v_perp",
            "v_par",
            "profile_psd",
            "log_psd",
            "energy_channel",
            "PA_bin",
            "pitch_angle",
            "count",
        ]
    ], "ok"


def getStandardizedFeatures(features: np.ndarray) -> np.ndarray:
    featureMeans = np.mean(features, axis=0)
    featureScales = np.std(features, axis=0)
    featureScales[featureScales == 0] = 1.0
    return (features - featureMeans) / featureScales


def getSingleVelocityTransitionLabels(
    features: np.ndarray,
    validMask: np.ndarray,
    minSegmentSize: int = MIN_SEGMENT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(len(features), -1, dtype=int)
    validIndices = np.where(validMask)[0]
    if len(validIndices) < 2 * minSegmentSize:
        return labels, np.array([], dtype=int)

    validFeatures = getStandardizedFeatures(features[validMask])
    featureSums = np.vstack([np.zeros(validFeatures.shape[1]), np.cumsum(validFeatures, axis=0)])
    squaredDistanceSums = np.concatenate(
        [
            [0.0],
            np.cumsum(np.sum(validFeatures ** 2, axis=1)),
        ]
    )

    splitCandidates = np.arange(minSegmentSize, len(validFeatures) - minSegmentSize + 1)
    leftSizes = splitCandidates
    rightSizes = len(validFeatures) - splitCandidates
    leftSums = featureSums[splitCandidates]
    rightSums = featureSums[-1] - leftSums
    leftSquaredSums = squaredDistanceSums[splitCandidates]
    rightSquaredSums = squaredDistanceSums[-1] - leftSquaredSums

    leftErrors = leftSquaredSums - np.sum(leftSums * leftSums, axis=1) / leftSizes
    rightErrors = rightSquaredSums - np.sum(rightSums * rightSums, axis=1) / rightSizes
    bestSplit = int(splitCandidates[int(np.argmin(leftErrors + rightErrors))])

    labels[validIndices[:bestSplit]] = 0
    labels[validIndices[bestSplit:]] = 1
    transitions = np.array([validIndices[bestSplit - 1]], dtype=int)
    return labels, transitions


def calculateTransitions(
    profile: pd.DataFrame,
    featureNames: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, str]:
    if len(profile) < MIN_PROFILE_LENGTH:
        emptyProfile = profile.assign(smoothed=np.nan, deriv=np.nan, second_deriv=np.nan, label=-1)
        return emptyProfile, np.array([], dtype=int), "too_few_points"

    logVelocity = np.log10(profile["v_perp"].to_numpy())
    logPhaseSpaceDensity = profile["log_psd"].to_numpy()
    velocitySpacing = np.mean(np.diff(logVelocity))
    if not np.isfinite(velocitySpacing) or velocitySpacing <= 0:
        emptyProfile = profile.assign(smoothed=np.nan, deriv=np.nan, second_deriv=np.nan, label=-1)
        return emptyProfile, np.array([], dtype=int), "invalid_log_velocity_spacing"

    windowLength = getTransitionPreservingWindow(len(logVelocity))
    maxWindowLength = len(logVelocity) if len(logVelocity) % 2 == 1 else len(logVelocity) - 1
    windowLength = min(windowLength, maxWindowLength)
    if windowLength < MIN_SAVGOL_WINDOW:
        emptyProfile = profile.assign(smoothed=np.nan, deriv=np.nan, second_deriv=np.nan, label=-1)
        return emptyProfile, np.array([], dtype=int), "too_few_points"

    derivative = savgol_filter(
        logPhaseSpaceDensity,
        windowLength,
        1,
        deriv=1,
        delta=velocitySpacing,
    )
    secondDerivative = savgol_filter(
        logPhaseSpaceDensity,
        windowLength,
        2,
        deriv=2,
        delta=velocitySpacing,
    )
    smoothedProfile = savgol_filter(logPhaseSpaceDensity, windowLength, 1, deriv=0)
    profileWithFeatures = profile.assign(
        smoothed=smoothedProfile,
        deriv=derivative,
        second_deriv=secondDerivative,
    )

    featureMatrix = np.column_stack(
        [profileWithFeatures[featureName].to_numpy(dtype=float) for featureName in featureNames]
    )
    validMask = np.isfinite(featureMatrix).all(axis=1)
    if validMask.sum() < 2 * MIN_SEGMENT_SIZE:
        profileWithFeatures = profileWithFeatures.assign(label=-1)
        return profileWithFeatures, np.array([], dtype=int), "no_valid_features"

    labels, transitions = getSingleVelocityTransitionLabels(featureMatrix, validMask)
    status = "ok" if len(transitions) > 0 else "no_transition"
    return profileWithFeatures.assign(label=labels), transitions, status


def summarizeTransition(profile: pd.DataFrame, transitions: np.ndarray) -> tuple[float, float, float]:
    if len(transitions) == 0:
        return np.nan, np.nan, np.nan

    breakIndex = int(transitions[0])
    breakVelocity = float(profile["v_perp"].iloc[breakIndex])
    breakEnergy = float(vel_to_eV(breakVelocity))
    return breakVelocity, breakEnergy, float(breakIndex)


def formatPaBins(profile: pd.DataFrame) -> str:
    if profile.empty or "PA_bin" not in profile:
        return ""

    paBins = sorted(profile["PA_bin"].dropna().astype(int).unique().tolist())
    return ",".join(str(paBin) for paBin in paBins)


def createResultRow(
    epochIndex: int,
    timestamp: object,
    inputFile: str,
    coneHalfWidthDeg: float,
    featureNames: Sequence[str],
    status: str,
    nPoints: int = 0,
    paBins: str = "",
    breakVelocity: float = np.nan,
    breakEnergy: float = np.nan,
    breakIndex: float = np.nan,
    errorMessage: str = "",
) -> dict[str, object]:
    return {
        "epoch_id": epochIndex,
        "timestamp": timestamp,
        "break_v": breakVelocity,
        "break_E_eV": breakEnergy,
        "n_points": nPoints,
        "PA_bins": paBins,
        "status": status,
        "features": ",".join(featureNames),
        "cone_half_width_deg": coneHalfWidthDeg,
        "input_file": inputFile,
        "break_index": breakIndex,
        "error_message": errorMessage,
    }


def processEpoch(
    epochIndex: int,
    pitchAngleVectors: np.ndarray,
    uncertaintyPitchAngleVectors: np.ndarray,
    pitchAngleBins: np.ndarray,
    solarWindEnergyValues: np.ndarray,
    spacecraftPotentials: np.ndarray,
    epochTimestamps: np.ndarray,
    coneHalfWidthDeg: float,
    featureNames: Sequence[str],
    inputFile: str,
) -> dict[str, object]:
    try:
        profile, profileStatus = createConePixelProfile(
            epochIndex,
            pitchAngleVectors,
            uncertaintyPitchAngleVectors,
            pitchAngleBins,
            solarWindEnergyValues,
            spacecraftPotentials,
            coneHalfWidthDeg,
        )
        if profileStatus != "ok":
            return createResultRow(
                epochIndex,
                epochTimestamps[epochIndex],
                inputFile,
                coneHalfWidthDeg,
                featureNames,
                profileStatus,
            )

        profileWithFeatures, transitions, transitionStatus = calculateTransitions(profile, featureNames)
        breakVelocity, breakEnergy, breakIndex = summarizeTransition(profileWithFeatures, transitions)
        paBins = formatPaBins(profileWithFeatures)
        return createResultRow(
            epochIndex,
            epochTimestamps[epochIndex],
            inputFile,
            coneHalfWidthDeg,
            featureNames,
            transitionStatus,
            nPoints=len(profileWithFeatures),
            paBins=paBins,
            breakVelocity=breakVelocity,
            breakEnergy=breakEnergy,
            breakIndex=breakIndex,
        )
    except Exception as error:
        return createResultRow(
            epochIndex,
            epochTimestamps[epochIndex],
            inputFile,
            coneHalfWidthDeg,
            featureNames,
            "error",
            errorMessage=f"{type(error).__name__}: {error}",
        )


def saveBreakEnergyPlot(
    resultDataFrame: pd.DataFrame,
    outputBasePath: Path,
    epochColumn: str = "epoch_id",
) -> Path | None:
    plotDataFrame = resultDataFrame[np.isfinite(resultDataFrame["break_E_eV"])].copy()
    if plotDataFrame.empty:
        print("No finite break energy values available for plotting.")
        return None

    matplotlibConfigDir = outputBasePath.parent / ".matplotlib"
    matplotlibConfigDir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlibConfigDir)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        plotDataFrame[epochColumn],
        plotDataFrame["break_E_eV"],
        marker="o",
        linestyle="-",
        linewidth=1.5,
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Break energy [eV]")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    plotOutputPath = outputBasePath.with_name(f"{outputBasePath.name}_break_energy.png")
    figure.savefig(plotOutputPath, dpi=200)
    plt.close(figure)
    print(f"Saved {plotOutputPath}")
    return plotOutputPath


def normalizeFeatureNames(rawFeatureValues: Sequence[str]) -> tuple[str, ...]:
    featureNames: list[str] = []
    for rawFeatureValue in rawFeatureValues:
        featureNames.extend(
            featureName.strip()
            for featureName in rawFeatureValue.split(",")
            if featureName.strip()
        )
    if not featureNames:
        raise ValueError("At least one feature is required.")

    unsupportedFeatures = sorted(set(featureNames) - SUPPORTED_FEATURES)
    if unsupportedFeatures:
        supportedFeatureText = ", ".join(sorted(SUPPORTED_FEATURES))
        unsupportedFeatureText = ", ".join(unsupportedFeatures)
        raise ValueError(
            f"Unsupported feature(s): {unsupportedFeatureText}. "
            f"Supported features: {supportedFeatureText}."
        )
    return tuple(featureNames)


def normalizeEpochIndices(rawEpochValues: Sequence[str] | None) -> tuple[int, ...] | None:
    if rawEpochValues is None:
        return None

    epochIndices: list[int] = []
    for rawEpochValue in rawEpochValues:
        for rawEpochIndex in rawEpochValue.split(","):
            epochText = rawEpochIndex.strip()
            if not epochText:
                continue
            try:
                epochIndex = int(epochText)
            except ValueError as error:
                raise ValueError(f"Invalid epoch index: {epochText}") from error
            if epochIndex < 0:
                raise ValueError(f"Epoch index must be non-negative: {epochIndex}")
            epochIndices.append(epochIndex)

    if not epochIndices:
        raise ValueError("At least one epoch index is required when --epoch-indices is used.")
    return tuple(sorted(set(epochIndices)))


def hasGlobPattern(pathText: str) -> bool:
    return any(patternCharacter in pathText for patternCharacter in "*?[")


def getInputFilePaths(pickleDir: Path, pickleFiles: Sequence[str] | None) -> list[Path]:
    if not pickleFiles:
        return sorted(
            inputPath
            for inputPattern in DEFAULT_INPUT_PATTERNS
            for inputPath in pickleDir.glob(inputPattern)
        )

    filePaths: list[Path] = []
    for pickleFile in pickleFiles:
        candidatePath = Path(pickleFile)
        if candidatePath.is_absolute():
            searchPath = candidatePath
        elif candidatePath.exists():
            searchPath = candidatePath
        else:
            searchPath = pickleDir / candidatePath

        if hasGlobPattern(str(searchPath)):
            filePaths.extend(sorted(searchPath.parent.glob(searchPath.name)))
        else:
            filePaths.append(searchPath)

    uniqueFilePaths = sorted({filePath.resolve() for filePath in filePaths})
    missingFiles = [filePath for filePath in uniqueFilePaths if not filePath.exists()]
    if missingFiles:
        missingText = "\n".join(str(filePath) for filePath in missingFiles)
        raise FileNotFoundError(f"Input file(s) not found:\n{missingText}")
    return uniqueFilePaths


def isStandardResultFile(resultPath: Path, combinedOutputStem: str) -> bool:
    resultName = resultPath.name
    return (
        resultName.endswith("_cone_feature_split.pkl")
        and "_epochs_" not in resultName
        and not resultName.startswith("combined_")
        and resultPath.stem != combinedOutputStem
    )


def getExistingResultPaths(outputDir: Path, combinedOutputStem: str) -> list[Path]:
    if not outputDir.exists():
        return []
    return sorted(
        resultPath
        for resultPath in outputDir.glob("*_cone_feature_split.pkl")
        if isStandardResultFile(resultPath, combinedOutputStem)
    )


def combineResultFiles(
    resultPaths: Sequence[Path],
    outputDir: Path,
    outputStem: str,
) -> Path:
    if not resultPaths:
        raise ValueError("No result files were provided for combining.")

    resultDataFrames = [pd.read_pickle(resultPath) for resultPath in resultPaths]
    combinedDataFrame = pd.concat(resultDataFrames, ignore_index=True, sort=False)
    combinedDataFrame = combinedDataFrame.reindex(columns=RESULT_COLUMNS)
    combinedDataFrame = combinedDataFrame.sort_values(
        ["timestamp", "input_file", "epoch_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    # Reassign a global index after sorting; duplicated boundary timestamps stay as separate rows.
    combinedDataFrame.insert(0, COMBINED_EPOCH_COLUMN, np.arange(len(combinedDataFrame)))

    outputDir.mkdir(parents=True, exist_ok=True)
    outputBasePath = outputDir / outputStem
    pickleOutputPath = outputBasePath.with_suffix(".pkl")
    csvOutputPath = outputBasePath.with_suffix(".csv")
    combinedDataFrame.to_pickle(pickleOutputPath)
    combinedDataFrame.to_csv(csvOutputPath, index=False)
    saveBreakEnergyPlot(combinedDataFrame, outputBasePath, epochColumn=COMBINED_EPOCH_COLUMN)
    print(f"Combined {len(resultPaths)} result files.")
    print(f"Saved {pickleOutputPath}")
    print(f"Saved {csvOutputPath}")
    return pickleOutputPath


def processFile(
    inputFilePath: Path,
    outputDir: Path,
    coneHalfWidthDeg: float,
    workerCount: int,
    featureNames: Sequence[str],
    maxEpochs: int | None,
    epochIndicesToProcess: Sequence[int] | None,
) -> Path:
    print(f"Processing file: {inputFilePath}")
    rawData = load_data(inputFilePath.parent, inputFilePath.name)
    convertedData = convert_units(rawData)

    pitchAngleVectors = convertedData["pa_vec_in"]
    uncertaintyPitchAngleVectors = convertedData["unc_pa_vec_in"]
    pitchAngleBins = convertedData["pitch_angles"]
    solarWindEnergyValues = convertedData["swa_energy"]
    spacecraftPotentials = np.asarray(
        getattr(convertedData["scpot_aligned"], "values", convertedData["scpot_aligned"])
    )
    epochTimestamps = convertedData["time_EAS"]

    epochCount = len(epochTimestamps)
    if epochIndicesToProcess is None:
        selectedEpochCount = epochCount if maxEpochs is None else min(maxEpochs, epochCount)
        epochIndices = tuple(range(selectedEpochCount))
    else:
        invalidEpochIndices = [
            epochIndex for epochIndex in epochIndicesToProcess if epochIndex >= epochCount
        ]
        if invalidEpochIndices:
            invalidEpochText = ", ".join(str(epochIndex) for epochIndex in invalidEpochIndices)
            raise IndexError(
                f"Epoch index out of range for {inputFilePath.name}: "
                f"{invalidEpochText}; valid range is 0-{epochCount - 1}."
            )
        epochIndices = tuple(epochIndicesToProcess)
        selectedEpochCount = len(epochIndices)

    taskIterator = (
        delayed(processEpoch)(
            epochIndex,
            pitchAngleVectors,
            uncertaintyPitchAngleVectors,
            pitchAngleBins,
            solarWindEnergyValues,
            spacecraftPotentials,
            epochTimestamps,
            coneHalfWidthDeg,
            featureNames,
            inputFilePath.name,
        )
        for epochIndex in epochIndices
    )
    parallelResults = Parallel(n_jobs=workerCount, return_as="generator_unordered")(taskIterator)
    resultRows = []
    progressDescription = f"Epochs in {inputFilePath.name}"
    with tqdm(total=selectedEpochCount, desc=progressDescription, unit="epoch") as progressBar:
        for resultRow in parallelResults:
            resultRows.append(resultRow)
            progressBar.update(1)

    resultDataFrame = pd.DataFrame(resultRows).sort_values("epoch_id").reset_index(drop=True)
    resultDataFrame = resultDataFrame.reindex(columns=RESULT_COLUMNS)

    outputDir.mkdir(parents=True, exist_ok=True)
    if epochIndicesToProcess is None:
        outputStem = f"{inputFilePath.stem}_cone_feature_split"
    else:
        epochSuffix = "_".join(str(epochIndex) for epochIndex in epochIndices)
        outputStem = f"{inputFilePath.stem}_cone_feature_split_epochs_{epochSuffix}"
    outputBasePath = outputDir / outputStem
    pickleOutputPath = outputBasePath.with_suffix(".pkl")
    csvOutputPath = outputBasePath.with_suffix(".csv")
    resultDataFrame.to_pickle(pickleOutputPath)
    resultDataFrame.to_csv(csvOutputPath, index=False)
    saveBreakEnergyPlot(resultDataFrame, outputBasePath)
    print(f"Saved {pickleOutputPath}")
    print(f"Saved {csvOutputPath}")
    return pickleOutputPath


def runBreakEnergyPipeline(
    inputFilePaths: Sequence[Path | str],
    outputDir: Path | str,
    coneHalfWidthDeg: float = DEFAULT_CONE_HALF_WIDTH_DEG,
    workerCount: int = DEFAULT_WORKERS,
    featureNames: Sequence[str] = DEFAULT_FEATURES,
    combinedOutputName: str = DEFAULT_COMBINED_OUTPUT_NAME,
    skipCombine: bool = False,
    maxEpochs: int | None = None,
    epochIndicesToProcess: Sequence[int] | None = None,
) -> list[Path]:
    """Run per-file break energy detection and combine the results.

    Shared entry point for the CLI and for preprocess_data_for_fit_hdf5, which
    calls it directly on freshly produced VDF files.
    """
    resultPaths = []
    for inputFilePath in inputFilePaths:
        resultPath = processFile(
            inputFilePath=Path(inputFilePath),
            outputDir=Path(outputDir),
            coneHalfWidthDeg=coneHalfWidthDeg,
            workerCount=workerCount,
            featureNames=featureNames,
            maxEpochs=maxEpochs,
            epochIndicesToProcess=epochIndicesToProcess,
        )
        resultPaths.append(resultPath)

    if not skipCombine:
        combineResultFiles(
            resultPaths=resultPaths,
            outputDir=Path(outputDir),
            outputStem=combinedOutputName,
        )
    return resultPaths


def createArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute cone-pixel break energy with configurable ordered-split features."
    )
    parser.add_argument(
        "--pickle-dir",
        type=Path,
        default=DEFAULT_PICKLE_DIR,
        help=f"Directory containing input pickle files. Default: {DEFAULT_PICKLE_DIR}",
    )
    parser.add_argument(
        "--pickle-files",
        nargs="*",
        default=None,
        help=(
            "Input file names, paths, or glob patterns (.pkl or .h5). "
            f"Default: {' and '.join(DEFAULT_INPUT_PATTERNS)} under --pickle-dir."
        ),
    )
    parser.add_argument(
        "--cone-half-width-deg",
        type=float,
        default=DEFAULT_CONE_HALF_WIDTH_DEG,
        help=f"Cone half width around {CONE_CENTER_DEG:g} deg. Default: {DEFAULT_CONE_HALF_WIDTH_DEG:g}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel worker count per file. Default: {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=[",".join(DEFAULT_FEATURES)],
        help=(
            "Comma- or space-separated feature names. "
            f"Default: {','.join(DEFAULT_FEATURES)}"
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Only process the first N epochs in each file for testing.",
    )
    parser.add_argument(
        "--epoch-indices",
        nargs="+",
        default=None,
        help="Specific epoch indices to process. Accepts space- or comma-separated values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for .pkl and .csv outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--combined-output-name",
        default=DEFAULT_COMBINED_OUTPUT_NAME,
        help=f"Output stem for combined .pkl, .csv, and plot. Default: {DEFAULT_COMBINED_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Do not create a combined result file after per-file processing.",
    )
    parser.add_argument(
        "--combine-existing",
        action="store_true",
        help="Only combine existing standard per-file result .pkl files in --output-dir.",
    )
    return parser


def runFromCli(argv: Sequence[str] | None = None) -> int:
    parser = createArgumentParser()
    args = parser.parse_args(argv)

    try:
        featureNames = normalizeFeatureNames(args.features)
    except ValueError as error:
        parser.error(str(error))
    try:
        epochIndicesToProcess = normalizeEpochIndices(args.epoch_indices)
    except ValueError as error:
        parser.error(str(error))

    if args.cone_half_width_deg < 0 or args.cone_half_width_deg > 180:
        parser.error("--cone-half-width-deg must be between 0 and 180.")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be at least 1.")
    if args.max_epochs is not None and epochIndicesToProcess is not None:
        parser.error("--max-epochs and --epoch-indices cannot be used together.")
    if args.combine_existing and args.skip_combine:
        parser.error("--combine-existing and --skip-combine cannot be used together.")
    if not args.combined_output_name:
        parser.error("--combined-output-name cannot be empty.")

    if args.combine_existing:
        resultPaths = getExistingResultPaths(args.output_dir, args.combined_output_name)
        if not resultPaths:
            parser.error(f"No standard result .pkl files found in {args.output_dir}.")

        print(f"Combining existing result files: {len(resultPaths)}")
        print(f"Output directory: {args.output_dir}")
        combineResultFiles(
            resultPaths=resultPaths,
            outputDir=args.output_dir,
            outputStem=args.combined_output_name,
        )
        return 0

    try:
        inputFilePaths = getInputFilePaths(args.pickle_dir, args.pickle_files)
    except FileNotFoundError as error:
        parser.error(str(error))

    if not inputFilePaths:
        parser.error(f"No input files matched {' or '.join(DEFAULT_INPUT_PATTERNS)} in {args.pickle_dir}.")

    print(f"Input files: {len(inputFilePaths)}")
    print(f"Cone half width: {args.cone_half_width_deg:g} deg")
    print(f"Workers: {args.workers}")
    print(f"Features: {','.join(featureNames)}")
    if epochIndicesToProcess is not None:
        print(f"Epoch indices: {','.join(str(epochIndex) for epochIndex in epochIndicesToProcess)}")
    print(f"Output directory: {args.output_dir}")

    runBreakEnergyPipeline(
        inputFilePaths=inputFilePaths,
        outputDir=args.output_dir,
        coneHalfWidthDeg=args.cone_half_width_deg,
        workerCount=args.workers,
        featureNames=featureNames,
        combinedOutputName=args.combined_output_name,
        skipCombine=args.skip_combine,
        maxEpochs=args.max_epochs,
        epochIndicesToProcess=epochIndicesToProcess,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(runFromCli())
