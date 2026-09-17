"""Candidate methods for locating the core/halo and core/secondary boundaries of an eVDF.

Every ``find*`` function takes an already-built profile and returns a result dict with
``break_E_eV``, ``status`` and method-specific diagnostics, so one epoch can be pushed
through all of them and the answers compared side by side.

Three ideas separate the alternatives from the production change-point method:

* Pixels below the one-count level are stored with ``PSD = 0`` and the production filter
  ``count >= 1`` drops them. Averaging only the survivors is a censored mean, biased high
  wherever the cone is not fully detected - which is both the lowest channels and the
  suprathermal tail. Averaging *all* cone pixels in linear space, zeros included, is
  unbiased, and that is what aggregateByEnergyChannel does.
* The cone holds ~400 pixels per energy channel whose v_perp values interleave across
  neighbouring channels, so a per-pixel derivative measures pitch-angle scatter more than
  the shape of the VDF. One weighted point per channel gives the fits a clean 1D VDF.
* A break between two populations is a statement about their functional forms, so it is
  better found by fitting those forms than by clustering in a feature space where monotone
  coordinates dominate the variance.

Nothing here mutates the production pipeline: the baseline change-point machinery is
imported from compute_break_energy_cone_features and reused as-is.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit
from scipy.signal import savgol_filter

from compute_break_energy_cone_features import (
    CONE_CENTER_DEG,
    DEFAULT_FEATURES,
    ENERGY_MAX_EV,
    MIN_COUNT,
    MIN_SAVGOL_WINDOW,
    MIN_SEGMENT_SIZE,
    flattenPitchAngleVector,
    getSingleVelocityTransitionLabels,
    getTransitionPreservingWindow,
)
from utils import vel_to_eV

NATURAL_LOG_OF_TEN = np.log(10.0)

# The production cone profile starts at 12 eV instrument energy, below which spacecraft
# secondary and photo electrons dominate.
BASELINE_LOW_ENERGY_EV = 12.0

# A channel enters the fits once its cone-averaged f is measured with this signal-to-noise;
# below it the channel is at the one-count level and carries no usable shape information.
MIN_CHANNEL_SIGNAL_TO_NOISE = 3.0
# Pixel-to-pixel scatter within a channel is real anisotropy, not just counting noise, so
# the weights are floored to keep a few thousand pixels from driving the fits.
MIN_LOG_PSD_SIGMA = 0.01

MIN_SEGMENT_CHANNELS = 4
MIN_TWO_POPULATION_CHANNELS = 8

# A Maxwellian core exceeded by a factor of two is exactly where the suprathermal
# population equals the core, i.e. the crossing that defines the core/halo break.
DEFAULT_CORE_HALO_RATIO = 2.0
DEFAULT_SIGMA_THRESHOLD = 3.0
DEFAULT_CONSECUTIVE_CHANNELS = 2
# Matches the production low-energy cut of 12 eV instrument energy once a spacecraft
# potential of roughly 10 V has been added; callers should pass 12 - scpot explicitly.
DEFAULT_CORE_FIT_LOW_EV = 22.0
DEFAULT_CORE_FIT_HIGH_EV = 45.0

DEFAULT_VALLEY_SEARCH_MAX_EV = 40.0
DEFAULT_VALLEY_PROMINENCE_DEX = 0.05
DEFAULT_VALLEY_LOOK_AHEAD = 3
VALLEY_SMOOTHING_WINDOW = 7
VALLEY_SMOOTHING_POLYORDER = 2

PLATEAU_TOLERANCE = 1.45
MIN_PLATEAU_CHANNELS = 5


def getEnergyRow(data: dict, epochIndex: int) -> np.ndarray:
    """Return the energy table for one epoch, handling both the 1D and per-epoch 2D layouts."""
    energyTable = np.asarray(data["swa_energy"])
    return energyTable[epochIndex] if energyTable.ndim == 2 else energyTable


def buildConeProfile(
    epochIndex: int,
    data: dict,
    coneHalfWidthDeg: float,
    minEnergyEv: float | None = None,
    maxEnergyEv: float = ENERGY_MAX_EV,
) -> tuple[pd.DataFrame, str]:
    """Every cone pixel in the energy window, zero-count pixels included.

    Zero-count pixels are kept because dropping them is what biases the channel means; the
    is_detected column marks them so the per-pixel methods can reproduce the production
    selection exactly while the channel aggregation can average over the whole cone.
    """
    pitchAngleVector = data["pa_vec_in"][epochIndex]
    if np.isnan(pitchAngleVector[:, :, 0]).all():
        return pd.DataFrame(), "all_nan"

    energyRow = getEnergyRow(data, epochIndex)
    pixels = flattenPitchAngleVector(
        pitchAngleVector,
        data["pitch_angles"],
        data["unc_pa_vec_in"][epochIndex],
    )
    if pixels.empty:
        return pd.DataFrame(), "no_pixels"

    channelEnergies = energyRow[pixels["energy_channel"].to_numpy()]
    isInEnergyRange = np.isfinite(channelEnergies) & (channelEnergies <= maxEnergyEv)
    if minEnergyEv is not None:
        isInEnergyRange &= channelEnergies >= minEnergyEv
    isInCone = np.abs(pixels["pitch_angle"].to_numpy() - CONE_CENTER_DEG) <= coneHalfWidthDeg

    conePixels = pixels[pd.Series(isInEnergyRange & isInCone, index=pixels.index)].copy()
    if conePixels.empty:
        return pd.DataFrame(), "no_cone_pixels"

    conePixels["channel_E_eV"] = energyRow[conePixels["energy_channel"].to_numpy()]
    conePixels["is_detected"] = (
        (conePixels["count"] >= MIN_COUNT) & (conePixels["PSD"] > 0) & (conePixels["v_perp"] > 0)
    )
    if not conePixels["is_detected"].any():
        return pd.DataFrame(), "no_detected_pixels"

    conePixels["E_eV"] = vel_to_eV(conePixels["v_perp"].to_numpy())
    return conePixels.sort_values("v_perp").reset_index(drop=True), "ok"


def getDetectedPixels(profile: pd.DataFrame) -> pd.DataFrame:
    """The pixels the production pipeline sees: at least one count, with the columns it needs."""
    detectedPixels = profile[profile["is_detected"]].copy()
    detectedPixels["profile_psd"] = detectedPixels["PSD"]
    detectedPixels["log_psd"] = np.log10(detectedPixels["PSD"])
    return detectedPixels.sort_values("v_perp").reset_index(drop=True)


def aggregateByEnergyChannel(profile: pd.DataFrame) -> pd.DataFrame:
    """Cone-averaged f per energy channel, in ascending energy.

    The mean is taken in linear space over every cone pixel, zero-count pixels included, so
    it is an unbiased estimate of the cone-averaged phase space density. detection_fraction
    records how much of the cone was above one count, and is_reliable marks the channels
    whose mean is resolved at MIN_CHANNEL_SIGNAL_TO_NOISE or better.
    """
    channelColumns = [
        "energy_channel",
        "E_eV",
        "psd",
        "log_psd",
        "sigma_log_psd",
        "signal_to_noise",
        "n_pixels",
        "detection_fraction",
        "total_count",
        "is_reliable",
    ]
    if profile.empty:
        return pd.DataFrame(columns=channelColumns)

    channelRows = []
    for energyChannel, channelPixels in profile.groupby("energy_channel"):
        phaseSpaceDensity = channelPixels["PSD"].to_numpy(dtype=float)
        pixelCount = len(phaseSpaceDensity)
        meanPsd = float(np.mean(phaseSpaceDensity))
        if meanPsd <= 0:
            continue

        standardError = (
            float(np.std(phaseSpaceDensity, ddof=1) / np.sqrt(pixelCount)) if pixelCount > 1 else meanPsd
        )
        signalToNoise = meanPsd / standardError if standardError > 0 else np.inf
        sigmaLogPsd = max(standardError / (meanPsd * NATURAL_LOG_OF_TEN), MIN_LOG_PSD_SIGMA)

        channelRows.append(
            {
                "energy_channel": int(energyChannel),
                "E_eV": float(channelPixels["channel_E_eV"].iloc[0]),
                "psd": meanPsd,
                "log_psd": float(np.log10(meanPsd)),
                "sigma_log_psd": sigmaLogPsd,
                "signal_to_noise": float(signalToNoise),
                "n_pixels": pixelCount,
                "detection_fraction": float(channelPixels["is_detected"].mean()),
                "total_count": float(channelPixels["count"].sum()),
                "is_reliable": bool(signalToNoise >= MIN_CHANNEL_SIGNAL_TO_NOISE),
            }
        )

    if not channelRows:
        return pd.DataFrame(columns=channelColumns)
    return pd.DataFrame(channelRows, columns=channelColumns).sort_values("E_eV").reset_index(drop=True)


def getReliableChannels(
    channelProfile: pd.DataFrame,
    minEnergyEv: float | None = None,
    maxEnergyEv: float = ENERGY_MAX_EV,
) -> pd.DataFrame:
    """The longest unbroken run of well-measured channels inside the requested energy window.

    A contiguous run matters more than a mask: the fits and the derivative need neighbouring
    channels, and a hole punched by one noisy channel would silently join two disjoint
    energy ranges into what looks like a single profile.
    """
    selected = channelProfile[channelProfile["E_eV"] <= maxEnergyEv]
    if minEnergyEv is not None:
        selected = selected[selected["E_eV"] >= minEnergyEv]
    selected = selected.reset_index(drop=True)
    if selected.empty:
        return selected

    isReliable = selected["is_reliable"].to_numpy()
    bestStart, bestLength, runStart = 0, 0, None
    for channelIndex in range(len(isReliable) + 1):
        if channelIndex < len(isReliable) and isReliable[channelIndex]:
            runStart = channelIndex if runStart is None else runStart
            continue
        if runStart is not None:
            runLength = channelIndex - runStart
            if runLength > bestLength:
                bestStart, bestLength = runStart, runLength
            runStart = None

    return selected.iloc[bestStart : bestStart + bestLength].reset_index(drop=True)


def getChannelLogSlope(channelProfile: pd.DataFrame) -> np.ndarray:
    """d log10(f) / d log10(E) on the channel-aggregated profile."""
    if len(channelProfile) < 2:
        return np.full(len(channelProfile), np.nan)
    return np.gradient(
        channelProfile["log_psd"].to_numpy(),
        np.log10(channelProfile["E_eV"].to_numpy()),
    )


def getLocalTemperature(channelProfile: pd.DataFrame) -> np.ndarray:
    """Temperature a Maxwellian would need to produce the locally observed slope, per channel.

    For log10(f) = A - E/(kT ln10) the log-log slope is -E/kT, so kT_local = -E / slope. A
    Maxwellian core is therefore a flat plateau and a power-law tail a straight rise, and the
    core/halo break is where the plateau ends. Nothing is fitted, so this is the fairest
    referee available for the methods below.
    """
    logSlope = getChannelLogSlope(channelProfile)
    energy = channelProfile["E_eV"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        localTemperature = -energy / logSlope
    return np.where(np.isfinite(localTemperature) & (localTemperature > 0), localTemperature, np.nan)


def findTemperaturePlateau(channelProfile: pd.DataFrame) -> dict[str, object]:
    """Longest energy range over which kT_local stays flat: the range where the core alone rules."""
    localTemperature = getLocalTemperature(channelProfile)
    energy = channelProfile["E_eV"].to_numpy()
    isUsable = np.isfinite(localTemperature)

    bestStart, bestEnd = -1, -1
    for startIndex in range(len(localTemperature)):
        if not isUsable[startIndex]:
            continue
        for endIndex in range(startIndex + MIN_PLATEAU_CHANNELS - 1, len(localTemperature)):
            window = localTemperature[startIndex : endIndex + 1]
            if not np.isfinite(window).all():
                break
            if window.max() / window.min() > PLATEAU_TOLERANCE:
                break
            if endIndex - startIndex > bestEnd - bestStart:
                bestStart, bestEnd = startIndex, endIndex

    if bestStart < 0:
        return {"status": "no_plateau", "core_T_eV": np.nan}
    return {
        "status": "ok",
        "core_T_eV": float(np.median(localTemperature[bestStart : bestEnd + 1])),
        "plateau_low_eV": float(energy[bestStart]),
        "plateau_high_eV": float(energy[bestEnd]),
        "plateau_channels": int(bestEnd - bestStart + 1),
    }


def fitWeightedLine(
    xValues: np.ndarray,
    yValues: np.ndarray,
    sigmaValues: np.ndarray,
) -> tuple[float, float, float]:
    """Weighted least-squares straight-line fit, returning (slope, intercept, chiSquare).

    Returns NaNs instead of raising when the fit is not defined (fewer than two
    points, or a non-positive / non-finite uncertainty, which would hand polyfit
    an infinite weight and crash the worker process).
    """
    isUsable = np.isfinite(xValues) & np.isfinite(yValues) & np.isfinite(sigmaValues) & (sigmaValues > 0)
    if isUsable.sum() < 2 or np.unique(xValues[isUsable]).size < 2:
        return float("nan"), float("nan"), float("nan")
    xValues, yValues, sigmaValues = xValues[isUsable], yValues[isUsable], sigmaValues[isUsable]
    slope, intercept = np.polyfit(xValues, yValues, 1, w=1.0 / sigmaValues)
    residuals = yValues - (slope * xValues + intercept)
    return float(slope), float(intercept), float(np.sum((residuals / sigmaValues) ** 2))


def getCoreTemperatureFromSlope(slope: float) -> float:
    """A Maxwellian core gives log10(f) = c - E / (kT ln10), so kT follows from the slope."""
    if not np.isfinite(slope) or slope >= 0:
        return np.nan
    return float(-1.0 / (slope * NATURAL_LOG_OF_TEN))


def findChangePointBreak(
    profile: pd.DataFrame,
    featureNames: Sequence[str] = DEFAULT_FEATURES,
    windowLength: int | None = None,
) -> dict[str, object]:
    """Single change point on standardized per-pixel features, sorted by v_perp.

    Takes the detected-pixel profile. With the default features and window this reproduces
    the production calculateTransitions exactly; featureNames and windowLength are exposed so
    the derivative-only and short-window variants run through the same code path.
    """
    emptyResult = {
        "break_E_eV": np.nan,
        "break_v": np.nan,
        "break_index": np.nan,
        "break_fraction": np.nan,
        "status": "too_few_points",
        "profile_with_features": profile,
    }
    if len(profile) < 2 * MIN_SEGMENT_SIZE:
        return emptyResult

    logVelocity = np.log10(profile["v_perp"].to_numpy())
    logPhaseSpaceDensity = profile["log_psd"].to_numpy()
    velocitySpacing = float(np.mean(np.diff(logVelocity)))
    if not np.isfinite(velocitySpacing) or velocitySpacing <= 0:
        return {**emptyResult, "status": "invalid_log_velocity_spacing"}

    resolvedWindow = (
        getTransitionPreservingWindow(len(logVelocity)) if windowLength is None else int(windowLength)
    )
    if resolvedWindow % 2 == 0:
        resolvedWindow += 1
    maxWindow = len(logVelocity) if len(logVelocity) % 2 == 1 else len(logVelocity) - 1
    resolvedWindow = min(resolvedWindow, maxWindow)
    if resolvedWindow < MIN_SAVGOL_WINDOW:
        return emptyResult

    profileWithFeatures = profile.assign(
        smoothed=savgol_filter(logPhaseSpaceDensity, resolvedWindow, 1, deriv=0),
        deriv=savgol_filter(logPhaseSpaceDensity, resolvedWindow, 1, deriv=1, delta=velocitySpacing),
        second_deriv=savgol_filter(logPhaseSpaceDensity, resolvedWindow, 2, deriv=2, delta=velocitySpacing),
    )

    featureMatrix = np.column_stack(
        [profileWithFeatures[featureName].to_numpy(dtype=float) for featureName in featureNames]
    )
    validMask = np.isfinite(featureMatrix).all(axis=1)
    if validMask.sum() < 2 * MIN_SEGMENT_SIZE:
        return {
            **emptyResult,
            "status": "no_valid_features",
            "profile_with_features": profileWithFeatures.assign(label=-1),
        }

    labels, transitions = getSingleVelocityTransitionLabels(featureMatrix, validMask)
    profileWithFeatures = profileWithFeatures.assign(label=labels)
    if len(transitions) == 0:
        return {**emptyResult, "status": "no_transition", "profile_with_features": profileWithFeatures}

    breakIndex = int(transitions[0])
    breakVelocity = float(profileWithFeatures["v_perp"].iloc[breakIndex])
    return {
        "break_E_eV": float(vel_to_eV(breakVelocity)),
        "break_v": breakVelocity,
        "break_index": breakIndex,
        # A break that always lands at the same fraction of the sorted point list is a
        # property of the sampling, not of the plasma; this is the number that reveals it.
        "break_fraction": breakIndex / len(profileWithFeatures),
        "window_length": resolvedWindow,
        "status": "ok",
        "profile_with_features": profileWithFeatures,
    }


def findChannelSlopeChangePoint(
    channelProfile: pd.DataFrame,
    minEnergyEv: float | None = None,
    maxEnergyEv: float = ENERGY_MAX_EV,
) -> dict[str, object]:
    """The production change-point idea, applied to the channel-aggregated log-log slope.

    Same single-split objective as findChangePointBreak, but the feature is one clean
    d log f / d log E per energy channel instead of thousands of per-pixel derivatives, and
    no monotone coordinate is handed to the splitter.
    """
    fitFrame = getReliableChannels(channelProfile, minEnergyEv, maxEnergyEv)
    if len(fitFrame) < 2 * MIN_SEGMENT_SIZE:
        return {"break_E_eV": np.nan, "status": "too_few_channels"}

    logSlope = getChannelLogSlope(fitFrame)
    validMask = np.isfinite(logSlope)
    if validMask.sum() < 2 * MIN_SEGMENT_SIZE:
        return {"break_E_eV": np.nan, "status": "no_valid_features"}

    labels, transitions = getSingleVelocityTransitionLabels(logSlope.reshape(-1, 1), validMask)
    if len(transitions) == 0:
        return {"break_E_eV": np.nan, "status": "no_transition"}

    energy = fitFrame["E_eV"].to_numpy()
    breakIndex = int(transitions[0])
    nextIndex = min(breakIndex + 1, len(energy) - 1)
    return {
        "break_E_eV": float(np.sqrt(energy[breakIndex] * energy[nextIndex])),
        "break_index": breakIndex,
        "log_slope": logSlope,
        "labels": labels,
        "fit_frame": fitFrame,
        "status": "ok",
    }


def findTwoPieceBreak(
    channelProfile: pd.DataFrame,
    minEnergyEv: float | None = None,
    maxEnergyEv: float = ENERGY_MAX_EV,
    minSegmentChannels: int = MIN_SEGMENT_CHANNELS,
) -> dict[str, object]:
    """Knot where a Maxwellian core and a power-law tail together minimise the total chi-square.

    The core segment is log10(f) linear in E (the exact Maxwellian form) and the tail segment
    is log10(f) linear in log10(E) (a kappa tail is a power law well above the core), so the
    knot is chosen by which functional form the data prefer rather than by geometry in a
    feature space. Note it answers "where does the exponential stop describing the data",
    which is not the same question as "where does the halo equal the core".

    The chi-square is reduced-scaled before the Delta chi-square = 1 interval is read off,
    because the model mismatch dominates the formal errors and an unscaled interval would
    collapse onto a single channel.
    """
    fitFrame = getReliableChannels(channelProfile, minEnergyEv, maxEnergyEv)
    if len(fitFrame) < 2 * minSegmentChannels:
        return {"break_E_eV": np.nan, "status": "too_few_channels"}

    energy = fitFrame["E_eV"].to_numpy()
    logEnergy = np.log10(energy)
    logPhaseSpaceDensity = fitFrame["log_psd"].to_numpy()
    sigma = fitFrame["sigma_log_psd"].to_numpy()

    knotIndices = np.arange(minSegmentChannels, len(fitFrame) - minSegmentChannels + 1)
    chiSquares = np.full(len(knotIndices), np.inf)
    coreSlopes = np.full(len(knotIndices), np.nan)
    coreIntercepts = np.full(len(knotIndices), np.nan)
    haloSlopes = np.full(len(knotIndices), np.nan)
    haloIntercepts = np.full(len(knotIndices), np.nan)

    for candidateIndex, knotIndex in enumerate(knotIndices):
        coreSlope, coreIntercept, coreChiSquare = fitWeightedLine(
            energy[:knotIndex], logPhaseSpaceDensity[:knotIndex], sigma[:knotIndex]
        )
        haloSlope, haloIntercept, haloChiSquare = fitWeightedLine(
            logEnergy[knotIndex:], logPhaseSpaceDensity[knotIndex:], sigma[knotIndex:]
        )
        chiSquares[candidateIndex] = coreChiSquare + haloChiSquare
        coreSlopes[candidateIndex] = coreSlope
        coreIntercepts[candidateIndex] = coreIntercept
        haloSlopes[candidateIndex] = haloSlope
        haloIntercepts[candidateIndex] = haloIntercept

    # The knot lies between the last core channel and the first tail channel; the geometric
    # mean of the two is the least biased single energy to report.
    knotEnergies = np.sqrt(energy[knotIndices - 1] * energy[knotIndices])
    bestCandidate = int(np.argmin(chiSquares))
    degreesOfFreedom = max(len(fitFrame) - 4, 1)
    reducedChiSquare = chiSquares[bestCandidate] / degreesOfFreedom

    withinOneSigma = chiSquares <= chiSquares[bestCandidate] + reducedChiSquare
    energyLow = float(knotEnergies[withinOneSigma].min())
    energyHigh = float(knotEnergies[withinOneSigma].max())
    scanSpan = float(np.log10(knotEnergies[-1] / knotEnergies[0]))
    intervalSpan = float(np.log10(energyHigh / energyLow)) if energyHigh > energyLow else 0.0

    return {
        "break_E_eV": float(knotEnergies[bestCandidate]),
        "break_E_low_eV": energyLow,
        "break_E_high_eV": energyHigh,
        "core_T_eV": getCoreTemperatureFromSlope(coreSlopes[bestCandidate]),
        "core_slope": float(coreSlopes[bestCandidate]),
        "core_intercept": float(coreIntercepts[bestCandidate]),
        "halo_power_index": float(haloSlopes[bestCandidate]),
        "halo_intercept": float(haloIntercepts[bestCandidate]),
        "reduced_chi_square": float(reducedChiSquare),
        "knot_energies": knotEnergies,
        "chi_square_curve": chiSquares,
        "fit_frame": fitFrame,
        "status": "unconstrained" if scanSpan > 0 and intervalSpan > 0.5 * scanSpan else "ok",
    }


def findMaxwellianDepartureBreak(
    channelProfile: pd.DataFrame,
    coreLowEv: float = DEFAULT_CORE_FIT_LOW_EV,
    coreHighEv: float = DEFAULT_CORE_FIT_HIGH_EV,
    ratioThreshold: float = DEFAULT_CORE_HALO_RATIO,
    sigmaThreshold: float = DEFAULT_SIGMA_THRESHOLD,
    consecutiveChannels: int = DEFAULT_CONSECUTIVE_CHANNELS,
    maxEnergyEv: float = ENERGY_MAX_EV,
) -> dict[str, object]:
    """Break where the measured VDF first rises decisively above the extrapolated core Maxwellian.

    With ratioThreshold = 2 the criterion is f = 2 x f_core, i.e. the suprathermal population
    has grown to equal the core - the boundary a core-plus-halo crossing defines, obtained
    without fitting the halo. Its weak point is that the answer inherits whatever core window
    the caller chose, which is exactly what findTwoPopulationCrossing removes.
    """
    fitFrame = getReliableChannels(channelProfile, None, maxEnergyEv)
    if fitFrame.empty:
        return {"break_E_eV": np.nan, "status": "no_reliable_channels"}

    energy = fitFrame["E_eV"].to_numpy()
    logPhaseSpaceDensity = fitFrame["log_psd"].to_numpy()
    sigma = fitFrame["sigma_log_psd"].to_numpy()

    isCoreChannel = (energy >= coreLowEv) & (energy <= coreHighEv)
    if isCoreChannel.sum() < MIN_SEGMENT_CHANNELS:
        return {"break_E_eV": np.nan, "status": "too_few_core_channels"}

    coreSlope, coreIntercept, _ = fitWeightedLine(
        energy[isCoreChannel], logPhaseSpaceDensity[isCoreChannel], sigma[isCoreChannel]
    )
    if not np.isfinite(coreSlope) or coreSlope >= 0:
        return {"break_E_eV": np.nan, "status": "non_decaying_core"}

    coreModel = coreSlope * energy + coreIntercept
    logExcess = logPhaseSpaceDensity - coreModel
    excessThreshold = np.maximum(np.log10(ratioThreshold), sigmaThreshold * sigma)
    hasDeparted = (energy > coreHighEv) & (logExcess >= excessThreshold)

    breakEnergy = np.nan
    for startIndex in range(len(hasDeparted) - consecutiveChannels + 1):
        if not hasDeparted[startIndex : startIndex + consecutiveChannels].all():
            continue
        # Report where the excess actually crosses the threshold rather than the first channel
        # past it, so the answer is not quantised onto the energy table.
        breakEnergy = interpolateThresholdCrossing(energy, logExcess - excessThreshold, startIndex)
        break

    return {
        "break_E_eV": breakEnergy,
        "core_T_eV": getCoreTemperatureFromSlope(coreSlope),
        "core_slope": coreSlope,
        "core_intercept": coreIntercept,
        "core_fit_low_eV": coreLowEv,
        "core_fit_high_eV": coreHighEv,
        "core_model_log_psd": coreModel,
        "core_model_E_eV": energy,
        "log_excess": logExcess,
        "fit_frame": fitFrame,
        "status": "ok" if np.isfinite(breakEnergy) else "no_departure",
    }


def interpolateThresholdCrossing(
    energy: np.ndarray,
    excessAboveThreshold: np.ndarray,
    crossingIndex: int,
) -> float:
    """Log-linear interpolation of the energy where excessAboveThreshold turns positive."""
    if crossingIndex == 0 or excessAboveThreshold[crossingIndex - 1] >= 0:
        return float(energy[crossingIndex])

    previousExcess = excessAboveThreshold[crossingIndex - 1]
    currentExcess = excessAboveThreshold[crossingIndex]
    crossingWeight = -previousExcess / (currentExcess - previousExcess)
    logEnergyLow = np.log10(energy[crossingIndex - 1])
    logEnergyHigh = np.log10(energy[crossingIndex])
    return float(10 ** (logEnergyLow + crossingWeight * (logEnergyHigh - logEnergyLow)))


def getTwoPopulationLogPsd(
    energy: np.ndarray,
    logCoreAmplitude: float,
    coreTemperature: float,
    logTailAmplitude: float,
    tailIndex: float,
) -> np.ndarray:
    """log10 of a Maxwellian core plus a power-law suprathermal tail."""
    corePart = 10**logCoreAmplitude * np.exp(-energy / coreTemperature)
    tailPart = 10**logTailAmplitude * energy ** (-tailIndex)
    return np.log10(corePart + tailPart)


def findTwoPopulationCrossing(
    channelProfile: pd.DataFrame,
    minEnergyEv: float | None = None,
    maxEnergyEv: float = ENERGY_MAX_EV,
) -> dict[str, object]:
    """Break where a jointly fitted Maxwellian core and power-law tail contribute equally.

    Fitting both populations at once is what breaks the circularity the other methods share:
    they all need a core window chosen in advance, and the answer moves with that choice.
    Here the core temperature and the tail index are determined together by the whole profile,
    and the break is the crossing of the two components - the standard core/halo definition
    (Stverak et al. 2009) - with no window to pick and no break needed as an input.
    """
    fitFrame = getReliableChannels(channelProfile, minEnergyEv, maxEnergyEv)
    if len(fitFrame) < MIN_TWO_POPULATION_CHANNELS:
        return {"break_E_eV": np.nan, "status": "too_few_channels"}

    energy = fitFrame["E_eV"].to_numpy()
    logPhaseSpaceDensity = fitFrame["log_psd"].to_numpy()
    sigma = fitFrame["sigma_log_psd"].to_numpy()

    lowerThird = energy <= np.quantile(energy, 0.4)
    upperThird = energy >= np.quantile(energy, 0.6)
    coreSlope, coreIntercept, _ = fitWeightedLine(
        energy[lowerThird], logPhaseSpaceDensity[lowerThird], sigma[lowerThird]
    )
    tailSlope, tailIntercept, _ = fitWeightedLine(
        np.log10(energy[upperThird]), logPhaseSpaceDensity[upperThird], sigma[upperThird]
    )
    initialTemperature = getCoreTemperatureFromSlope(coreSlope)
    if not np.isfinite(initialTemperature):
        return {"break_E_eV": np.nan, "status": "non_decaying_core"}
    initialGuess = [coreIntercept, initialTemperature, tailIntercept, max(-tailSlope, 0.5)]

    try:
        parameters, parameterCovariance = curve_fit(
            getTwoPopulationLogPsd,
            energy,
            logPhaseSpaceDensity,
            p0=initialGuess,
            sigma=sigma,
            bounds=([-60.0, 1.0, -60.0, 0.1], [0.0, 200.0, 60.0, 12.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError, SystemError) as error:
        return {"break_E_eV": np.nan, "status": f"fit_failed: {type(error).__name__}"}

    logCoreAmplitude, coreTemperature, logTailAmplitude, tailIndex = parameters

    def getComponentLogRatio(testEnergy):
        logCore = logCoreAmplitude - testEnergy / (coreTemperature * NATURAL_LOG_OF_TEN)
        logTail = logTailAmplitude - tailIndex * np.log10(testEnergy)
        return logCore - logTail

    if getComponentLogRatio(energy[0]) <= 0:
        return {"break_E_eV": np.nan, "status": "tail_dominates_everywhere", "core_T_eV": coreTemperature}
    if getComponentLogRatio(energy[-1]) >= 0:
        return {"break_E_eV": np.nan, "status": "core_dominates_everywhere", "core_T_eV": coreTemperature}

    breakEnergy = float(brentq(getComponentLogRatio, energy[0], energy[-1]))
    modelEnergy = np.logspace(np.log10(energy[0]), np.log10(energy[-1]), 200)
    return {
        "break_E_eV": breakEnergy,
        "core_T_eV": float(coreTemperature),
        "tail_index": float(tailIndex),
        "parameters": parameters,
        "parameter_covariance": parameterCovariance,
        "model_E_eV": modelEnergy,
        "model_core_psd": 10**logCoreAmplitude * np.exp(-modelEnergy / coreTemperature),
        "model_tail_psd": 10**logTailAmplitude * modelEnergy ** (-tailIndex),
        "model_total_psd": 10 ** getTwoPopulationLogPsd(modelEnergy, *parameters),
        "fit_frame": fitFrame,
        "status": "ok",
    }


def findLowEnergyValley(
    channelProfile: pd.DataFrame,
    searchMaxEv: float = DEFAULT_VALLEY_SEARCH_MAX_EV,
    minProminenceDex: float = DEFAULT_VALLEY_PROMINENCE_DEX,
    lookAheadChannels: int = DEFAULT_VALLEY_LOOK_AHEAD,
) -> dict[str, object]:
    """Core/secondary boundary: the local minimum of f(E) between spacecraft and ambient electrons.

    Spacecraft photo and secondary electrons pile up at the lowest energies, so the boundary
    shows up as a valley: f falls with energy, bottoms out, then rises again. Returns
    no_valley when f is monotonic over the search range, which is the honest answer when the
    two populations are not separated by the lowest channels.
    """
    searchFrame = channelProfile.reset_index(drop=True)
    if len(searchFrame) < MIN_SAVGOL_WINDOW:
        return {"break_E_eV": np.nan, "status": "too_few_channels"}

    logPhaseSpaceDensity = searchFrame["log_psd"].to_numpy()
    energy = searchFrame["E_eV"].to_numpy()

    smoothingWindow = min(VALLEY_SMOOTHING_WINDOW, len(searchFrame))
    if smoothingWindow % 2 == 0:
        smoothingWindow -= 1
    polyOrder = min(VALLEY_SMOOTHING_POLYORDER, smoothingWindow - 1)
    smoothedLogPsd = savgol_filter(logPhaseSpaceDensity, smoothingWindow, polyOrder)

    for channelIndex in range(1, len(smoothedLogPsd) - 1):
        if energy[channelIndex] > searchMaxEv:
            break
        isLocalMinimum = (
            smoothedLogPsd[channelIndex] <= smoothedLogPsd[channelIndex - 1]
            and smoothedLogPsd[channelIndex] < smoothedLogPsd[channelIndex + 1]
        )
        if not isLocalMinimum:
            continue

        lookAheadEnd = min(channelIndex + 1 + lookAheadChannels, len(smoothedLogPsd))
        rise = float(smoothedLogPsd[channelIndex + 1 : lookAheadEnd].max() - smoothedLogPsd[channelIndex])
        if rise >= minProminenceDex:
            return {
                "break_E_eV": float(energy[channelIndex]),
                "valley_prominence_dex": rise,
                "smoothed_log_psd": smoothedLogPsd,
                "status": "ok",
            }

    return {"break_E_eV": np.nan, "smoothed_log_psd": smoothedLogPsd, "status": "no_valley"}


def getFirstReliableEnergy(channelProfile: pd.DataFrame) -> float:
    """Lowest energy of the longest unbroken run of well-measured channels."""
    reliableChannels = getReliableChannels(channelProfile)
    return float(reliableChannels["E_eV"].min()) if not reliableChannels.empty else np.nan
