"""Break energy of every epoch with the two-population crossing method.

For each epoch the pixels within 90 +/- 20 deg pitch angle are averaged per
energy channel (zero counts included, so the profile is unbiased at low
counts), a Maxwellian core plus power-law tail is fitted jointly to the
channel profile, and the break energy is the energy at which the two
components cross. The low-energy edge of the profile follows the production
rule ``12 eV - scpot`` so residual spacecraft secondaries stay out.

One CSV per input file is written to the output directory as
``<file stem>_two_population_crossing.csv`` (the fitting runner looks the file
up by that name), plus a combined CSV and overview figure. Files whose CSV
already exists are skipped, so an interrupted run can simply be restarted.

Usage, from the repository root:

    python src/compute_break_energy_crossing.py \\
        --vdf-dir examples/example_data \\
        --output-dir examples/example_data      # default: results/break_energy/<vdf-dir name>/

    from compute_break_energy_crossing import runCrossingPipeline
    runCrossingPipeline('examples/example_data', 'examples/example_data')
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import break_energy_methods as bem
from compute_break_energy_cone_features import getScpotValue, resolveSpacecraftPotentials
from data_io import convert_units, load_data

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "break_energy"
FILE_PATTERN = "VDF_eas_forFitting_*_1min.h5"
METHOD_NAME = "two_population_crossing"
CONE_HALF_WIDTH_DEG = 20.0
DEFAULT_WORKERS = 4

RESULT_COLUMNS = [
    "timestamp", "epoch_id", "scpot_V",
    "kT_plateau_eV", "plateau_low_eV", "plateau_high_eV", "plateau_status",
    "n_reliable_channels",
    "low_cut_eV", "break_E_eV", "core_T_eV", "tail_index", "status",
]


def analyzeEpoch(epochIndex: int, data: dict, spacecraftPotentials: np.ndarray,
                 coneHalfWidthDeg: float = CONE_HALF_WIDTH_DEG) -> dict:
    """One epoch through the crossing method; returns the result row.

    ``data`` is the unit-converted output of data_io.load_data; the row keeps
    the diagnostics (kT plateau, channel count, tail index) next to the break."""
    timestamp = pd.Timestamp(data["time_EAS"][epochIndex])
    scpotV = getScpotValue(spacecraftPotentials[epochIndex])
    lowCutEv = bem.BASELINE_LOW_ENERGY_EV - scpotV
    row = {"timestamp": timestamp, "epoch_id": epochIndex, "scpot_V": scpotV,
           "low_cut_eV": lowCutEv}

    profile, profileStatus = bem.buildConeProfile(epochIndex, data, coneHalfWidthDeg)
    if profileStatus != "ok":
        row["status"] = f"profile_{profileStatus}"
        return row

    channels = bem.aggregateByEnergyChannel(profile)
    reliableChannels = bem.getReliableChannels(channels)
    plateau = bem.findTemperaturePlateau(reliableChannels)
    row["kT_plateau_eV"] = plateau.get("core_T_eV", np.nan)
    row["plateau_low_eV"] = plateau.get("plateau_low_eV", np.nan)
    row["plateau_high_eV"] = plateau.get("plateau_high_eV", np.nan)
    row["plateau_status"] = plateau["status"]
    row["n_reliable_channels"] = len(reliableChannels)

    crossing = bem.findTwoPopulationCrossing(channels, minEnergyEv=lowCutEv)
    row["break_E_eV"] = crossing.get("break_E_eV", np.nan)
    row["core_T_eV"] = crossing.get("core_T_eV", np.nan)
    row["tail_index"] = crossing.get("tail_index", np.nan)
    row["status"] = crossing["status"]
    return row


def getCsvPath(outputDir: Path, fileName: str) -> Path:
    return Path(outputDir) / f"{Path(fileName).stem}_{METHOD_NAME}.csv"


def processFile(vdfDirText: str, outputDirText: str, fileName: str,
                maxEpochs: int | None = None,
                coneHalfWidthDeg: float = CONE_HALF_WIDTH_DEG) -> str:
    """Worker: one input file -> one CSV. Returns a summary line."""
    vdfDir, outputDir = Path(vdfDirText), Path(outputDirText)
    startTime = time.monotonic()

    data = convert_units(load_data(vdfDir, fileName))
    epochCount = len(data["time_EAS"])
    spacecraftPotentials = resolveSpacecraftPotentials(data["scpot_aligned"], epochCount)
    if maxEpochs is not None:
        epochCount = min(epochCount, maxEpochs)

    rows = []
    for epochIndex in range(epochCount):
        try:
            rows.append(analyzeEpoch(epochIndex, data, spacecraftPotentials, coneHalfWidthDeg))
        except Exception as error:   # one pathological epoch must not take the whole file down
            rows.append({"timestamp": pd.Timestamp(data["time_EAS"][epochIndex]), "epoch_id": epochIndex,
                         "status": f"error_{type(error).__name__}"})
            print(f"{fileName} epoch {epochIndex}: {type(error).__name__}: {error}")
    frame = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    frame.to_csv(getCsvPath(outputDir, fileName), index=False)
    okCount = int((frame["status"] == "ok").sum())
    return (f"{fileName}: {epochCount} epochs in {time.monotonic() - startTime:.0f} s, "
            f"break ok {okCount}/{epochCount}")


def combineResults(outputDir: Path) -> Path:
    """Concatenate the per-file CSVs into a combined CSV and overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputDir = Path(outputDir)
    perFilePaths = sorted(path for path in outputDir.glob(f"*_{METHOD_NAME}.csv")
                          if not path.stem.startswith("combined_"))
    combinedFrame = pd.concat(
        [pd.read_csv(path, parse_dates=["timestamp"]) for path in perFilePaths],
        ignore_index=True,
    ).sort_values(["timestamp", "epoch_id"], kind="stable")
    combinedFrame.insert(0, "combined_epoch_id", np.arange(len(combinedFrame)))
    combinedPath = outputDir / f"combined_{METHOD_NAME}.csv"
    combinedFrame.to_csv(combinedPath, index=False)

    okFrame = combinedFrame[combinedFrame["status"] == "ok"]
    okFraction = len(okFrame) / max(len(combinedFrame), 1)
    figure, axis = plt.subplots(figsize=(14, 4.5))
    axis.plot(okFrame["timestamp"], okFrame["break_E_eV"], ".", ms=3, color="#36AE7C")
    axis.set_ylabel("break energy [eV]")
    axis.set_title(f"{METHOD_NAME}   {len(okFrame)}/{len(combinedFrame)} epochs ok "
                   f"({100 * okFraction:.0f} %)")
    axis.grid(alpha=0.3)
    figure.autofmt_xdate()
    figure.savefig(outputDir / f"combined_{METHOD_NAME}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"combined {len(combinedFrame)} epochs from {len(perFilePaths)} files "
          f"({100 * okFraction:.0f} % ok) -> {combinedPath}")
    return combinedPath


def runCrossingPipeline(vdfDir, outputDir=None, fileNames=None,
                        workers: int = DEFAULT_WORKERS,
                        maxEpochsPerFile: int | None = None,
                        overwrite: bool = False, writeCombined: bool = True,
                        coneHalfWidthDeg: float = CONE_HALF_WIDTH_DEG) -> Path:
    """Run the crossing method over a directory of preprocessed VDF files.

    outputDir defaults to results/break_energy/<vdfDir name>/. fileNames pins
    the inputs (basenames or paths); without it every FILE_PATTERN file in
    vdfDir is processed. Existing CSVs are skipped unless overwrite is set.
    Returns the output directory."""
    vdfDir = Path(vdfDir)
    outputDir = DEFAULT_OUTPUT_ROOT / vdfDir.name if outputDir is None else Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    if fileNames is None:
        fileNames = sorted(path.name for path in vdfDir.glob(FILE_PATTERN))
    else:
        fileNames = sorted({Path(fileName).name for fileName in fileNames})
    if not fileNames:
        raise FileNotFoundError(f"No {FILE_PATTERN} files in {vdfDir}")

    pendingFileNames = [fileName for fileName in fileNames
                        if overwrite or not getCsvPath(outputDir, fileName).exists()]
    print(f"{len(fileNames)} files, {len(pendingFileNames)} to process "
          f"({len(fileNames) - len(pendingFileNames)} already done), {workers} workers")

    if pendingFileNames:
        pendingCount = len(pendingFileNames)
        with ProcessPoolExecutor(max_workers=min(workers, pendingCount)) as executor:
            summaries = executor.map(
                processFile,
                [str(vdfDir)] * pendingCount, [str(outputDir)] * pendingCount,
                pendingFileNames, [maxEpochsPerFile] * pendingCount,
                [coneHalfWidthDeg] * pendingCount)
            for summary in summaries:
                print(summary, flush=True)

    if writeCombined:
        combineResults(outputDir)
    return outputDir


def createArgumentParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vdf-dir", type=Path, required=True,
                        help="directory with the preprocessed VDF h5 files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: results/break_energy/<vdf-dir name>/")
    parser.add_argument("--files", nargs="*", default=None,
                        help="process only these file names (default: all)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="parallel workers, one file each")
    parser.add_argument("--max-epochs-per-file", type=int, default=None,
                        help="only the first N epochs per file (tests)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute files whose CSV already exists")
    parser.add_argument("--no-combined", action="store_true",
                        help="skip the combined CSV and figure")
    return parser


def main() -> None:
    args = createArgumentParser().parse_args()
    runCrossingPipeline(args.vdf_dir, args.output_dir, fileNames=args.files,
                        workers=args.workers, maxEpochsPerFile=args.max_epochs_per_file,
                        overwrite=args.overwrite, writeCombined=not args.no_combined)


if __name__ == "__main__":
    main()
