"""Per-epoch diagnostic figure for the Poisson fitting pipeline.

Layout follows the production overall-fit 4-panel figure:
top row = parallel / perpendicular 1D cuts in production style (energy top
axis, break-velocity markers, dashed component curves, black total), with the
count>=0 channel means + Poisson errors as data points; bottom row = the new
2D style (zero-count pixels light grey, counts>0 coloured by log10 PSD, total
model contours) plus a per-pixel Poisson deviance pull map.
"""

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from .models import (biMaxwellian, biKappa, strahlTruncatedBiKappa,
                                      temperatureFromThermalSpeed,
                                      thermalSpeedFromTemperature)
from .metrics import sliceDataMeans

AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 12
TEXT_COLOR = 'black'


def _styleProductionAxes(ax, withGrid=True):
    """Production figure style: full black box, outward ticks, minor ticks on,
    major+minor grid (grid on the slice panels only)."""
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='out', colors='black',
                   labelsize=TICK_LABEL_SIZE)
    if withGrid:
        ax.grid(True, which='major', color='0.85', linewidth=0.8)
        ax.grid(True, which='minor', color='0.93', linewidth=0.5)
        ax.set_axisbelow(True)

DATA_COLOR = '#757575'
CORE_COLOR = '#e34948'
HALO_COLOR = '#eda100'
STRAHL_COLOR = '#2a78d6'
PULL_LIMIT = 3.0


def _componentFields(params, v_par, v_perp):
    return {'core': biMaxwellian(v_par, v_perp, params['n_c'], params['u_par_c'],
                                 params['v_th_par_c'], params['v_th_perp_c']),
            'halo': biKappa(v_par, v_perp, params['n_h'], params.get('u_par_h', 0.0),
                            params['v_th_par_h'], params['v_th_perp_h'],
                            params['kappa_h']),
            'strahl': strahlTruncatedBiKappa(v_par, v_perp, params['n_b'],
                                             params['u_par_b'], params['v_th_par_b'],
                                             params['v_th_perp_b'], params['kappa_b'])}


COMPONENT_COLORS = {'core': CORE_COLOR, 'halo': HALO_COLOR, 'strahl': STRAHL_COLOR}
COMPONENT_LABELS = {'core': 'Core fit', 'halo': 'Halo fit', 'strahl': 'Strahl fit'}
# Full width of the perpendicular slice band around 90 deg pitch (data means).
PERP_SLICE_FULL_WIDTH_DEG = 30.0
# Slice-panel y-axis floor: never extend below this even if stray channel
# means do.
PANEL_YMIN_FLOOR = 1e-31
PANEL_SPEED_LIMIT = 2.0e9      # top cut panels
PANEL_2D_SPEED_LIMIT = 1.8e9   # bottom 2D panels

# Figure margins (test_two_stage_fit conventions): the top row stops short of
# FIGURE_RIGHT so the two cuts keep their velocity-span width ratio; adjusting
# TOP_ROW_RIGHT narrows them without affecting the 2D panels.
FIGURE_LEFT = 0.07
FIGURE_RIGHT = 0.97
TOP_ROW_RIGHT = 0.94
PARALLEL_WING_MAX_PITCH_DEG = 15.0
# Line colours for overlays of several fits of the same epoch.
COMPARISON_COLORS = ('black', '#d62728', '#2ca02c', '#9467bd')
BREAK_LINE_COLOR = '#888888'


def wingMask(grid, wingName):
    """Valid pixels of one slice cone: 'par' (pitch < 15 deg), 'anti'
    (pitch > 165 deg) or 'perp' (90 +/- 15 deg)."""
    pitch = grid['pitch_angle']
    if wingName == 'par':
        cone = pitch < PARALLEL_WING_MAX_PITCH_DEG
    elif wingName == 'anti':
        cone = pitch > 180 - PARALLEL_WING_MAX_PITCH_DEG
    elif wingName == 'perp':
        halfWidth = PERP_SLICE_FULL_WIDTH_DEG / 2
        cone = (pitch > 90 - halfWidth) & (pitch < 90 + halfWidth)
    else:
        raise ValueError(f"unknown wing {wingName!r}; use 'par', 'anti' or 'perp'")
    return grid['valid'] & cone


def _meanPitchRay(grid, coneMask, speedGrid):
    """(v_par, v_perp) along the cone's mean pitch-angle direction."""
    meanPitchRad = np.radians(np.mean(grid['pitch_angle'][coneMask]))
    return speedGrid * np.cos(meanPitchRad), speedGrid * np.sin(meanPitchRad)


def _energyTopAxis(ax, speedScale, signed):
    lowerSpeed = -PANEL_SPEED_LIMIT if signed else 0.0
    tickSpeeds = np.arange(lowerSpeed, PANEL_SPEED_LIMIT + 1e8, 0.5e9)
    tickEnergies = temperatureFromThermalSpeed(np.abs(tickSpeeds))  # E(v) in eV
    topAxis = ax.secondary_xaxis('top')
    topAxis.set_xticks(tickSpeeds / speedScale)
    topAxis.set_xticklabels([f'{energy / 100:.1f}' for energy in tickEnergies],
                            fontsize=TICK_LABEL_SIZE)
    topAxis.set_xlabel(r'Energy [$\times 10^2$ eV]', fontsize=AXIS_LABEL_SIZE)
    topAxis.tick_params(which='both', direction='out', colors='black')


def _drawWing(ax, grid, psd, counts, params, wingMask, sign, speedScale,
              withLabels=False):
    """One pitch-angle cone at signed velocity positions.

    Data points: per energy row inside the cone, mean PSD over ALL valid pixels
    with count >= 0 — zero-count pixels enter as PSD = 0, so the mean is the
    unbiased estimator of f (plotting only counts>0 pixels would show the
    censored, upward-biased survivor view). Error bar = Poisson error of that
    mean, sqrt(total counts in the row) * mean(one-count PSD) / N_pixels.
    Model curves (params not None): the analytic components evaluated on a
    dense speed grid along the wedge's mean pitch-angle ray, so the curves
    are continuous through v ~ 0 instead of stopping at the lowest channel.
    """
    energies, means, errors = sliceDataMeans(psd, counts, grid, wingMask)
    speeds = sign * thermalSpeedFromTemperature(energies)
    ax.errorbar(speeds / speedScale, means, yerr=errors, fmt='s', markersize=3.5,
                color=DATA_COLOR, ecolor=DATA_COLOR, elinewidth=0.8,
                linestyle='none', zorder=2,
                label='EAS VDF' if withLabels else None)

    if params is None:
        return
    speedGrid = np.linspace(0.0, PANEL_SPEED_LIMIT, 400)
    rayFields = _componentFields(params, *_meanPitchRay(grid, wingMask, speedGrid))
    xRay = sign * speedGrid / speedScale
    total = None
    for name, field in rayFields.items():
        total = field if total is None else total + field
        if not np.any(field > 0):
            continue   # component switched off (zero density): no curve, no legend entry
        ax.plot(xRay, field, color=COMPONENT_COLORS[name],
                linewidth=1.2, alpha=0.9, linestyle='--', zorder=3,
                label=COMPONENT_LABELS[name] if withLabels else None)
    ax.plot(xRay, total, color='black', linewidth=1.8,
            zorder=4, label='Final fit' if withLabels else None)


def _drawParallelPanel(ax, grid, psd, counts, params, speedScale):
    _drawWing(ax, grid, psd, counts, params, wingMask(grid, 'par'),
              1, speedScale, withLabels=True)
    _drawWing(ax, grid, psd, counts, params, wingMask(grid, 'anti'),
              -1, speedScale)
    ax.set_xlim(-PANEL_SPEED_LIMIT / speedScale, PANEL_SPEED_LIMIT / speedScale)
    ax.set_xlabel(r'$V_\parallel$ [$\times 10^9$ cm/s]', fontsize=AXIS_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')


def _drawPerpendicularPanel(ax, grid, psd, counts, params, speedScale):
    _drawWing(ax, grid, psd, counts, params, wingMask(grid, 'perp'),
              1, speedScale)
    ax.set_xlim(0, PANEL_SPEED_LIMIT / speedScale)
    ax.set_xlabel(r'$V_\perp$ [$\times 10^9$ cm/s]', fontsize=AXIS_LABEL_SIZE)


def _drawModelContours(ax, params, speedScale, logMin, logMax):
    """Total-model decade contours (white/black) plus the strahl alone (dashed)."""
    meshMax = PANEL_2D_SPEED_LIMIT
    parAxis = np.linspace(-meshMax, meshMax, 320)
    perpAxis = np.linspace(0, meshMax, 160)
    parMesh, perpMesh = np.meshgrid(parAxis, perpAxis, indexing='ij')
    model = sum(_componentFields(params, parMesh, perpMesh).values())
    with np.errstate(divide='ignore'):
        logModel = np.log10(np.maximum(model, 1e-40))
    ax.contour(parMesh / speedScale, perpMesh / speedScale, logModel,
               levels=np.arange(logMin, logMax), colors='white',
               linewidths=2.0, linestyles='solid')
    ax.contour(parMesh / speedScale, perpMesh / speedScale, logModel,
               levels=np.arange(logMin, logMax), colors='black',
               linewidths=0.9, linestyles='solid')
    # Strahl component alone (dashed) so the fitted beam shape is visible.
    if params.get('n_b', 0) > 0:
        strahlField = _componentFields(params, parMesh, perpMesh)['strahl']
        with np.errstate(divide='ignore'):
            logStrahl = np.log10(np.maximum(strahlField, 1e-40))
        ax.contour(parMesh / speedScale, perpMesh / speedScale, logStrahl,
                   levels=np.arange(logMin, logMax), colors=STRAHL_COLOR,
                   linewidths=1.1, linestyles='dashed')
        ax.plot([], [], color=STRAHL_COLOR, linestyle='--', linewidth=1.1,
                label='Strahl model')
        ax.legend(loc='upper left', fontsize=TICK_LABEL_SIZE - 1,
                  framealpha=0.85)


def _draw2dPanel(ax, fig, grid, psd, counts, params, speedScale, cax):
    # Continuous measured-VDF field: bin ALL valid count>=0 pixels onto a
    # regular velocity grid and take the per-bin mean PSD (zero-count pixels
    # enter as zeros, so each bin is the unbiased estimator of f there — the
    # binned version of the slice-panel channel means; single surviving
    # one-count pixels no longer dominate the outskirts).
    windowMask = (grid['valid']
                  & (np.abs(grid['v_par']) <= PANEL_2D_SPEED_LIMIT)
                  & (grid['v_perp'] <= PANEL_2D_SPEED_LIMIT))
    parEdges = np.linspace(-PANEL_2D_SPEED_LIMIT, PANEL_2D_SPEED_LIMIT, 73)
    perpEdges = np.linspace(0, PANEL_2D_SPEED_LIMIT, 37)
    psdSum, _, _ = np.histogram2d(grid['v_par'][windowMask],
                                  grid['v_perp'][windowMask],
                                  bins=(parEdges, perpEdges),
                                  weights=psd[windowMask])
    pixelCount, _, _ = np.histogram2d(grid['v_par'][windowMask],
                                      grid['v_perp'][windowMask],
                                      bins=(parEdges, perpEdges))
    with np.errstate(invalid='ignore'):
        meanField = np.where(pixelCount > 0, psdSum / np.maximum(pixelCount, 1),
                             np.nan)
    meanField = np.ma.masked_where(~np.isfinite(meanField) | (meanField <= 0),
                                   meanField)

    # Colour range from the binned means, rounded to decades; the values are
    # scaled by a mid-range decade so the ticks read as plain numbers with a
    # single 10^offset beside the bar (production colorbar style).
    logMin = np.floor(np.log10(meanField.min()))
    logMax = np.ceil(np.log10(meanField.max()))
    offsetExponent = int(np.round((logMin + logMax) / 2))
    offsetScale = 10.0 ** offsetExponent
    colorNorm = LogNorm(vmin=10.0 ** (logMin - offsetExponent),
                        vmax=10.0 ** (logMax - offsetExponent))
    scatterHandle = ax.pcolormesh(parEdges / speedScale, perpEdges / speedScale,
                                  (meanField / offsetScale).T, cmap='viridis',
                                  norm=colorNorm, shading='flat')
    if params is not None:
        _drawModelContours(ax, params, speedScale, logMin, logMax)
    ax.set_xlim(-PANEL_2D_SPEED_LIMIT / speedScale,
                PANEL_2D_SPEED_LIMIT / speedScale)
    ax.set_ylim(0, PANEL_2D_SPEED_LIMIT / speedScale)
    ax.set_xlabel(r'$v_{par}$ [$\times 10^9$ cm/s]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r'$v_{perp}$ [$\times 10^9$ cm/s]', fontsize=AXIS_LABEL_SIZE)
    ax.set_title('Measured VDF & Final-fit contours' if params is not None
                 else 'Measured VDF', fontsize=AXIS_LABEL_SIZE, color=TEXT_COLOR, pad=8)
    colorbar = fig.colorbar(scatterHandle, cax=cax, orientation='horizontal')
    tickDecades = np.arange(logMin - offsetExponent, logMax - offsetExponent + 1)
    colorbar.set_ticks(10.0 ** tickDecades)
    colorbar.set_ticklabels([f'{10.0 ** decade:g}' for decade in tickDecades])
    cax.text(1.02, 0.5, rf'$\times 10^{{{offsetExponent}}}$ s$^3$/cm$^6$',
             transform=cax.transAxes, va='center', ha='left',
             fontsize=TICK_LABEL_SIZE - 1)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE - 1)
    colorbar.outline.set_edgecolor('black')


def _drawPullPanel(ax, fig, grid, psd, counts, params, speedScale, cax):
    valid = grid['valid']
    model = sum(_componentFields(params, grid['v_par'], grid['v_perp']).values())
    expected = np.maximum(model / grid['one_count_psd'], 1e-12)
    observed = counts
    with np.errstate(divide='ignore', invalid='ignore'):
        devianceTerm = expected - observed + np.where(
            observed > 0, observed * np.log(observed / np.maximum(expected, 1e-300)), 0.0)
    pull = np.sign(observed - expected) * np.sqrt(2 * np.maximum(devianceTerm, 0))
    scatterHandle = ax.scatter(grid['v_par'][valid] / speedScale,
                               grid['v_perp'][valid] / speedScale, s=3,
                               c=pull[valid], cmap='RdBu_r',
                               vmin=-PULL_LIMIT, vmax=PULL_LIMIT, linewidths=0)
    ax.set_xlim(-PANEL_2D_SPEED_LIMIT / speedScale,
                PANEL_2D_SPEED_LIMIT / speedScale)
    ax.set_ylim(0, PANEL_2D_SPEED_LIMIT / speedScale)
    ax.set_xlabel(r'$v_{par}$ [$\times 10^9$ cm/s]', fontsize=AXIS_LABEL_SIZE)
    ax.set_title(r'Poisson deviance pull  sign$(k-\mu)\sqrt{2D}$',
                 fontsize=AXIS_LABEL_SIZE, color=TEXT_COLOR, pad=8)
    colorbar = fig.colorbar(scatterHandle, cax=cax, orientation='horizontal')
    cax.text(1.02, 0.5, 'pull', transform=cax.transAxes, va='center',
             ha='left', fontsize=TICK_LABEL_SIZE - 1)
    colorbar.ax.tick_params(labelsize=TICK_LABEL_SIZE - 1)
    colorbar.outline.set_edgecolor('black')


def sliceYLimits(grid, psd, counts):
    """Shared y-limits for the slice panels from the channel means of all
    three cones, rounded to full decades; never below PANEL_YMIN_FLOOR (the
    lowest means are single-count pixels with huge error bars)."""
    coneMeans = np.concatenate([sliceDataMeans(psd, counts, grid, wingMask(grid, name))[1]
                                for name in ('par', 'anti', 'perp')])
    ymax = 10 ** np.ceil(np.log10(coneMeans.max() * 2))
    ymin = max(10 ** np.floor(np.log10(coneMeans.min() / 2)), PANEL_YMIN_FLOOR)
    return ymin, ymax


def _styleSlicePanels(parAxis, perpAxis, ymin, ymax, speedScale, breakEnergyEv=None):
    """Log y-axes, production styling, energy top axes and break-energy markers."""
    for ax in (parAxis, perpAxis):
        ax.set_yscale('log')
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(r'PSD [s$^3$/cm$^6$]', fontsize=AXIS_LABEL_SIZE)
        _styleProductionAxes(ax, withGrid=True)
    if breakEnergyEv is not None and np.isfinite(breakEnergyEv):
        breakSpeed = thermalSpeedFromTemperature(breakEnergyEv)
        for signedSpeed in (breakSpeed, -breakSpeed):
            parAxis.axvline(signedSpeed / speedScale, color=BREAK_LINE_COLOR,
                            linestyle=':', linewidth=1.2, zorder=1)
        perpAxis.axvline(breakSpeed / speedScale, color=BREAK_LINE_COLOR,
                         linestyle=':', linewidth=1.2, zorder=1)
        parAxis.text(breakSpeed / speedScale, 0.03, f' break {breakEnergyEv:.0f} eV',
                     color='#666666', fontsize=9,
                     transform=parAxis.get_xaxis_transform(), va='bottom')
    _energyTopAxis(parAxis, speedScale, signed=True)
    _energyTopAxis(perpAxis, speedScale, signed=False)


FOUR_PANEL_FIGSIZE = (13, 11.5)


def drawPoissonFitPanels(fig, grid, psd, counts, params, diagnostics, deviancePerDof,
                         timeLabel, verticalExtent=(0.0, 1.0)):
    """Draw the four diagnostic panels of one fitted epoch into ``fig``.

    verticalExtent=(bottom, top) is the figure-fraction band the panels
    occupy (the whole figure by default); the layout below is designed for a
    band of FOUR_PANEL_FIGSIZE proportions, so a caller reserving space above
    it should scale the figure height accordingly."""
    speedScale = 1e9
    bandBottom, bandTop = verticalExtent

    def mapY(fraction):
        return bandBottom + fraction * (bandTop - bandBottom)
    # Layout copied from test_two_stage_fit.plotTwoStage: two independent grids
    # rather than one nested pair — the cuts and the 2D panels get their own
    # right margin, so the top row can be narrowed (TOP_ROW_RIGHT) without
    # touching the 2D panels underneath. Panel widths follow the velocity
    # spans (par -2..2, perp 0..2 -> 2:1), equal cm/s per inch.
    parallelSpan = 2 * PANEL_SPEED_LIMIT
    perpendicularSpan = PANEL_SPEED_LIMIT
    topGrid = fig.add_gridspec(
        1, 2, width_ratios=[parallelSpan, perpendicularSpan], wspace=0.22,
        left=FIGURE_LEFT, right=TOP_ROW_RIGHT, top=mapY(0.94), bottom=mapY(0.45))
    # Both colorbars live in a shared right-hand strip (panel c's on top, the
    # pull's below), so neither 2D panel loses x-axis width to a colorbar.
    # The 2D panels are drawn with equal axis scale (y span is half the x
    # span, so the box is half as tall as it is wide).
    bottomGrid = fig.add_gridspec(1, 2, wspace=0.14,
                                  left=FIGURE_LEFT, right=0.94,
                                  top=mapY(0.32), bottom=mapY(0.09))
    # Right-hand vertical colorbar strip (kept for reference):
    # vdf2dColorbarAxis = fig.add_axes([0.925, 0.28, 0.012, 0.115])
    # pullColorbarAxis = fig.add_axes([0.925, 0.14, 0.012, 0.115])
    # Horizontal colorbars above each 2D panel (panel boxes are anchored to
    # the bottom of their cells, so the strip above stays free).
    colorbarHeight = 0.013 * (bandTop - bandBottom)
    vdf2dColorbarAxis = fig.add_axes([0.118, mapY(0.372), 0.31, colorbarHeight])
    pullColorbarAxis = fig.add_axes([0.582, mapY(0.372), 0.31, colorbarHeight])
    parAxis = fig.add_subplot(topGrid[0, 0])
    perpAxis = fig.add_subplot(topGrid[0, 1])
    vdf2dAxis = fig.add_subplot(bottomGrid[0, 0])
    pullAxis = fig.add_subplot(bottomGrid[0, 1])

    ymin, ymax = sliceYLimits(grid, psd, counts)
    _drawParallelPanel(parAxis, grid, psd, counts, params, speedScale)
    _drawPerpendicularPanel(perpAxis, grid, psd, counts, params, speedScale)
    _styleSlicePanels(parAxis, perpAxis, ymin, ymax, speedScale,
                      diagnostics['break_energy_eV'])

    _draw2dPanel(vdf2dAxis, fig, grid, psd, counts, params, speedScale,
                 vdf2dColorbarAxis)
    vdf2dAxis.annotate(f'deviance/dof = {deviancePerDof:.2f}', xy=(0.97, 0.97),
                       xycoords='axes fraction', ha='right', va='top',
                       fontsize=TICK_LABEL_SIZE, color=TEXT_COLOR)
    _drawPullPanel(pullAxis, fig, grid, psd, counts, params, speedScale,
                   pullColorbarAxis)
    for ax in (vdf2dAxis, pullAxis):
        _styleProductionAxes(ax, withGrid=False)
        ax.set_aspect('equal', adjustable='box')
        ax.set_anchor('S')

    fig.text(0.5, mapY(0.02), timeLabel, ha='center', fontsize=AXIS_LABEL_SIZE+1, color=TEXT_COLOR)


def plotPoissonFitEpoch(grid, psd, counts, params, diagnostics, deviancePerDof,
                        timeLabel, outputPath=None):
    """Four-panel diagnostic figure of one fitted epoch.

    With outputPath the figure is saved (dpi 100) and closed, returning None;
    without it the open figure is returned for inline display."""
    fig = plt.figure(figsize=FOUR_PANEL_FIGSIZE)
    drawPoissonFitPanels(fig, grid, psd, counts, params, diagnostics, deviancePerDof,
                         timeLabel)
    if outputPath is None:
        return fig
    fig.savefig(outputPath, dpi=100)
    plt.close(fig)
    return None


# ------------------------------------------------------------ notebook helpers

def _drawTotalRay(ax, grid, params, coneMask, sign, speedScale, color, label=None):
    speedGrid = np.linspace(0.0, PANEL_SPEED_LIMIT, 400)
    total = sum(_componentFields(params, *_meanPitchRay(grid, coneMask, speedGrid)).values())
    ax.plot(sign * speedGrid / speedScale, total, color=color, linewidth=1.8,
            zorder=4, label=label)


def plotMeasuredVdf(grid, psd, counts, params=None, breakEnergyEv=None,
                    timeLabel='', highlightMask=None, highlightLabel='pixels used'):
    """Parallel / perpendicular slices and the 2D measured VDF of one epoch.

    params: optional model parameters -> component curves and total contours.
    highlightMask: optional boolean pixel mask drawn over the 2D panel (for
    instance the pixels one fit stage used). Returns the figure."""
    speedScale = 1e9
    fig = plt.figure(figsize=(16, 5.0))
    layout = fig.add_gridspec(1, 3, width_ratios=[2, 1, 2.3], wspace=0.3,
                              left=0.05, right=0.92, top=0.8, bottom=0.14)
    parAxis = fig.add_subplot(layout[0, 0])
    perpAxis = fig.add_subplot(layout[0, 1])
    vdfAxis = fig.add_subplot(layout[0, 2])
    ymin, ymax = sliceYLimits(grid, psd, counts)
    _drawParallelPanel(parAxis, grid, psd, counts, params, speedScale)
    _drawPerpendicularPanel(perpAxis, grid, psd, counts, params, speedScale)
    _styleSlicePanels(parAxis, perpAxis, ymin, ymax, speedScale, breakEnergyEv)
    colorbarAxis = vdfAxis.inset_axes([0.0, 1.22, 1.0, 0.06])
    _draw2dPanel(vdfAxis, fig, grid, psd, counts, params, speedScale, colorbarAxis)
    if highlightMask is not None:
        vdfAxis.scatter(grid['v_par'][highlightMask] / speedScale,
                        grid['v_perp'][highlightMask] / speedScale, s=1.5,
                        color='black', alpha=0.2, linewidths=0, label=highlightLabel)
        vdfAxis.legend(loc='upper left', fontsize=TICK_LABEL_SIZE - 1, framealpha=0.85)
    _styleProductionAxes(vdfAxis, withGrid=False)
    vdfAxis.set_aspect('equal', adjustable='box')
    if timeLabel:
        fig.text(0.02, 0.97, timeLabel, ha='left', va='top',
                 fontsize=AXIS_LABEL_SIZE, color=TEXT_COLOR)
    return fig


def plotFitComparison(grid, psd, counts, paramsByLabel, breakEnergyEv=None,
                      timeLabel='', colors=None):
    """Several fits of the same epoch side by side (e.g. Poisson vs chi2):
    slices with the total model of every fit, and one Poisson deviance pull
    map per fit. paramsByLabel: {label: params dict}. Returns the figure."""
    speedScale = 1e9
    labels = list(paramsByLabel)
    colors = colors or dict(zip(labels, COMPARISON_COLORS))
    fig = plt.figure(figsize=(13, 9))
    topGrid = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.22,
                               left=FIGURE_LEFT, right=TOP_ROW_RIGHT, top=0.93, bottom=0.55)
    bottomGrid = fig.add_gridspec(1, len(labels), wspace=0.14,
                                  left=FIGURE_LEFT, right=0.94, top=0.33, bottom=0.08)
    parAxis = fig.add_subplot(topGrid[0, 0])
    perpAxis = fig.add_subplot(topGrid[0, 1])
    ymin, ymax = sliceYLimits(grid, psd, counts)
    _drawParallelPanel(parAxis, grid, psd, counts, None, speedScale)
    _drawPerpendicularPanel(perpAxis, grid, psd, counts, None, speedScale)
    for label in labels:
        for wingName, sign, ax in (('par', 1, parAxis), ('anti', -1, parAxis),
                                   ('perp', 1, perpAxis)):
            _drawTotalRay(ax, grid, paramsByLabel[label], wingMask(grid, wingName),
                          sign, speedScale, colors[label],
                          label=label if wingName == 'par' else None)
    parAxis.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
    _styleSlicePanels(parAxis, perpAxis, ymin, ymax, speedScale, breakEnergyEv)
    for columnIndex, label in enumerate(labels):
        pullAxis = fig.add_subplot(bottomGrid[0, columnIndex])
        colorbarAxis = pullAxis.inset_axes([0.0, 1.3, 1.0, 0.07])
        _drawPullPanel(pullAxis, fig, grid, psd, counts, paramsByLabel[label],
                       speedScale, colorbarAxis)
        pullAxis.set_title(f'{label}: Poisson deviance pull', fontsize=AXIS_LABEL_SIZE,
                           color=colors[label], pad=8)
        _styleProductionAxes(pullAxis, withGrid=False)
        pullAxis.set_aspect('equal', adjustable='box')
        if columnIndex == 0:
            pullAxis.set_ylabel(r'$v_{perp}$ [$\times 10^9$ cm/s]', fontsize=AXIS_LABEL_SIZE)
    if timeLabel:
        fig.text(0.5, 0.01, timeLabel, ha='center', fontsize=AXIS_LABEL_SIZE, color=TEXT_COLOR)
    return fig


def plotPadSpectrogram(chunkPath, epochIndex, ax=None, energyRangeEv=(15.0, 1200.0),
                       bandEv=None):
    """Pitch-angle distribution of one epoch, normalised per energy channel to
    its maximum, as pitch angle x energy image. bandEv=(low, high) marks the
    energy band used for strahl detection. Returns the axes."""
    with h5py.File(chunkPath, 'r') as h5:
        pad = h5['pad_in'][epochIndex][:, :, 0]
        pitchAngles = h5['pitch_angles'][:]
        swaEnergyAll = h5['swa_energy']
        energies = (swaEnergyAll[epochIndex] if swaEnergyAll.ndim == 2
                    else swaEnergyAll[:])
    keep = np.isfinite(energies)
    energies, pad = energies[keep], pad[keep]
    order = np.argsort(energies)
    energies, pad = energies[order], pad[order]
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = pad / np.nanmax(pad, axis=1, keepdims=True)
    logEnergy = np.log10(energies)
    innerEdges = 0.5 * (logEnergy[1:] + logEnergy[:-1])
    energyEdges = 10 ** np.concatenate([[2 * logEnergy[0] - innerEdges[0]], innerEdges,
                                        [2 * logEnergy[-1] - innerEdges[-1]]])
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    mesh = ax.pcolormesh(pitchAngles, energyEdges, normalised, cmap='viridis',
                         vmin=0, vmax=1, shading='flat')
    ax.set_yscale('log')
    ax.set_ylim(*energyRangeEv)
    ax.set_xlabel('pitch angle [deg]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('energy [eV]', fontsize=AXIS_LABEL_SIZE)
    if bandEv is not None:
        for edge in bandEv:
            ax.axhline(edge, color='white', linestyle='--', linewidth=1.2)
    ax.figure.colorbar(mesh, ax=ax, label='PSD / max per channel')
    _styleProductionAxes(ax, withGrid=False)
    return ax


def plotPadBandFit(padFit, detection=None, ax=None):
    """Band-mean PAD (points) with its double-Gaussian fit (line); padFit is
    the dict from pad_detection.fitBandMeanPad, detection the optional verdict
    from detectStrahlFromPad (annotated in the title). Returns the axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(padFit['pitch_centers'], padFit['mean_pad'], 's', color=DATA_COLOR,
            label='band-mean PAD')
    ax.plot(padFit['fit_alpha'], padFit['fit_curve'], '-', color=STRAHL_COLOR,
            linewidth=1.8, label='double-Gaussian fit')
    ax.set_yscale('log')
    ax.set_xlabel('pitch angle [deg]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('mean PSD in band', fontsize=AXIS_LABEL_SIZE)
    low, high = padFit['band_ev']
    title = f'PAD band {low:.0f}-{high:.0f} eV'
    if detection is not None:
        title += (f"   {detection['class']}   beta_par={detection['beta_par']:.2f} "
                  f"beta_anti={detection['beta_anti']:.2f}")
    ax.set_title(title, fontsize=TICK_LABEL_SIZE + 1)
    ax.legend(fontsize=LEGEND_FONT_SIZE - 1)
    _styleProductionAxes(ax, withGrid=True)
    return ax


def plotBreakEnergyCrossing(crossing, ax=None, lowCutEv=None):
    """Cone-averaged channel profile with the fitted Maxwellian core, power-law
    tail and their crossing (the break energy). crossing is the result dict of
    break_energy_methods.findTwoPopulationCrossing with status 'ok'."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    fitFrame = crossing['fit_frame']
    psdValues = 10 ** fitFrame['log_psd'].to_numpy()
    sigmaLog = fitFrame['sigma_log_psd'].to_numpy()
    ax.errorbar(fitFrame['E_eV'], psdValues,
                yerr=[psdValues * (1 - 10 ** -sigmaLog), psdValues * (10 ** sigmaLog - 1)],
                fmt='s', color=DATA_COLOR, markersize=4, label='cone channel mean')
    ax.plot(crossing['model_E_eV'], crossing['model_core_psd'], '--', color=CORE_COLOR,
            label=f"Maxwellian core, T = {crossing['core_T_eV']:.1f} eV")
    ax.plot(crossing['model_E_eV'], crossing['model_tail_psd'], '--', color=HALO_COLOR,
            label=f"power-law tail, index {crossing['tail_index']:.2f}")
    ax.plot(crossing['model_E_eV'], crossing['model_total_psd'], '-', color='black',
            linewidth=1.8, label='core + tail')
    ax.axvline(crossing['break_E_eV'], color=BREAK_LINE_COLOR, linestyle=':', linewidth=1.5,
               label=f"break {crossing['break_E_eV']:.1f} eV")
    if lowCutEv is not None:
        ax.axvline(lowCutEv, color='0.6', linestyle='-.', linewidth=1.0, label='low-energy cut')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(psdValues.min() / 30, psdValues.max() * 3)
    ax.set_xlabel('energy [eV]', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r'PSD [s$^3$/cm$^6$]', fontsize=AXIS_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE - 2)
    _styleProductionAxes(ax, withGrid=True)
    ax.figure.tight_layout()
    return ax


def main():
    """Fit and plot a single epoch from the command line.

    Usage:
        PYTHONPATH=src python -m evdf_poisson.plotting \
            --chunk examples/example_data/poisson_2023-04-04/VDF_eas_forFitting_20230404005008_20230404011908_1min.h5 \
            --epoch 5 [--out results/fitting/single_epoch] [--break-energy 92.6]
    """
    import argparse
    from pathlib import Path
    from .runner import fitOneEpoch

    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument('--chunk', required=True, help='preprocessed chunk h5 file')
    parser.add_argument('--epoch', type=int, required=True, help='epoch index in the chunk')
    parser.add_argument('--out', default='results/fitting/single_epoch',
                        help='output directory for the figure')
    parser.add_argument('--break-energy', type=float, default=None,
                        help='crossing break energy in eV (default: 7.7 T_c proxy)')
    args = parser.parse_args()

    outputDir = Path(args.out)
    outputDir.mkdir(parents=True, exist_ok=True)
    chunkStem = Path(args.chunk).stem.replace('VDF_eas_forFitting_', '')
    plotPath = outputDir / f'poissonfit_{chunkStem}_epoch{args.epoch:04d}.png'

    row = fitOneEpoch(args.chunk, args.epoch, plotPath=str(plotPath),
                      breakEnergyEv=args.break_energy)
    if row.get('error'):
        print(f"epoch {args.epoch}: {row['error']}")
        return
    print(f"{row['time_iso']}  deviance/dof={row['deviance_per_dof']:.3f}  "
          f"n_c={row['n_c']:.3f} n_h={row['n_h']:.4f} n_b={row['n_b']:.4f} cm^-3  "
          f"kappa_h={row['kappa_h']:.2f}  ({row['runtime_s']:.1f} s)")
    print('figure ->', plotPath)


if __name__ == '__main__':
    main()
