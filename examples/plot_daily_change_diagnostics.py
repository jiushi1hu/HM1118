"""Render per-seed daily reservoir level and release change diagnostics."""
import argparse
import csv
from datetime import date, timedelta
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.daily_change_diagnostics import (  # noqa: E402
    DAILY_CHANGE_FIELDS, IDS, daily_change_figure_paths)


def _number(value, field, allow_missing=False):
    if value in ('', None):
        if allow_missing:
            return None
        raise ValueError(field+' is missing')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(field+' must be numeric') from exc
    if not math.isfinite(result):
        raise ValueError(field+' must be finite')
    return result


def _consecutive_runs(days):
    """Split actual dates where source years are not adjacent in calendar time."""
    for index in range(1, len(days)):
        if days[index] != days[index-1]+timedelta(days=1):
            yield index


def load_daily_change_csv(path):
    """Load one seed table and require a complete five-reservoir daily layout."""
    path = Path(path)
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != DAILY_CHANGE_FIELDS:
            raise ValueError('Unexpected daily-change CSV columns')
        rows = list(reader)
    if not rows or len(rows) % len(IDS):
        raise ValueError('Daily-change CSV must contain complete five-reservoir days')
    seed = _number(rows[0]['seed'], 'seed')
    if not seed.is_integer() or any(_number(row['seed'], 'seed') != seed for row in rows):
        raise ValueError('Daily-change CSV must contain one integer seed')
    days = len(rows)//len(IDS)
    dates, years, day_of_year = [], np.empty(days, dtype=int), np.empty(days, dtype=int)
    levels = np.empty((days, len(IDS)), dtype=float)
    deltas = np.full((days, len(IDS)), np.nan, dtype=float)
    releases = np.empty((days, len(IDS)), dtype=float)
    release_deltas = np.full((days, len(IDS)), np.nan, dtype=float)
    for day_index in range(days):
        expected_day = None
        for reservoir_index, reservoir in enumerate(IDS):
            row = rows[day_index*len(IDS)+reservoir_index]
            if row['reservoir'] != reservoir:
                raise ValueError('Daily-change CSV must use WDD/BHT/XLD/XJB/THR order')
            day = date.fromisoformat(row['date'])
            if expected_day is None:
                expected_day = day
                dates.append(day)
                years[day_index] = int(row['year'])
                day_of_year[day_index] = int(row['day_of_year'])
                if years[day_index] != day.year or day_of_year[day_index] != day.timetuple().tm_yday:
                    raise ValueError('Daily-change CSV year or day_of_year disagrees with its date')
            elif day != expected_day:
                raise ValueError('Every reservoir must use the same actual date on a given day')
            levels[day_index, reservoir_index] = _number(row['level_m'], 'level_m')
            releases[day_index, reservoir_index] = _number(row['release_m3s'], 'release_m3s')
            delta = _number(row['delta_level_m'], 'delta_level_m', allow_missing=True)
            release_delta = _number(row['delta_release_m3s'], 'delta_release_m3s', allow_missing=True)
            if day_index == 0:
                if delta is not None or release_delta is not None:
                    raise ValueError('The first daily difference must be missing')
            elif delta is None or release_delta is None:
                raise ValueError('Only the first daily difference may be missing')
            else:
                deltas[day_index, reservoir_index] = delta
                release_deltas[day_index, reservoir_index] = release_delta
            magnitude = _number(row['abs_delta_level_m'], 'abs_delta_level_m', allow_missing=True)
            if delta is None:
                if magnitude is not None:
                    raise ValueError('Missing level change requires missing absolute level change')
            elif magnitude is None or not np.isclose(magnitude, abs(delta), rtol=1e-12, atol=1e-12):
                raise ValueError('Absolute level change must equal abs(delta_level_m)')
    if len(set(dates)) != len(dates):
        raise ValueError('Daily-change dates must be unique')
    return dict(seed=int(seed), dates=dates, years=years, day_of_year=day_of_year,
                levels=levels, deltas=deltas, releases=releases,
                release_deltas=release_deltas)


def _plot_broken_lines(ax, dates, values, label):
    starts = [0]+list(_consecutive_runs(dates))
    stops = starts[1:]+[len(dates)]
    for index, (start, stop) in enumerate(zip(starts, stops)):
        ax.plot(dates[start:stop], values[start:stop], color='#16608a', linewidth=1.1,
                marker='.' if stop-start < 3 else None, markersize=3.,
                label=label if index == 0 else None)


def make_daily_level_change_figure(series):
    figure, axes = plt.subplots(len(IDS), 1, figsize=(14, 11), sharex=True)
    figure.suptitle('Daily end-of-day water-level change · seed {}'.format(series['seed']))
    for index, (ax, reservoir) in enumerate(zip(axes, IDS)):
        _plot_broken_lines(ax, series['dates'], series['deltas'][:, index],
                           'current end-of-day level minus previous')
        ax.axhline(0., color='0.35', linewidth=.8)
        ax.set_ylabel(reservoir+'\nΔZ (m/day)')
        ax.grid(alpha=.2)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        axes[-1].xaxis.get_major_locator()))
    axes[-1].set_xlabel('Actual source date (line breaks at nonconsecutive year joins)')
    figure.tight_layout(rect=(0., 0., 1., .97))
    return figure


def _heatmap_matrix(series, reservoir_index):
    years = sorted(int(year) for year in np.unique(series['years']))
    matrix = np.full((len(years), 366), np.nan)
    for day_index, year in enumerate(series['years']):
        row = years.index(int(year))
        column = int(series['day_of_year'][day_index])-1
        matrix[row, column] = abs(series['deltas'][day_index, reservoir_index])
    return years, matrix


def make_level_change_heatmap_figure(series):
    years = sorted(int(year) for year in np.unique(series['years']))
    matrices = [_heatmap_matrix(series, index)[1] for index in range(len(IDS))]
    finite = np.concatenate([matrix[np.isfinite(matrix)].ravel() for matrix in matrices])
    maximum = float(np.max(finite)) if finite.size else 1.
    if maximum <= 0.:
        maximum = 1.
    figure, axes = plt.subplots(len(IDS), 1, figsize=(14, 14), sharex=True, sharey=True,
                                constrained_layout=True)
    figure.suptitle('Absolute daily water-level change by year and day of year · seed {}'.format(
        series['seed']))
    for ax, reservoir, matrix in zip(axes, IDS, matrices):
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', origin='lower',
            interpolation='nearest', extent=(.5, 366.5, years[0]-.5, years[-1]+.5),
            cmap='magma', vmin=0., vmax=maximum)
        ax.set_ylabel(reservoir+'\nYear')
        ax.set_yticks(years)
        ax.tick_params(axis='y', labelsize=8)
    ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[-1].set_xlabel('Day of year (actual calendar date; leap days retained)')
    colorbar = figure.colorbar(image, ax=list(axes), pad=.015, fraction=.025)
    colorbar.set_label('|ΔZ| (m/day)')
    return figure


def level_statistics(series):
    magnitude = np.abs(series['deltas'])
    result = []
    for reservoir_index, reservoir in enumerate(IDS):
        values = magnitude[1:, reservoir_index]
        values = values[np.isfinite(values)]
        result.append(dict(reservoir=reservoir, mean=float(np.mean(values)),
            p95=float(np.percentile(values, 95)), maximum=float(np.max(values)),
            sample_count=int(len(values))))
    return result


def make_level_change_statistics_figure(series):
    statistics = level_statistics(series)
    figure, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(IDS))
    width = .26
    values = (('mean', 'Mean', '#2878a3'), ('p95', 'P95', '#e69f00'),
              ('maximum', 'Maximum', '#b13f45'))
    for offset, (key, label, color) in zip((-width, 0., width), values):
        heights = [row[key] for row in statistics]
        bars = ax.bar(x+offset, heights, width, label=label, color=color)
        for bar, height in zip(bars, heights):
            ax.annotate('{:.4g}'.format(height), (bar.get_x()+bar.get_width()/2., height),
                xytext=(0., 3.), textcoords='offset points', ha='center', va='bottom',
                fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(IDS)
    ax.set_ylabel('|ΔZ| (m/day)')
    ax.set_title('Daily water-level change statistics · seed {}'.format(series['seed']))
    ax.margins(y=.14)
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)
    figure.tight_layout()
    return figure


def make_daily_release_change_figure(series):
    figure, axes = plt.subplots(len(IDS), 1, figsize=(14, 11), sharex=True)
    figure.suptitle('Daily release change · seed {}'.format(series['seed']))
    for ax, reservoir, index in zip(axes, IDS, range(len(IDS))):
        _plot_broken_lines(ax, series['dates'], series['release_deltas'][:, index],
                           'current daily release minus previous daily release')
        ax.axhline(0., color='0.35', linewidth=.8)
        ax.set_ylabel(reservoir+'\nΔQ (m³/s/day)')
        ax.grid(alpha=.2)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        axes[-1].xaxis.get_major_locator()))
    axes[-1].set_xlabel('Actual source date (line breaks at nonconsecutive year joins)')
    figure.tight_layout(rect=(0., 0., 1., .97))
    return figure


def save_daily_change_figures(input_path, output_stem):
    """Render the four reserved PNGs from one per-seed CSV."""
    series = load_daily_change_csv(input_path)
    paths = daily_change_figure_paths(output_stem)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    figures = dict(
        daily_level_change=make_daily_level_change_figure(series),
        level_change_heatmap=make_level_change_heatmap_figure(series),
        level_change_statistics=make_level_change_statistics_figure(series),
        daily_release_change=make_daily_release_change_figure(series))
    try:
        for name, figure in figures.items():
            figure.savefig(paths[name], dpi=160, bbox_inches='tight')
    finally:
        for figure in figures.values():
            plt.close(figure)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-backend', action='store_true',
                        help='Check that a noninteractive Matplotlib backend imports')
    parser.add_argument('--input', type=Path, help='One per-seed daily-change CSV')
    parser.add_argument('--output-stem', type=Path,
                        help='Per-seed path stem used to create four PNG files')
    args = parser.parse_args(argv)
    if args.check_backend:
        print('Matplotlib backend: '+matplotlib.get_backend())
        return 0
    if args.input is None or args.output_stem is None:
        parser.error('--input and --output-stem are required unless --check-backend is used')
    paths = save_daily_change_figures(args.input, args.output_stem)
    for path in paths.values():
        print(path.resolve())
    return 0


if __name__ == '__main__':
    main()
