"""Observational daily level and release changes from evaluation trajectories.

The diagnostic layer only reads executed end-of-day levels and daily releases.
It never feeds differences back into observations, rewards, or optimization.
Source-year joins remain part of one continuous episode sequence, so a change is
computed across a join and only the very first daily step is missing.
"""
import csv
from datetime import date
from pathlib import Path

import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS


SCHEMA_VERSION = 1
DAILY_CHANGE_FIELDS = (
    'seed', 'date', 'year', 'day_of_year', 'reservoir', 'level_m',
    'delta_level_m', 'abs_delta_level_m', 'release_m3s', 'delta_release_m3s')
STATISTICS_FIELDS = (
    'seed', 'reservoir', 'sample_count', 'mean_abs_delta_level_m',
    'p95_abs_delta_level_m', 'max_abs_delta_level_m')
FIGURE_SUFFIXES = {
    'daily_level_change': '_daily_level_change.png',
    'level_change_heatmap': '_abs_level_change_heatmap.png',
    'level_change_statistics': '_level_change_statistics.png',
    'daily_release_change': '_daily_release_change.png',
}


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    return int(value)


def _calendar_date(value):
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError('trajectory dates must be ISO calendar dates') from exc


def _matrix(value, name, days=None):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(name+' must be a finite (days, 5) matrix') from exc
    if result.ndim != 2 or result.shape[1] != len(IDS) or (days is not None and len(result) != days):
        raise ValueError(name+' must have shape (days, 5)')
    if not np.isfinite(result).all():
        raise ValueError(name+' must contain only finite executed values')
    return result


def validate_daily_series(dates, levels, releases, expected_steps=None):
    """Validate one reservoir-major daily sequence without imposing year order.

    Held-out years may be nonconsecutive and even move backwards at a source-year
    join. Their source dates stay authoritative, while differences remain
    continuous in episode order across each join.
    """
    parsed = tuple(_calendar_date(day) for day in dates)
    if not parsed:
        raise ValueError('A daily diagnostic sequence must contain at least one day')
    if len(set(parsed)) != len(parsed):
        raise ValueError('Daily diagnostic dates must be unique')
    if expected_steps is not None and len(parsed) != _integer(expected_steps, 'expected_steps'):
        raise ValueError('Daily diagnostic sequence does not contain every expected evaluation day')
    levels = _matrix(levels, 'levels_end', len(parsed))
    releases = _matrix(releases, 'release', len(parsed))
    return parsed, levels, releases


def daily_changes(levels, releases):
    """Return level and release differences in episode order, with step 0 missing."""
    levels = _matrix(levels, 'levels_end')
    releases = _matrix(releases, 'release')
    if len(levels) != len(releases):
        raise ValueError('levels_end and release must contain the same number of days')
    level_change = np.full_like(levels, np.nan)
    release_change = np.full_like(releases, np.nan)
    if len(levels) > 1:
        level_change[1:] = np.diff(levels, axis=0)
        release_change[1:] = np.diff(releases, axis=0)
    return level_change, release_change


def level_change_statistics(levels):
    """Summarize finite absolute level changes independently for five reservoirs."""
    levels = _matrix(levels, 'levels_end')
    change, _ = daily_changes(levels, np.zeros_like(levels))
    magnitude = np.abs(change)
    rows = []
    for index, reservoir in enumerate(IDS):
        values = magnitude[1:, index]
        values = values[np.isfinite(values)]
        rows.append(dict(reservoir=reservoir, sample_count=int(len(values)),
            mean_abs_delta_level_m=(float(np.mean(values)) if len(values) else None),
            p95_abs_delta_level_m=(float(np.percentile(values, 95)) if len(values) else None),
            max_abs_delta_level_m=(float(np.max(values)) if len(values) else None)))
    return rows


def _csv_value(value):
    if value is None:
        return ''
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return ''
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return format(float(value), '.15g')
    return str(value)


def write_daily_change_csv(path, trajectory):
    """Write one exclusive per-seed table; the first delta in each channel is empty."""
    path = Path(path)
    seed = _integer(trajectory.get('seed'), 'seed')
    dates, levels, releases = validate_daily_series(
        trajectory.get('dates', ()), trajectory.get('levels_end', ()),
        trajectory.get('release', ()), trajectory.get('steps'))
    level_change, release_change = daily_changes(levels, releases)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=DAILY_CHANGE_FIELDS)
        writer.writeheader()
        for day_index, day in enumerate(dates):
            for reservoir_index, reservoir in enumerate(IDS):
                level_delta = level_change[day_index, reservoir_index]
                writer.writerow(dict(seed=seed, date=day.isoformat(), year=day.year,
                    day_of_year=day.timetuple().tm_yday, reservoir=reservoir,
                    level_m=_csv_value(levels[day_index, reservoir_index]),
                    delta_level_m=_csv_value(level_delta),
                    abs_delta_level_m=_csv_value(abs(level_delta) if np.isfinite(level_delta) else None),
                    release_m3s=_csv_value(releases[day_index, reservoir_index]),
                    delta_release_m3s=_csv_value(release_change[day_index, reservoir_index])))
    return path


def write_level_change_statistics_csv(path, seed, levels):
    """Write the same mean/P95/max values displayed by the statistics figure."""
    path = Path(path)
    seed = _integer(seed, 'seed')
    rows = level_change_statistics(levels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=STATISTICS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(seed=seed, reservoir=row['reservoir'],
                sample_count=row['sample_count'],
                mean_abs_delta_level_m=_csv_value(row['mean_abs_delta_level_m']),
                p95_abs_delta_level_m=_csv_value(row['p95_abs_delta_level_m']),
                max_abs_delta_level_m=_csv_value(row['max_abs_delta_level_m'])))
    return path


def daily_change_figure_paths(stem):
    """Map a per-seed output stem to the four reserved PNG paths."""
    stem = Path(stem)
    return {name:stem.with_name(stem.name+suffix)
            for name, suffix in FIGURE_SUFFIXES.items()}


def daily_change_output_paths(directory, seeds, algorithm='happo'):
    """Reserve one raw CSV, one summary CSV and four figures for every seed."""
    if directory is None:
        return {}
    directory = Path(directory)
    algorithm = str(algorithm).lower()
    if not algorithm:
        raise ValueError('algorithm must be nonempty')
    result = {}
    for seed in seeds:
        seed = _integer(seed, 'seed')
        stem = directory/(algorithm+'_seed'+str(seed))
        paths = dict(csv=stem.with_name(stem.name+'_daily_changes.csv'),
                     statistics_csv=stem.with_name(stem.name+'_level_change_statistics.csv'))
        paths.update(daily_change_figure_paths(stem))
        result[seed] = paths
    return result


def flatten_output_paths(paths):
    return [path for seed_paths in paths.values() for path in seed_paths.values()]


class DailyChangeRecorder:
    """Capture executed daily level and release vectors from one evaluation."""

    def __init__(self, seed):
        self.seed = _integer(seed, 'seed')
        self.dates, self.levels, self.releases = [], [], []

    def capture(self, transition):
        if not isinstance(transition, dict):
            raise ValueError('A daily transition must be a mapping')
        day = _calendar_date(transition.get('date'))
        if self.dates and day == self.dates[-1]:
            raise ValueError('A daily evaluation cannot repeat a calendar date')
        level = np.asarray(transition.get('levels_end'), dtype=np.float64)
        release = np.asarray(transition.get('release'), dtype=np.float64)
        if level.shape != (len(IDS),) or not np.isfinite(level).all():
            raise ValueError('levels_end must be a finite five-vector')
        if release.shape != (len(IDS),) or not np.isfinite(release).all():
            raise ValueError('release must be a finite five-vector')
        self.dates.append(day.isoformat())
        self.levels.append(level.tolist())
        self.releases.append(release.tolist())

    def finish(self, expected_steps=None):
        dates, levels, releases = validate_daily_series(
            self.dates, self.levels, self.releases, expected_steps)
        return dict(schema_version=SCHEMA_VERSION,
            record_type='evaluation_daily_change_series', seed=self.seed,
            steps=len(dates), dates=[day.isoformat() for day in dates],
            levels_end=levels.tolist(), release=releases.tolist(),
            level_timestamp='levels_end is the actual end-of-day level after each dated daily action',
            release_timestamp='release is the actual total daily release for each dated action',
            difference_rule='current minus previous in episode order; first daily difference is missing; '
                            'differences continue across source-year joins')
