"""Render eight observational correctness figures using NumPy and Matplotlib only."""
import argparse
import calendar
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np

FIGURE_KEYS = ('mass_balance', 'water_levels', 'release_feasibility',
               'storage_change', 'reward_reconciliation', 'happo_updates',
               'operation_trajectories', 'training_reliability')
IDS = ('WDD', 'BHT', 'XLD', 'XJB', 'THR')
VECTOR_FIELDS = ('storage_start', 'storage_end', 'levels_start', 'levels_end',
    'external', 'inflow', 'release', 'low', 'high', 'controls', 'power', 'energy',
    'turbine', 'spill', 'net_head_m', 'local_generation_rewards',
    'local_reward_penalties', 'local_rewards', 'mixed_rewards')
SCALAR_REWARDS = ('reward', 'generation_reward', 'reward_penalty', 'global_reward', 'system_reward')
QUANTILES = ('min', 'median', 'p95', 'max')


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(name+' must be a finite array with shape '+str(shape)) from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite array with shape '+str(shape))
    return result


def _finite_tree(value, name='evidence'):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, name+'.'+str(key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item, name)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(name+' contains a non-finite value')


def validate_payload(payload):
    """Reject malformed evidence before allocating figures; preserve source values."""
    if not isinstance(payload, dict) or payload.get('schema') != 'cascade_correctness_v2':
        raise ValueError('Expected evidence schema cascade_correctness_v2')
    _finite_tree(payload)
    runs = payload.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('Evidence must contain a nonempty runs list')
    seeds = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError('Every run must be an evidence object')
        seed = run.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in seeds:
            raise ValueError('Every run requires a distinct integer seed')
        seeds.add(seed)
        trajectory = run.get('trajectory')
        if not isinstance(trajectory, dict) or trajectory.get('seed') != seed:
            raise ValueError('Each run requires trajectory evidence with the same seed')
        steps = trajectory.get('steps')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('trajectory.steps must be a nonnegative integer')
        if trajectory.get('reservoir_ids') != list(IDS):
            raise ValueError('Trajectory reservoir_ids must be WDD/BHT/XLD/XJB/THR')
        physical = trajectory['physical']
        if not isinstance(physical, dict):
            raise ValueError('trajectory.physical must be an evidence object')
        for key in ('power_cap_mw', 'release_cap_m3s', 'turbine_cap_m3s',
                    'storage_min_m3', 'storage_max_m3', 'ecological_flow_m3s'):
            _array(physical[key], (5,), 'physical.'+key)
        if steps:
            series, independent = trajectory['series'], trajectory['independent']
            if len(series['date']) != steps:
                raise ValueError('series.date length must equal trajectory.steps')
            for key in VECTOR_FIELDS:
                _array(series[key], (steps, 5), 'series.'+key)
            for key in SCALAR_REWARDS:
                _array(series[key], (steps,), 'series.'+key)
            for key in ('mass_balance_residual_m3', 'mass_balance_tolerance_m3'):
                _array(independent[key], (steps, 5), 'independent.'+key)
            for key in ('cascade_mass_balance_residual_m3', 'cascade_mass_balance_tolerance_m3'):
                _array(independent[key], (steps,), 'independent.'+key)
            if (np.asarray(independent['mass_balance_tolerance_m3']) <= 0).any():
                raise ValueError('Independent mass-balance tolerances must be positive')
            if (np.asarray(independent['cascade_mass_balance_tolerance_m3']) <= 0).any():
                raise ValueError('Independent whole-cascade mass-balance tolerances must be positive')
            for key in SCALAR_REWARDS:
                _array(independent[key], (steps,), 'independent.'+key)
                _array(trajectory['reward_differences'][key], (steps,), 'reward_differences.'+key)
            for key in ('local_generation_rewards', 'local_reward_penalties', 'local_rewards', 'mixed_rewards'):
                _array(independent[key], (steps, 5), 'independent.'+key)
                _array(trajectory['reward_differences'][key], (steps, 5), 'reward_differences.'+key)
            for key in ('next_state_level_lower_m', 'next_state_level_upper_m',
                        'physical_low_m3s', 'physical_high_m3s'):
                rows = series.get(key)
                if rows is not None and any(row is not None for row in rows):
                    _array(rows, (steps, 5), 'series.'+key)
        for record in run.get('storage_records', []):
            if record.get('available', False):
                for key in ('initial_storage_m3', 'final_storage_m3', 'storage_change_m3'):
                    _array(record[key], (5,), 'storage_records.'+key)
        previous = 0
        for record in run.get('update_records', []):
            number = record['update']
            if isinstance(number, bool) or not isinstance(number, int) or number <= previous:
                raise ValueError('HAPPO update numbers must be positive and strictly increasing')
            previous = number
            actors = record['actors']
            if sorted(actor['agent_id'] for actor in actors) != list(range(5)):
                raise ValueError('Each HAPPO update requires five distinct actor summaries')
            for actor in actors:
                for key in ('used_factor', 'expected_factor', 'final_sampled_ratio'):
                    stats = actor[key]
                    if stats.get('count', 0):
                        _array([stats[q] for q in QUANTILES], (4,), key)
                for key in ('factor_max_abs_error', 'factor_max_relative_error'):
                    _array(actor[key], (), key)
        critic_quality = run.get('critic_quality')
        if critic_quality is not None:
            if not isinstance(critic_quality, dict) or not isinstance(critic_quality.get('channels', []), list):
                raise ValueError('critic_quality must be an evidence object with a channels list')
            previous_channel = -1
            for channel in critic_quality.get('channels', []):
                number = channel.get('value_channel') if isinstance(channel, dict) else None
                if (isinstance(number, bool) or not isinstance(number, int) or number < 0 or
                        number <= previous_channel):
                    raise ValueError('Critic value channels must be distinct nonnegative integers')
                previous_channel = number
                for name in ('critic_mc_pre', 'critic_mc_post'):
                    metrics = channel.get(name)
                    if metrics is not None:
                        if not isinstance(metrics, dict):
                            raise ValueError(name+' must be a metrics object')
                        for key, value in metrics.items():
                            if value is not None:
                                _array(value, (), name+'.'+key)
        previous_round = -1
        for record in run.get('evaluation_records', []):
            if not isinstance(record, dict):
                raise ValueError('Every evaluation record must be an object')
            number = record.get('evaluation_round')
            if (isinstance(number, bool) or not isinstance(number, int) or
                    number < 0 or number <= previous_round):
                raise ValueError('Evaluation rounds must be strictly increasing nonnegative integers')
            previous_round = number
            for key in ('training_steps', 'completed_window_count', 'successful_window_count', 'window_count'):
                value = record.get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                    raise ValueError('evaluation_records.'+key+' must be a nonnegative integer or null')
            for key in ('evaluator_return', 'ecological_flow_violation_rate'):
                value = record.get(key)
                if value is not None:
                    _array(value, (), 'evaluation_records.'+key)
    return runs


def _panels(plt, runs, title, rows=5):
    figure, axes = plt.subplots(rows, len(runs), figsize=(5.2*len(runs), 2.8*rows),
                               squeeze=False, constrained_layout=True)
    figure.suptitle(title, fontsize=14)
    for column, run in enumerate(runs):
        trajectory = run['trajectory']
        status = trajectory.get('status', 'unknown')
        count = trajectory['steps']
        expected = trajectory.get('expected_steps', count)
        context = 'Seed {} | {}\n{}/{} executed days ({})'.format(
            run['seed'], trajectory.get('scenario_id', 'selected episode'), count, expected, status)
        for row in range(rows):
            ax = axes[row, column]
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(context, fontsize=9)
            if row == rows-1:
                ax.set_xlabel('Executed day (source dates retained)')
            if not count:
                ax.text(.5, .5, 'Unavailable: no executed steps\n'+status,
                        ha='center', va='center', transform=ax.transAxes)
    return figure, axes


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, loc='best')


def _likelihood_axis(ax, values):
    """Show small changes around one in ordinary decimal likelihood units."""
    values = np.asarray([value for value in values if value is not None], dtype=float)
    values = values[np.isfinite(values)]
    wide = False
    if values.size:
        low, high = float(values.min()), float(values.max())
        wide = high/low > 100. if low > 0. else high > 100.
    if wide:
        ax.set_yscale('symlog', linthresh=.1)
    else:
        from matplotlib.ticker import MaxNLocator
        ax.set_yscale('linear')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.axhline(1., color='black', ls='--', lw=.6, label='_nolegend_')


def _operation_series(run):
    """Map retained calendar dates to five annual segments, including leap days.

    Levels are end-of-action-day values; releases are averages over that day.
    Earlier evidence may lack the full-calendar replay. Its recorded prefix is
    usable, but missing days and December 31 markers are never manufactured.
    """
    trajectory = run.get('operation_trajectory')
    if trajectory is None:
        trajectory = run['trajectory']
    if not isinstance(trajectory, dict):
        raise ValueError('operation_trajectory must be an evidence object')
    count = trajectory.get('steps', 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('operation_trajectory.steps must be a nonnegative integer')
    series = trajectory.get('series', {})
    raw_dates = series.get('date', [])
    if len(raw_dates) != count:
        raise ValueError('Operation dates must match the recorded step count')
    try:
        dates = [date.fromisoformat(value) for value in raw_dates]
    except (TypeError, ValueError) as exc:
        raise ValueError('Operation dates must use the YYYY-MM-DD calendar format') from exc
    years = list(trajectory.get('source_years', []))
    if not years and dates:
        years = list(dict.fromkeys(day.year for day in dates))
    if (len(years) > 5 or len(set(years)) != len(years) or
            any(isinstance(year, bool) or not isinstance(year, int) or
                not 1 <= year <= 9999 for year in years)):
        raise ValueError('Operation source_years must contain at most five distinct calendar years')
    # A partial/legacy trace still has a five-year display, with its unexecuted
    # suffix left blank. These inferred years supply axis labels only.
    while years and len(years) < 5:
        years.append(years[-1]+1)
    x_end, x_mid = [], []
    for day in dates:
        if day.year not in years:
            raise ValueError('An operation date falls outside the five annual segments')
        segment = years.index(day.year)
        year_days = 366 if calendar.isleap(day.year) else 365
        ordinal = (day-date(day.year, 1, 1)).days+1
        x_end.append(segment+ordinal/year_days)
        x_mid.append(segment+(ordinal-.5)/year_days)
    if dates:
        for previous, current in zip(dates, dates[1:]):
            same_year = current.year == previous.year
            joined_year = (years.index(current.year) == years.index(previous.year)+1 and
                           (previous.month, previous.day) == (12, 31) and
                           (current.month, current.day) == (1, 1))
            if not ((same_year and (current-previous).days == 1) or joined_year):
                raise ValueError('Operation dates must be daily and ordered within annual segments')
        levels = _array(series['levels_end'], (count, 5), 'operation levels_end')
        starts = _array(series['levels_start'], (count, 5), 'operation levels_start')
        release = _array(series['release'], (count, 5), 'operation release')
        initial_x = x_end[0]-1/(366 if calendar.isleap(dates[0].year) else 365)
    else:
        levels = starts = release = np.empty((0, 5))
        initial_x = 0.
    return dict(trajectory=trajectory, years=years, dates=dates,
                x_end=np.asarray(x_end), x_mid=np.asarray(x_mid), initial_x=initial_x,
                levels=levels, starts=starts, release=release,
                december31=np.asarray([(day.month, day.day) == (12, 31) for day in dates], dtype=bool))


def make_operation_figure(payload):
    """Plot the final HAPPO policy over exactly five calendar-year segments."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    runs = payload['runs']
    prepared = [_operation_series(run) for run in runs]
    level_color, marker_color = '#ed858f', '#bc5663'
    with plt.rc_context({'font.family': 'serif', 'font.size': 9}):
        figure, axes = plt.subplots(5, len(runs), figsize=(6.1*len(runs), 10.5),
                                   squeeze=False, sharex=True)
        figure.subplots_adjust(left=.085 if len(runs) == 1 else .055,
                               right=.88 if len(runs) == 1 else .95,
                               bottom=.13, top=.88, hspace=.16, wspace=.36)
        figure.suptitle('HAPPO: five-year reservoir operation trajectories', y=.985, fontsize=14)
        figure.legend(handles=[
            Line2D([], [], color=level_color, lw=1.1, label='Daily water level (left axis)'),
            Line2D([], [], color=marker_color, lw=.8, ls='--', label='Daily release (right axis)'),
            Line2D([], [], color=marker_color, marker='o', lw=0,
                   label='December 31 end-of-day water level')],
            loc='upper center', bbox_to_anchor=(.5, .96), ncol=3, frameon=False, fontsize=9)
        for column, (run, data) in enumerate(zip(runs, prepared)):
            trajectory, years = data['trajectory'], data['years']
            full_days = sum(366 if calendar.isleap(year) else 365 for year in years)
            complete = bool(len(data['dates']) == full_days and len(years) == 5 and
                            data['december31'].sum() == 5)
            context = 'Seed {} | {}'.format(run['seed'], trajectory.get('scenario_id', 'selected scenario'))
            if not complete:
                context += '\n{} / {} recorded days ({})'.format(
                    len(data['dates']), full_days if years else 'five years',
                    trajectory.get('status', 'partial'))
            ticks, tick_labels = [], []
            for segment, year in enumerate(years):
                year_days = 366 if calendar.isleap(year) else 365
                ticks.extend((segment, segment+(date(year, 7, 1)-date(year, 1, 1)).days/year_days))
                tick_labels.extend(('Jan', 'Jul'))
            ticks.append(5.)
            tick_labels.append('Jan')
            for i, reservoir in enumerate(IDS):
                ax = axes[i, column]
                flow = ax.twinx()
                ax.set_gid('operation_level_seed{}_{}'.format(run['seed'], reservoir))
                flow.set_gid('operation_release_seed{}_{}'.format(run['seed'], reservoir))
                ax.spines['top'].set_visible(False)
                flow.spines['top'].set_visible(False)
                ax.grid(axis='y', alpha=.25, ls=':', lw=.6)
                ax.set_ylabel(reservoir+'\nWater level (m)', fontsize=10)
                flow.set_ylabel('Release (m³/s)', fontsize=9)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
                flow.yaxis.set_major_locator(MaxNLocator(nbins=4))
                flow.ticklabel_format(axis='y', style='plain', useOffset=False)
                for segment, year in enumerate(years):
                    year_days = 366 if calendar.isleap(year) else 365
                    flood_start = segment+(date(year, 6, 1)-date(year, 1, 1)).days/year_days
                    flood_end = segment+(date(year, 10, 1)-date(year, 1, 1)).days/year_days
                    ax.axvspan(flood_start, flood_end, color='gray', alpha=.10, lw=0)
                for boundary in range(1, 6):
                    ax.axvline(boundary, color=marker_color, ls='--', lw=.55, alpha=.45)
                if data['dates']:
                    level_values = np.r_[data['starts'][0, i], data['levels'][:, i]]
                    level_line, = ax.plot(np.r_[data['initial_x'], data['x_end']], level_values,
                                         color=level_color, lw=1.1, label='Daily water level')
                    level_line.set_gid('water_level')
                    release_line, = flow.plot(data['x_mid'], data['release'][:, i],
                                             color=marker_color, lw=.7, ls='--', alpha=.85,
                                             label='Daily release')
                    release_line.set_gid('release_discharge')
                    flow.set_ylim(max(float(data['release'][:, i].max())*1.12, 1.), 0.)
                    mask = data['december31']
                    markers = ax.scatter(data['x_end'][mask], data['levels'][mask, i],
                                         s=27, facecolors=marker_color, edgecolors=marker_color,
                                         linewidths=.5, zorder=5, label='December 31 water level')
                    markers.set_gid('december31_levels')
                    for x, y in zip(data['x_end'][mask], data['levels'][mask, i]):
                        ax.annotate('{:.1f} m'.format(y), (x, y), xytext=(-3 if x == 5. else 0, 9),
                                    textcoords='offset points', color=marker_color, fontsize=7,
                                    ha='right' if x == 5. else 'center', annotation_clip=False)
                    # Leave room for labels without imposing any operating target.
                    span = max(float(np.ptp(level_values)), 1.)
                    ax.set_ylim(float(level_values.min())-.08*span,
                                float(level_values.max())+.22*span)
                else:
                    flow.set_ylim(1., 0.)
                    ax.text(.5, .5, 'Unavailable: '+trajectory.get('reason', trajectory.get('status', 'no executed days')),
                            transform=ax.transAxes, ha='center', va='center', fontsize=8, wrap=True)
                ax.set_xlim(0., 5.)
                ax.set_xticks(ticks, tick_labels, fontsize=8)
                ax.text(.01, .04, '({})'.format(chr(ord('a')+i+5*column)),
                        transform=ax.transAxes, color=marker_color, fontsize=10, fontweight='bold')
                if i == 0:
                    ax.set_title(context, fontsize=10, color=marker_color, pad=9)
                if i == 4:
                    for segment in range(5):
                        label = 'Year {}\n{}'.format(segment+1, years[segment]) if years else 'Year '+str(segment+1)
                        ax.text(segment+.5, -.22, label, ha='center', va='top',
                                transform=ax.get_xaxis_transform(), fontsize=8)
            axes[-1, column].set_xlabel('Five annual operation segments', labelpad=31)
        figure.text(.5, .015,
                     'Gray bands: June–September. Release axis is reversed. '
                     'Markers use executed December 31 levels; unrecorded days remain blank.',
                     ha='center', fontsize=8)
    return figure


def _critic_channel_label(channel):
    agents = [agent for agent in channel.get('agent_ids', []) if 0 <= agent < len(IDS)]
    sources = channel.get('learning_reward_sources', [])
    if any(str(source).startswith('mixed_rewards') for source in sources) and agents:
        return 'Mixed reward ('+', '.join(IDS[agent] for agent in agents)+')'
    if any(str(source) == 'system_reward' for source in sources):
        return 'System reward (shared)'
    if sources:
        return str(sources[0])
    return 'Value channel '+str(channel.get('value_channel', '?'))


def make_training_reliability_figure(payload):
    """Show Critic fit and fixed-scenario histories without changing training."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    runs = payload['runs']
    with plt.rc_context({'font.size': 9}):
        figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
        figure.suptitle('HAPPO training reliability: Critic fit and fixed-scenario convergence', fontsize=14)
        critic_axes = axes[0]
        for ax, ylabel in zip(critic_axes, (
                'Explained variance', 'Normalized RMSE', 'Normalized bias')):
            ax.grid(alpha=.22)
            ax.set_xlabel('Completed policy update')
            ax.set_ylabel(ylabel)
        critic_axes[0].set_title('Critic fit to realized collected returns')
        critic_axes[1].set_title('Critic error relative to return spread')
        critic_axes[2].set_title('Signed Critic bias relative to return spread')
        critic_axes[0].axhline(1., color='black', ls='--', lw=.7, label='Perfect prediction')
        critic_axes[0].axhline(0., color='gray', ls=':', lw=.7, label='No explained variance')
        critic_axes[1].axhline(0., color='black', ls='--', lw=.7, label='Zero error')
        critic_axes[2].axhline(0., color='black', ls='--', lw=.7, label='Zero bias')
        colors = plt.get_cmap('tab10')
        critic_available = False

        def column_summary(values):
            values = np.asarray(values, dtype=float)
            valid = np.isfinite(values)
            count = valid.sum(axis=0)
            mean = np.divide(np.nansum(values, axis=0), count,
                             out=np.full(values.shape[1], np.nan), where=count > 0)
            low = np.where(valid, values, np.inf).min(axis=0)
            high = np.where(valid, values, -np.inf).max(axis=0)
            low[count == 0], high[count == 0] = np.nan, np.nan
            return mean, low, high

        channel_numbers = sorted({channel.get('value_channel') for run in runs
            for channel in run.get('critic_quality', {}).get('channels', [])})
        for channel_number in channel_numbers:
            metadata = next((channel for run in runs
                for channel in run.get('critic_quality', {}).get('channels', [])
                if channel.get('value_channel') == channel_number), None)
            if metadata is None:
                continue
            histories = []
            for run in runs:
                history = {}
                for record in run.get('update_records', []):
                    item = record.get('critic_quality')
                    if not isinstance(item, dict):
                        continue
                    match = next((value for value in item.get('channels', [])
                                  if value.get('value_channel') == channel_number), None)
                    if match is not None:
                        history[record['update']] = match
                if history:
                    histories.append(history)
            if not histories:
                continue
            critic_available = True
            label = _critic_channel_label(metadata)
            color = colors(channel_number % 10)
            updates = sorted({update for history in histories for update in history})
            x = np.asarray(updates)
            for axis, key in ((critic_axes[0], 'explained_variance'),
                              (critic_axes[1], 'normalized_rmse'),
                              (critic_axes[2], 'normalized_bias')):
                matrix = np.asarray([[history.get(update, {}).get('critic_mc_post', {}).get(key)
                                     if history.get(update, {}).get('critic_mc_post') else np.nan
                                     for update in updates] for history in histories], dtype=float)
                mean, low, high = column_summary(matrix)
                axis.plot(x, mean, color=color, lw=1., marker='o', ms=2.5,
                          label=label if axis is critic_axes[0] else None)
                if len(histories) > 1:
                    axis.fill_between(x, low, high, color=color, alpha=.12, lw=0)
                if key == 'explained_variance':
                    pre_matrix = np.asarray([[history.get(update, {}).get('critic_mc_pre', {}).get(key)
                                              if history.get(update, {}).get('critic_mc_pre') else np.nan
                                              for update in updates] for history in histories], dtype=float)
                    pre_mean, _, _ = column_summary(pre_matrix)
                    axis.plot(x, pre_mean, color=color, lw=.7, ls=':', alpha=.7)
        if not critic_available:
            for ax in critic_axes:
                ax.text(.5, .5, 'Unavailable: no Critic fit history', ha='center', va='center',
                        transform=ax.transAxes)
        critic_axes[0].text(.02, .02, 'Solid: post-update mean; band: seed range; dotted: pre-update',
                            transform=critic_axes[0].transAxes, fontsize=7)
        for ax in critic_axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=6, ncol=2, loc='best')

        convergence = axes[1]
        for ax, ylabel, title in zip(convergence, (
                'Fixed-scenario return', 'Ecological-flow violation rate (%)', 'Return of final five rounds'),
                ('Fixed-scenario evaluator return', 'Ecological-flow constraint frequency',
                 'Cross-seed final return spread')):
            ax.grid(alpha=.22)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        convergence[0].set_xlabel('Evaluation round (policy update)')
        convergence[1].set_xlabel('Evaluation round (policy update)')
        convergence[2].set_xlabel('Seed and cross-seed aggregate')
        evaluation_available = False
        final_returns = []
        for run_index, run in enumerate(runs):
            records = [record for record in run.get('evaluation_records', [])
                       if record.get('evaluator_return') is not None]
            if not records:
                continue
            evaluation_available = True
            color = colors(run_index % 10)
            rounds = np.asarray([record['evaluation_round'] for record in records])
            returns = np.asarray([record['evaluator_return'] for record in records], dtype=float)
            convergence[0].plot(rounds, returns, color=color, lw=1., marker='o', ms=2.5,
                                label='Seed '+str(run['seed']))
            ecology = [(record['evaluation_round'], record['ecological_flow_violation_rate'])
                       for record in records if record.get('ecological_flow_violation_rate') is not None]
            if ecology:
                convergence[1].plot([item[0] for item in ecology],
                                    [100.*item[1] for item in ecology], color=color,
                                    lw=1., marker='o', ms=2.5, label='Seed '+str(run['seed']))
            final_returns.append((run['seed'], float(np.mean(returns[-5:])), color))
        if not evaluation_available:
            convergence[0].text(.5, .5, 'Unavailable: no evaluation history', ha='center', va='center',
                                transform=convergence[0].transAxes)
        if not any(row.get('ecological_flow_violation_rate') is not None
                   for run in runs for row in run.get('evaluation_records', [])):
            convergence[1].text(.5, .5, 'Unavailable: no ecological-flow history', ha='center',
                                va='center', transform=convergence[1].transAxes)
        convergence[1].yaxis.set_major_formatter(PercentFormatter(100.))
        convergence[1].axhline(0., color='gray', ls=':', lw=.7)
        for position, (seed, value, color) in enumerate(final_returns, 1):
            convergence[2].scatter(position, value, s=35, color=color, label='Seed '+str(seed))
            convergence[2].annotate('{:.2f}'.format(value), (position, value), xytext=(3, 4),
                                    textcoords='offset points', fontsize=7)
        if final_returns:
            values = np.asarray([value for _, value, _ in final_returns])
            mean, spread = float(values.mean()), float(values.std(ddof=0))
            convergence[2].errorbar(0., mean, yerr=spread, fmt='s', color='black', capsize=4,
                                    label='Cross-seed mean ± SD')
            convergence[2].axhline(mean, color='black', ls='--', lw=.7)
            convergence[2].set_xticks(range(len(final_returns)+1),
                ['Mean']+['Seed '+str(seed) for seed, _, _ in final_returns])
            if len(values) < 2:
                convergence[2].text(.02, .98, 'SD requires at least two seeds',
                                    transform=convergence[2].transAxes, va='top', fontsize=7)
        else:
            convergence[2].text(.5, .5, 'Unavailable: fewer than five evaluator rounds', ha='center',
                                va='center', transform=convergence[2].transAxes)
        for ax in convergence:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7, loc='best')
    return figure


def make_figures(payload):
    """Return eight figures from saved evidence without model imports or mutation."""
    try:
        runs = validate_payload(payload)
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or malformed correctness evidence: '+str(exc)) from exc
    import matplotlib.pyplot as plt
    figures = {}
    titles = dict(mass_balance='Independent daily mass balance: residual and numerical tolerance',
        water_levels='Executed water levels and permitted next-state limits',
        release_feasibility='Executed release, physical / mapping bounds and ecological requirement',
        reward_reconciliation='Reported reward components and independent reconciliation')
    for key, title in titles.items():
        figures[key], axes = _panels(plt, runs, title,
                                    rows=6 if key in ('mass_balance', 'reward_reconciliation') else 5)
        if key == 'release_feasibility':
            figures[key].supxlabel(
                'Forced markers denote a collapsed physical release interval.', fontsize=8)
        for column, run in enumerate(runs):
            trajectory = run['trajectory']
            count = trajectory['steps']
            if not count:
                continue
            x = np.arange(1, count+1)
            series, check, physical = trajectory['series'], trajectory['independent'], trajectory['physical']
            for i, reservoir in enumerate(IDS):
                ax = axes[i, column]
                if key == 'mass_balance':
                    residual = np.asarray(check['mass_balance_residual_m3'])[:, i]
                    tolerance = np.asarray(check['mass_balance_tolerance_m3'])[:, i]
                    ax.plot(x, residual, lw=.8, label='Independent residual')
                    ax.plot(x, tolerance, color='darkorange', lw=.7, label='+ tolerance')
                    ax.plot(x, -tolerance, color='darkorange', lw=.7, ls='--', label='− tolerance')
                    ax.set_yscale('symlog', linthresh=max(float(tolerance.max())*.01, 1e-12))
                    ax.set_ylabel(reservoir+' balance (m³)')
                    if i == 0:
                        maximum = trajectory.get('summary', {}).get('max_abs_cascade_mass_balance_residual_m3')
                        if maximum is not None:
                            ax.text(.02, .96, 'Max whole-cascade residual: {:.3g} m³'.format(maximum),
                                    va='top', transform=ax.transAxes, fontsize=7)
                elif key == 'water_levels':
                    ax.plot(x, np.asarray(series['levels_end'])[:, i], lw=1, label='Executed next-state level')
                    lower, upper = series.get('next_state_level_lower_m'), series.get('next_state_level_upper_m')
                    if lower is not None and all(row is not None for row in lower):
                        ax.plot(x, np.asarray(lower)[:, i], color='gray', lw=.7, label='Lower bound')
                        ax.plot(x, np.asarray(upper)[:, i], color='gray', lw=.7, ls='--', label='Upper bound')
                    else:
                        bounds = payload.get('configuration', {}).get('physical_config', {}).get('level_bounds_m', {}).get(reservoir)
                        if bounds is not None:
                            ax.axhline(bounds[0], color='gray', label='Global lower bound')
                            ax.axhline(bounds[1], color='gray', ls='--', label='Global upper bound')
                    ax.set_ylabel(reservoir+' level (m)')
                elif key == 'release_feasibility':
                    low, high, release = (np.asarray(series[name])[:, i] for name in ('low', 'high', 'release'))
                    physical_low, physical_high = series.get('physical_low_m3s'), series.get('physical_high_m3s')
                    if physical_low is not None and physical_high is not None and all(
                            row is not None for row in physical_low+physical_high):
                        ax.fill_between(x, np.asarray(physical_low)[:, i], np.asarray(physical_high)[:, i],
                                        alpha=.12, label='Physical release bounds')
                    else:
                        ax.fill_between(x, low, high, alpha=.12,
                                        label='Dispatcher bounds (before forced override)')
                    ax.plot(x, release, lw=.9, label='Executed release')
                    ax.plot(x, low, color='gray', lw=.5, label='Mapping lower bound')
                    ax.plot(x, high, color='gray', lw=.5, ls='--', label='Mapping upper bound')
                    threshold = physical['ecological_flow_m3s'][i]
                    if threshold > 0:
                        ax.axhline(threshold, color='green', ls=':', label='Ecological requirement')
                    forced = np.asarray(series['controls'])[:, i] == 0
                    ax.scatter(x[forced], release[forced], marker='x', s=12, color='red',
                               label='Forced action ({} days)'.format(int(forced.sum())))
                    ax.set_ylabel(reservoir+' release (m³/s)')
                else:
                    for name, sign, label in (('local_generation_rewards', 1, 'Local generation'),
                        ('local_reward_penalties', -1, 'Ecological cost (negative)'), ('mixed_rewards', 1, 'Mixed reward')):
                        ax.plot(x, sign*np.asarray(series[name])[:, i], lw=.8, label=label)
                    difference = np.asarray(trajectory['reward_differences']['mixed_rewards'])[:, i]
                    error = ax.twinx()
                    error.plot(x, difference, color='red', ls=':', lw=.7, label='Reported − recomputed mixed reward')
                    error.set_ylabel('Reconciliation error', fontsize=7)
                    error.tick_params(labelsize=7)
                    error.legend(fontsize=6, loc='upper right')
                    ax.set_ylabel(reservoir+' reward')
                if key == 'reward_reconciliation':
                    ax.legend(fontsize=6, loc='lower left')
                else:
                    _legend(ax)
            if key == 'mass_balance':
                ax = axes[5, column]
                residual = np.asarray(check['cascade_mass_balance_residual_m3'])
                tolerance = np.asarray(check['cascade_mass_balance_tolerance_m3'])
                ax.plot(x, residual, lw=.8, label='Independent whole-cascade residual')
                ax.plot(x, tolerance, color='darkorange', lw=.7, label='+ tolerance')
                ax.plot(x, -tolerance, color='darkorange', lw=.7, ls='--', label='− tolerance')
                ax.set_yscale('symlog', linthresh=max(float(tolerance.max())*.01, 1e-12))
                ax.set_ylabel('Whole-cascade balance (m³)')
                _legend(ax)
            elif key == 'reward_reconciliation':
                ax = axes[5, column]
                for name, label in (('global_reward', 'Global generation'), ('system_reward', 'System reward'), ('reward', 'Legacy-team reward')):
                    ax.plot(x, series[name], lw=.9, label=label)
                    ax.plot(x, check[name], lw=.6, ls='--', label='Independent '+label.lower())
                ax.set_ylabel('System reward accounting')
                _legend(ax)
                maximum = trajectory.get('summary', {}).get('max_abs_reward_difference')
                if maximum is not None:
                    ax.text(.02, .03, 'Max reconciliation error (all rewards): {:.3g}'.format(maximum),
                            transform=ax.transAxes, fontsize=7)
    figures['storage_change'], axes = _panels(plt, runs,
        'Endpoint storage change: final batch / final evaluations (inventory accounting)')
    for column, run in enumerate(runs):
        records = []
        counts = []
        for phase, label in (('training', 'Train'), ('training_evaluation', 'Eval'), ('evaluation', 'Held-out')):
            history = [record for record in run.get('storage_records', []) if record.get('phase') == phase]
            counts.append('{}: {} inventories'.format(label, sum(record.get('available', False) for record in history)))
            if history:
                latest = max(record.get('policy_update', 0) for record in history)
                records.extend(record for record in history if phase == 'evaluation' or
                               record.get('policy_update', 0) == latest)
        labels = []
        for record in records:
            phase = {'training': 'Train', 'training_evaluation': 'Eval',
                     'evaluation': 'Held-out'}.get(record.get('phase'), 'Case')
            scenario = record.get('scenario_id', 'case').replace('collection_', '').replace('episode_', 'ep')
            if scenario == 'held_out_sequence':
                scenario = 'sequence'
            labels.append('{} {}\nu{}'.format(phase, scenario, record.get('policy_update', '?')))
        for i, reservoir in enumerate(IDS):
            ax = axes[i, column]
            ax.clear()
            ax.grid(alpha=.2)
            if i == 0:
                ax.set_title('Seed {} | all saved endpoints\n{}'.format(run['seed'], ' | '.join(counts)), fontsize=8)
            ax.set_ylabel(reservoir+' final − initial (billion m³)')
            ax.axhline(0., color='gray', ls='--', lw=.7, label='Zero change (reference)')
            for position, record in enumerate(records):
                if record.get('available', False):
                    ax.scatter(position, record['storage_change_m3'][i]/1e9, s=24, color='tab:blue')
                else:
                    ax.text(position, 0, 'Unavailable / '+record.get('status', 'partial'),
                            rotation=90, fontsize=7, ha='center')
            if not records:
                ax.text(.5, .5, 'Unavailable: no storage inventories', ha='center', transform=ax.transAxes)
            ax.set_xticks(range(len(records)), labels, rotation=35, ha='right', fontsize=7)
            if i == 4:
                ax.set_xlabel('Recorded episode / scenario; positive = accumulation, negative = depletion')
            _legend(ax)
    figures['happo_updates'], axes = _panels(plt, runs,
        'HAPPO independent likelihood-factor checks and policy-update diagnostics', rows=6)
    colors = plt.get_cmap('tab10')
    for column, run in enumerate(runs):
        records = run.get('update_records', [])
        likelihood_values = {0: [], 2: []}
        for row, ylabel in enumerate(('Likelihood factor quantiles', 'Maximum factor error',
            'Final sampled likelihood ratio', 'Policy KL', 'Clip fraction', 'Policy entropy')):
            ax = axes[row, column]
            ax.clear()
            ax.grid(alpha=.2)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Completed policy update')
            if row == 0:
                ax.set_title('Seed '+str(run['seed']))
            if not records:
                ax.text(.5, .5, 'Unavailable: no HAPPO update evidence', ha='center', transform=ax.transAxes)
        for agent in range(5):
            actors = [next(actor for actor in record['actors'] if actor['agent_id'] == agent) for record in records]
            updates = np.asarray([record['update'] for record in records])
            if not actors:
                continue
            color = colors(agent)
            for quantile, marker in zip(QUANTILES, ('v', 'o', '^', 's')):
                offset = (agent-2)*.035
                used = [actor['used_factor'][quantile] for actor in actors]
                expected = [actor['expected_factor'][quantile] for actor in actors]
                likelihood_values[0].extend(used+expected)
                axes[0, column].scatter(updates+offset, used, s=15, marker=marker, color=color,
                    label=IDS[agent] if quantile == 'median' else '_nolegend_')
                axes[0, column].scatter(updates+offset, expected, s=30, marker=marker,
                    facecolors='none', edgecolors=color, label='_nolegend_')
            axes[1, column].plot(updates, [actor['factor_max_abs_error'] for actor in actors], color=color, lw=.8, label=IDS[agent]+' absolute')
            axes[1, column].plot(updates, [actor['factor_max_relative_error'] for actor in actors], color=color, lw=.7, ls=':', label=IDS[agent]+' relative')
            for quantile in ('min', 'median', 'p95', 'max'):
                values = [actor['final_sampled_ratio'].get(quantile) for actor in actors]
                likelihood_values[2].extend(values)
                axes[2, column].plot(updates, [np.nan if value is None else value for value in values],
                    color=color, lw=.7, ls='-' if quantile == 'median' else ':',
                    label=IDS[agent] if quantile == 'median' else '_nolegend_')
            for row, field in ((3, 'final_policy_kl'), (4, 'final_clip_fraction'), (5, 'final_entropy')):
                axes[row, column].plot(updates, [np.nan if actor.get(field) is None else actor[field] for actor in actors],
                    color=color, lw=.9, label=IDS[agent])
        axes[1, column].set_yscale('symlog', linthresh=1e-8)
        for row in (0, 2):
            _likelihood_axis(axes[row, column], likelihood_values[row])
        for row in range(6):
            handles, labels = axes[row, column].get_legend_handles_labels()
            if handles:
                axes[row, column].legend(handles, labels, fontsize=6, ncol=2 if row == 1 else 1,
                                         loc='best')
        axes[5, column].ticklabel_format(axis='y', style='plain', useOffset=False)
        axes[0, column].text(.02, .98, 'Filled: used; open: independent\nv=min, o=median, ^=p95, square=max\nColors: '+', '.join(IDS),
            transform=axes[0, column].transAxes, va='top', fontsize=6)
    figures['operation_trajectories'] = make_operation_figure(payload)
    figures['training_reliability'] = make_training_reliability_figure(payload)
    return {key: figures[key] for key in FIGURE_KEYS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-backend', action='store_true', help='Check headless Matplotlib, then exit')
    parser.add_argument('--input', type=Path, help='Evidence JSON; omit to read JSON from standard input')
    parser.add_argument('--output-dir', type=Path, help='Output directory; may contain input evidence, but eight PNG targets must be new')
    parser.add_argument('--no-show', action='store_true', help='Use Agg and save figures without opening windows')
    args = parser.parse_args(argv)
    import matplotlib
    if args.no_show or args.check_backend:
        matplotlib.use('Agg', force=True)
    if args.check_backend:
        import matplotlib.pyplot as plt
        print('Headless Matplotlib ready: '+matplotlib.get_backend())
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required when rendering evidence')
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error('--output-dir must name a directory, not an existing file: '+str(args.output_dir))
    targets = {key: args.output_dir/(key+'.png') for key in FIGURE_KEYS}
    for path in targets.values():
        if path.exists():
            parser.error('Output already exists: '+str(path))
        if args.input is not None and path.resolve() == args.input.resolve():
            parser.error('A figure cannot overwrite its evidence input')
    try:
        payload = json.loads(args.input.read_text(encoding='utf-8') if args.input is not None else sys.stdin.read())
        figures = make_figures(payload)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    try:
        for key, path in targets.items():
            with path.open('xb') as stream:
                figures[key].savefig(stream, format='png', dpi=125)
            print(str(path.resolve()))
        if not args.no_show:
            plt.show()
    finally:
        for figure in figures.values():
            plt.close(figure)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
