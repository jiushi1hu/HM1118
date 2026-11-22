"""Paired ablation of downstream-to-upstream release-feasibility propagation.

Runs HAPPO twice on identical settings and seeds. The only difference is
``feasibility_mode``: ``backward`` is the original kernel, ``forward_local``
disables the compatibility pass. Both arms use the same training windows,
rewards, GAE, action grid, daily level limits and ramp limits, and are
evaluated on the same five-year held-out sequence.

Writes a JSON summary, a Markdown report and three figures.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


MODES = ('backward', 'forward_local')


def _load(path):
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows


def _records(rows, record_type):
    return [row for row in rows if row.get('record_type') == record_type]


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _std(values):
    values = [float(value) for value in values if value is not None]
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def summarise_mode(rows):
    """Reduce one arm's JSONL log to comparable aggregates."""
    updates = _records(rows, 'training_update')
    episodes = _records(rows, 'training_episode')
    complete = _records(rows, 'run_complete')
    evaluations = complete[0]['evaluations'] if complete else []
    mode = (evaluations[0].get('feasibility_mode') if evaluations else None) or (
        episodes[0].get('feasibility_mode') if episodes else None)
    if mode is None:
        mode = updates[0].get('feasibility_mode') if updates else None

    curve = [dict(update=int(row['update']), seed=int(row['seed']),
                  team_reward=row.get('episode_reward_mean'),
                  generation_TWh=row.get('episode_generation_TWh_mean')) for row in updates]

    by_seed = {}
    for row in updates:
        by_seed.setdefault(int(row['seed']), []).append(row.get('episode_reward_mean'))

    # The logged training return is the legacy team metric. The learning
    # objective is the system reward, kept in each episode record.
    episodes_by_seed = {}
    for row in episodes:
        episodes_by_seed.setdefault(int(row['seed']), []).append(row)
    learning_curve = []
    for row in updates:
        seed = int(row['seed'])
        window = episodes_by_seed.get(seed, [])[
            (int(row['update'])-1)*row.get('parallel_envs', 1):
            int(row['update'])*row.get('parallel_envs', 1)]
        learning_curve.append(dict(update=int(row['update']), seed=seed,
            learning_return=_mean([item.get('learning_rewards', {}).get(
                'episode_system_reward') for item in window]),
            generation_TWh=row.get('episode_generation_TWh_mean')))
    learning_by_seed = {}
    for row in learning_curve:
        learning_by_seed.setdefault(row['seed'], []).append(row['learning_return'])

    ep_generation = [row.get('actual_generation_TWh') for row in episodes
                     if row.get('episode_generation_TWh') is not None]
    return dict(feasibility_mode=mode,
        training_updates=len(updates), training_episodes=len(episodes),
        seeds=sorted({int(row['seed']) for row in updates}),
        training_curve=curve,
        learning_curve=learning_curve,
        learning_curve_by_seed={str(seed): values for seed, values in sorted(learning_by_seed.items())},
        training_curve_by_seed={str(seed): values for seed, values in sorted(by_seed.items())},
        mean_training_return=_mean([row.get('episode_reward_mean') for row in updates]),
        mean_learning_return=_mean([row['learning_return'] for row in learning_curve]),
        final_training_return={str(seed): values[-1] for seed, values in sorted(by_seed.items())},
        mean_episode_generation_TWh=_mean(ep_generation),
        total_episode_generation_TWh=float(sum(ep_generation)) if ep_generation else 0.,
        forced_actor_days=sum(int(row.get('forced_actor_days', 0)) for row in episodes),
        empty_interval_days=sum(int(row.get('empty_interval_days', 0)) for row in episodes),
        infeasible_episodes=sum(int(bool(row.get('hard_constraint_failed'))) for row in episodes),
        max_balance_residual_m3=max([float(row.get('max_balance_residual_m3', 0.))
                                     for row in episodes] or [0.]),
        mean_release_interval_fraction=_mean([
            np.mean(row['mean_release_interval_fraction']) for row in episodes
            if row.get('mean_release_interval_fraction')]),
        training_evaluations=[dict(seed=int(record['seed']),
                                   update=int(record['evaluation_round']),
                                   system_return=window.get('system_evaluator_return'),
                                   generation_TWh=window.get('generation_TWh'),
                                   ecological_violation_rate=window.get('ecological_flow_violation_rate'))
                              for record in _records(rows, 'periodic_evaluation')
                              for window in record.get('windows', [])],
        final_evaluations=evaluations)


def _scalar(row, name):
    value = row.get(name)
    return None if value is None else float(value)


def compare(arms):
    """Build the paired comparison table."""
    table = []
    for key, label in (
            ('system_evaluator_return', 'five_year_discounted_return'),
            ('generation_TWh', 'five_year_generation_TWh'),
            ('energy_MWh', 'five_year_generation_MWh'),
            ('forced_actor_rate', 'evaluation_forced_actor_rate'),
            ('empty_interval_rate', 'evaluation_empty_interval_rate'),
            ('infeasible_interval_rate', 'evaluation_infeasible_interval_rate'),
            ('max_balance_residual_m3', 'evaluation_max_balance_residual_m3'),
            ('max_daily_bound_excess_m3', 'evaluation_max_daily_bound_excess_m3'),
            ('violation_count', 'evaluation_daily_bound_violations'),
            ('total_storage_depletion_billion_m3', 'evaluation_storage_depletion_billion_m3')):
        entry = dict(metric=label)
        for mode in MODES:
            values = [_scalar(row, key) for row in arms[mode]['final_evaluations']]
            entry[mode] = dict(mean=_mean(values), std=_std(values),
                               values=values)
        a, b = entry['backward'], entry['forward_local']
        entry['difference'] = (None if a['mean'] is None or b['mean'] is None
                               else b['mean'] - a['mean'])
        table.append(entry)
    return table


def trajectory_comparison(arms):
    """Compare per-seed level trajectories between the two arms."""
    result = {}
    for mode in MODES:
        series = [row['level_trajectory']['mean'] for row in arms[mode]['final_evaluations']
                  if row.get('level_trajectory')]
        result[mode] = np.asarray(series, dtype=np.float64) if series else None
    if all(result[mode] is not None for mode in MODES):
        a, b = result['backward'], result['forward_local']
        if a.shape == b.shape:
            result['max_abs_difference_m'] = float(np.abs(a-b).max())
    return result


def _level_extents(trajectory):
    """Reduce a stored level trajectory to start/end/min/max without keeping it."""
    mean = np.asarray(trajectory['mean'], dtype=np.float64)
    lowest = np.asarray(trajectory['min'], dtype=np.float64) if 'min' in trajectory else mean
    highest = np.asarray(trajectory['max'], dtype=np.float64) if 'max' in trajectory else mean
    return dict(mean_start_m=mean[0].tolist(), mean_end_m=mean[-1].tolist(),
                min_level_m=np.min(lowest, axis=0).tolist(),
                max_level_m=np.max(highest, axis=0).tolist())


def _plot_curves(arms, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for mode, colour in zip(MODES, ('#1f4e79', '#c55a11')):
        for seed, values in arms[mode]['learning_curve_by_seed'].items():
            axes[0].plot(range(1, len(values)+1), values, colour, alpha=.5, lw=1)
        seeds = [values for values in arms[mode]['learning_curve_by_seed'].values()
                 if all(value is not None for value in values)]
        if not seeds:
            seeds = list(arms[mode]['learning_curve_by_seed'].values())
        if seeds:
            length = min(len(v) for v in seeds)
            mean = np.mean([[v[i] for i in range(length)] for v in seeds], axis=0)
            axes[0].plot(range(1, length+1), mean, colour, lw=2.2, label=mode)
    axes[0].set_xlabel('HAPPO update')
    axes[0].set_ylabel('mean episode system return (learning objective)')
    axes[0].set_title('Training return (thin: seeds, thick: mean)')
    axes[0].legend()
    axes[0].grid(alpha=.25)

    for mode, colour in zip(MODES, ('#1f4e79', '#c55a11')):
        rows = arms[mode]['training_evaluations']
        if rows:
            axes[1].plot([r['update'] for r in rows], [r['system_return'] for r in rows],
                         colour, marker='o', ms=3, lw=1.5, label=mode)
    axes[1].set_xlabel('HAPPO update')
    axes[1].set_ylabel('fixed-scenario evaluator return')
    axes[1].set_title('Periodic evaluation')
    axes[1].grid(alpha=.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_levels(arms, trajectories, path, years):
    from harl.envs.cascade_reservoir.data_loader import IDS
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    days = np.arange(len(trajectories['backward'][0]) if trajectories['backward'] is not None else 0)
    for index, name in enumerate(IDS):
        ax = axes[index]
        for mode, colour in zip(MODES, ('#1f4e79', '#c55a11')):
            series = trajectories[mode]
            if series is None:
                continue
            for seed_index in range(series.shape[0]):
                ax.plot(days, series[seed_index, :, index], colour, lw=1., alpha=.45)
            ax.fill_between(days, series[:, :, index].min(axis=0), series[:, :, index].max(axis=0),
                            color=colour, alpha=.18)
            ax.plot(days, series[:, :, index].mean(axis=0), colour, lw=1.8, label=mode)
        ax.set_ylabel(name+' level (m)')
        ax.grid(alpha=.25)
        ax.legend(loc='upper right', fontsize=8)
    for boundary in range(1, len(years)):
        axes[0].axvline(boundary*365, color='grey', ls=':', lw=1)
    axes[0].set_title('Held-out five-year level trajectories ('+', '.join(map(str, years))+')')
    axes[-1].set_xlabel('day of evaluation sequence')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_feasibility(arms, path):
    labels = ['forced actor-days', 'empty intervals', 'bound violations',
              'infeasible episodes']
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    width = .36
    for offset, (mode, colour) in enumerate(zip(MODES, ('#1f4e79', '#c55a11'))):
        arm = arms[mode]
        values = [arm['forced_actor_days'], arm['empty_interval_days'],
                  sum(int(row.get('violation_count') or 0) for row in arm['final_evaluations']),
                  arm['infeasible_episodes']]
        ax.bar(np.arange(len(labels))+(offset-.5)*width, values, width, color=colour, label=mode)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('count (training + final evaluation)')
    ax.set_title('Constraint and feasibility accounting')
    ax.legend()
    ax.grid(axis='y', alpha=.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(value, digits=3):
    return 'n/a' if value is None else format(float(value), '.'+str(digits)+'g')


def write_report(arms, table, trajectories, path, config):
    lines = ['# Release-feasibility propagation ablation', '',
             'Both arms use identical settings; only `feasibility_mode` differs.', '',
             '- seeds: '+', '.join(map(str, config['seeds'])),
             '- episodes per seed: '+str(config['episodes']),
             '- episode horizon: '+str(config['episode_days'])+' days',
             '- evaluation sequence: '+', '.join(map(str, config['evaluation_years'])), '']
    for mode in MODES:
        failure = arms[mode].get('failure')
        if failure:
            lines += ['**'+mode+' arm did not finish:** '+failure['error_type']+': '+
                      failure['reason'], '']
    lines += ['## Held-out five-year comparison', '',
             '| metric | backward | forward_local | forward - backward |',
             '|---|---|---|---|']
    for row in table:
        lines.append('| '+row['metric']+' | '+_fmt(row['backward']['mean'])+' ± '+
                     _fmt(row['backward']['std'])+' | '+_fmt(row['forward_local']['mean'])+' ± '+
                     _fmt(row['forward_local']['std'])+' | '+_fmt(row['difference'])+' |')
    lines += ['', '## Training and feasibility totals', '',
              '| measure | backward | forward_local |', '|---|---|---|']
    for key, label in (('mean_training_return', 'mean training return'),
                       ('mean_episode_generation_TWh', 'mean episode generation (TWh)'),
                       ('total_episode_generation_TWh', 'total training generation (TWh)'),
                       ('forced_actor_days', 'forced actor-days (training)'),
                       ('empty_interval_days', 'empty release intervals'),
                       ('infeasible_episodes', 'infeasible episodes'),
                       ('max_balance_residual_m3', 'max mass-balance residual (m3)')):
        lines.append('| '+label+' | '+_fmt(arms['backward'].get(key))+' | '+
                     _fmt(arms['forward_local'].get(key))+' |')
    max_difference = trajectories.get('max_abs_difference_m')
    lines += ['', '## Water-level trajectories', '']
    lines.append('Maximum absolute difference between arms across all seed-day-reservoir '
                 'level trajectories: '+('n/a' if max_difference is None else
                                         format(max_difference, '.3e')+' m')+'.')
    lines += ['', 'Figures: `ablation_training.png`, `ablation_levels.png`, '
              '`ablation_feasibility.png`.']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--episode-days', type=int, default=1825)
    parser.add_argument('--parallel-envs', type=int, default=6)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--output', type=Path, default=ROOT/'runs/feasibility_ablation')
    parser.add_argument('--skip-training', action='store_true',
                        help='Re-analyse existing logs without retraining')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    from examples import train_cascade
    from harl.envs.cascade_reservoir.dispatch_kernel import validate_feasibility_mode

    for mode in MODES:
        validate_feasibility_mode(mode)
    logs = {mode: args.output/('episodes_'+mode+'.jsonl') for mode in MODES}
    failures = {}
    if not args.skip_training:
        for path in logs.values():
            if path.exists():
                raise SystemExit('Refusing to overwrite existing log: '+str(path))
        for mode in MODES:
            try:
                train_cascade.main([
                    '--episodes', str(args.episodes),
                    '--episode-days', str(args.episode_days),
                    '--parallel-envs', str(args.parallel_envs),
                    '--seeds', *map(str, args.seeds),
                    '--feasibility-mode', mode,
                    '--eval-interval', '5',
                    '--log-interval', '5',
                    '--no-plot',
                    '--episode-log', str(logs[mode]),
                ])
            except Exception as exc:
                # A hard physical failure is itself an ablation result. Keep the
                # log already on disk and report the failure instead of losing
                # the paired comparison.
                failures[mode] = dict(error_type=type(exc).__name__, reason=str(exc))
                print(mode+' arm failed: '+type(exc).__name__+': '+str(exc), flush=True)

    missing = [str(path) for path in logs.values() if not path.exists()]
    if missing:
        raise SystemExit('Missing ablation log; run without --skip-training first: '+
                         ', '.join(missing))
    arms = {mode: summarise_mode(_load(logs[mode])) for mode in MODES}
    for mode, failure in failures.items():
        arms[mode]['failure'] = failure
    table = compare(arms)
    trajectories = trajectory_comparison(arms)
    config = dict(seeds=args.seeds, episodes=args.episodes, episode_days=args.episode_days,
                  evaluation_years=[1998, 1999, 1970, 1997, 2006])

    _plot_curves(arms, args.output/'ablation_training.png')
    if trajectories['backward'] is not None:
        _plot_levels(arms, trajectories, args.output/'ablation_levels.png', config['evaluation_years'])
    _plot_feasibility(arms, args.output/'ablation_feasibility.png')
    write_report(arms, table, trajectories, args.output/'ablation_report.md', config)

    # Keep the machine-readable summary compact: level trajectories live in the
    # figure and the original JSONL logs.
    finals = {mode: [dict({key: value for key, value in row.items()
                           if key != 'level_trajectory'},
                          level_trajectory_summary=(
                              _level_extents(row['level_trajectory'])
                              if row.get('level_trajectory') else None))
                    for row in arms[mode]['final_evaluations']] for mode in MODES}
    serialisable = dict(config=config,
        comparison=table,
        arms={mode: {key: value for key, value in arms[mode].items()
                     if key not in ('final_evaluations',) } for mode in MODES},
        finals=finals,
        max_abs_level_difference_m=trajectories.get('max_abs_difference_m'))
    (args.output/'ablation_summary.json').write_text(
        json.dumps(serialisable, indent=2, allow_nan=False), encoding='utf-8')
    print('Ablation complete: '+str(args.output/'ablation_report.md'))
    return serialisable


if __name__ == '__main__':
    main()
