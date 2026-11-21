"""Three matched cascade figures, with separate HAPPO and MAPPO seed means."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples import plot_cascade_parallel as legacy


SCHEMA = 'cascade_comparison_v1'
COLORS = {'happo': '#c0392b', 'mappo': '#2563eb'}
# Serialized layout versions are kept here so plotting does not import Gym,
# PyTorch or the simulator in a separate Matplotlib-only interpreter.
ACTOR_OBSERVATION_LAYOUT = 'local_neighbors_next_day_limits_next7d_forecast_v6'
VOLUME_ACTOR_OBSERVATION_LAYOUT = 'local_neighbors_next_day_limits_terminal_target_next7d_forecast_future_volume_v5'
TARGET_ACTOR_OBSERVATION_LAYOUT = 'local_neighbors_next_day_limits_terminal_target_next7d_forecast_v4'
PREVIOUS_ACTOR_OBSERVATION_LAYOUT = 'local_neighbors_next_day_limits_next7d_forecast_v3'
CRITIC_STATE_LAYOUT = 'system_storage_inflow_release_limits_next7d_forecasts_v3'
VOLUME_CRITIC_STATE_LAYOUT = 'system_storage_inflow_release_limits_forecasts_targets_future_volumes_v2'
TARGET_CRITIC_STATE_LAYOUT = 'system_storage_inflow_release_limits_next7d_forecasts_terminal_targets_v1'
CASCADE_ACTOR_OBSERVATION_LAYOUT = 'cascade_local_upstream_downstream_system_v7'
CASCADE_CRITIC_STATE_LAYOUT = 'concatenated_five_actor_observations_v1'
FORECAST_INPUT = 'five_reservoir_next7d_mean_inflow_forecast_1965_2024.csv'
SHARED_FIELDS = (
    'episode_horizon_days', 'parallel_environments', 'episodes_per_seed',
    'gamma', 'action_bins', 'seeds', 'training_years', 'evaluation_source_years',
    'window_sampling', 'physical_config', 'periodic_windows', 'evaluation_interval',
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _normalized_context(context):
    """Sharing is the intended architecture difference, not a task difference."""
    result = copy.deepcopy(context)
    settings = result.get('shared_settings', {})
    if isinstance(settings, dict) and isinstance(settings.get('algo'), dict):
        settings['algo'].pop('share_param', None)
    return result


def context_from_configuration(metadata):
    """Extract matched settings without output paths or algorithm labels."""
    if 'comparison_context' in metadata:
        return _normalized_context(metadata['comparison_context'])
    config = metadata['config']
    experiment = config['experiment']
    shared = dict(
        episode_horizon_days=metadata['episode_horizon_days'],
        parallel_environments=metadata['parallel_environments'],
        episodes_per_seed=metadata['episodes_per_seed'],
        gamma=config['algo']['gamma'], action_bins=config['model']['action_bins'],
        seeds=experiment['seeds'], training_years=metadata['training_years'],
        evaluation_source_years=metadata['evaluation_source_years'],
        window_sampling=experiment.get('window_sampling', 'coverage_balanced'),
        physical_config=metadata['physical_config'],
        periodic_windows=metadata['periodic_evaluation']['windows'],
        evaluation_interval=experiment.get('eval_interval', 1),
    )
    architecture = metadata.get('learning_reward_architecture', {})
    if 'actor_temporal_weighting' in architecture:
        shared['actor_temporal_weighting'] = architecture['actor_temporal_weighting']
    for key in ('device', 'train', 'model', 'algo'):
        if key in config:
            shared[key] = config[key]
    return _normalized_context(dict(shared_settings=shared, data_hashes=metadata['data_hashes']))


def _context(payload):
    context = payload.get('comparison_context')
    if not isinstance(context, dict):
        raise ValueError('Every algorithm payload requires comparison_context')
    if (payload.get('run_id') is not None and
            (not isinstance(payload.get('provenance'), dict) or
             payload['provenance'].get('run_id') != payload['run_id'])):
        raise ValueError('Comparison payload run_id disagrees with its provenance')
    settings, hashes = context.get('shared_settings'), context.get('data_hashes')
    if not isinstance(settings, dict) or any(key not in settings for key in SHARED_FIELDS):
        raise ValueError('comparison_context.shared_settings requires: '+', '.join(SHARED_FIELDS))
    if (not isinstance(hashes, dict) or not hashes or
            any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in hashes.items())):
        raise ValueError('comparison_context requires nonempty input data_hashes')
    _canonical(context)  # Reject NaN and non-JSON values before drawing.
    for key in ('episode_horizon_days', 'parallel_environments', 'episodes_per_seed',
                'action_bins', 'evaluation_interval'):
        legacy.positive_integer(settings[key], key)
    if settings['action_bins'] < 2:
        raise ValueError('action_bins must be at least two')
    if settings['episodes_per_seed'] % settings['parallel_environments']:
        raise ValueError('episodes_per_seed must be divisible by parallel_environments')
    gamma = settings['gamma']
    if isinstance(gamma, bool) or not np.isfinite(float(gamma)) or not 0 <= float(gamma) <= 1:
        raise ValueError('Comparison gamma must be finite and in [0, 1]')
    for key in ('seeds', 'training_years', 'evaluation_source_years'):
        values = settings[key]
        if (not isinstance(values, list) or not values or
                any(isinstance(v, bool) or not isinstance(v, int) for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(key+' must contain unique integers')
    if set(settings['training_years']) & set(settings['evaluation_source_years']):
        raise ValueError('Training and held-out source years must be disjoint')
    if settings['window_sampling'] not in ('coverage_balanced', 'uniform'):
        raise ValueError('Unknown comparison training-window sampling mode')
    if not isinstance(settings['physical_config'], dict) or not settings['physical_config']:
        raise ValueError('Comparison physical_config must be a nonempty mapping')
    expected_signature = legacy.window_signature(settings['periodic_windows'])
    for window in settings['periodic_windows']:
        if not set(window['source_years']) <= set(settings['training_years']):
            raise ValueError('Periodic comparison windows must contain training source years only')
    if float(payload['gamma']) != float(gamma):
        raise ValueError('Plot gamma disagrees with comparison_context')
    if payload['episode_horizon_days'] != settings['episode_horizon_days']:
        raise ValueError('Plot episode horizon disagrees with comparison_context')
    if ('parallel_envs' in payload and
            payload['parallel_envs'] != settings['parallel_environments']):
        raise ValueError('Plot worker count disagrees with comparison_context')
    updates = settings['episodes_per_seed']//settings['parallel_environments']
    if 'updates_per_seed' in payload and payload['updates_per_seed'] != updates:
        raise ValueError('Plot update budget disagrees with comparison_context')
    seeds = payload['seeds']
    if (not isinstance(seeds, list) or not seeds or
            any(isinstance(v, bool) or not isinstance(v, int) for v in seeds) or
            len(set(seeds)) != len(seeds) or not set(seeds) <= set(settings['seeds'])):
        raise ValueError('Plot seeds must be a nonempty subset of configured comparison seeds')
    if payload.get('configured_seeds', settings['seeds']) != settings['seeds']:
        raise ValueError('Configured plot seeds disagree with comparison_context')
    return _normalized_context(context), expected_signature


def _observation_dimensions(run):
    """Validate explicit layouts, retaining the dimensions of unversioned logs."""
    architecture = run.get('actor_architecture') or {}
    daily_limits = run['comparison_context']['shared_settings']['physical_config'].get(
        'use_daily_level_limits', True)
    has_forecast = FORECAST_INPUT in run['comparison_context']['data_hashes']
    actor_layout = architecture.get('actor_observation_layout')
    critic_layout = architecture.get('critic_state_layout')
    state_size = 38 if daily_limits else 18
    raw_dim = 15 if daily_limits else 11
    if actor_layout == CASCADE_ACTOR_OBSERVATION_LAYOUT:
        if not has_forecast:
            raise ValueError('The cascade actor observation layout requires the inflow forecast input hash')
        if critic_layout != CASCADE_CRITIC_STATE_LAYOUT:
            raise ValueError('The cascade actor observation layout requires the concatenated critic state layout')
        if architecture.get('critic_state_size') != 110:
            raise ValueError('Critic state size does not match its observation layout')
        return 22, 110
    if actor_layout in (ACTOR_OBSERVATION_LAYOUT, TARGET_ACTOR_OBSERVATION_LAYOUT, VOLUME_ACTOR_OBSERVATION_LAYOUT):
        if not has_forecast:
            raise ValueError('The current actor observation layout requires the inflow forecast input hash')
        extra, expected_critic = {ACTOR_OBSERVATION_LAYOUT: (1, CRITIC_STATE_LAYOUT),
            TARGET_ACTOR_OBSERVATION_LAYOUT: (2, TARGET_CRITIC_STATE_LAYOUT),
            VOLUME_ACTOR_OBSERVATION_LAYOUT: (3, VOLUME_CRITIC_STATE_LAYOUT)}[actor_layout]
        raw_dim += extra
        state_size += 5*extra
        if critic_layout != expected_critic:
            raise ValueError('Current actor observation layout requires the matching critic state layout')
        if architecture.get('critic_state_size') != state_size:
            raise ValueError('Critic state size does not match its observation layout')
    else:
        if actor_layout not in (None, PREVIOUS_ACTOR_OBSERVATION_LAYOUT):
            raise ValueError('Unsupported actor observation layout: '+str(actor_layout))
        if actor_layout == PREVIOUS_ACTOR_OBSERVATION_LAYOUT and not has_forecast:
            raise ValueError('The forecast actor observation layout requires the inflow forecast input hash')
        raw_dim += int(has_forecast)
        if critic_layout is not None:
            raise ValueError('Unsupported critic state layout for historical actor observations')
        if 'critic_state_size' in architecture and architecture['critic_state_size'] != state_size:
            raise ValueError('Historical critic state size does not match the cascade observations')
    return raw_dim, state_size


def _validate_architecture(algorithm, run):
    """Accept legacy independent actors or explicit ID-conditioned shared MAPPO."""
    settings = run['comparison_context']['shared_settings']
    requested = settings.get('algo', {}).get('share_param')
    if requested is not None and not isinstance(requested, bool):
        raise ValueError('Actor share_param must be a boolean')
    if algorithm == 'happo' and requested:
        raise ValueError('HAPPO comparison requires independent actor parameters')
    architecture = run.get('actor_architecture')
    objective = run.get('learning_objective')
    critic_dim = 6  # Historical runs used individual mixed values plus legacy.
    if objective is not None:
        if not isinstance(objective, dict):
            raise ValueError('learning_objective must be a mapping')
        counterfactual = (algorithm == 'happo' and
            settings.get('algo', {}).get('happo_credit_assignment','team_gae') == 'counterfactual')
        team = algorithm == 'happo' and objective.get('advantage_source') in ('shared_team','counterfactual_shared_team')
        mixed_happo = (algorithm == 'happo' and
                       objective.get('critic_reward_layout') == 'mixed_by_reservoir')
        objective_layout = ('shared_team' if team else
            'mixed_by_reservoir' if objective.get('critic_reward_layout') == 'mixed_by_reservoir' else
            'mixed_by_reservoir_then_legacy_team')
        expected_objective = dict(
            advantage_source='counterfactual_shared_team' if counterfactual else 'shared_team' if team else 'individual_mixed',
            reward_formula=('0.5 * sum(local rewards) + 0.5 * global generation reward'
                            if team else '0.5 * own local reward + 0.5 * global generation reward'),
            critic_reward_layout=objective_layout,
            actor_value_channels=[0]*5 if team else list(range(5)),
            normalization=('common_controllable_rms_without_centering' if counterfactual else
                           'once_over_common_rollout_rows' if team
                           else 'per_reservoir_over_common_rollout_rows'
                           if mixed_happo or settings.get('algo', {}).get('normalize_shared_advantages', False)
                           else 'per_reservoir_over_controllable_rows'))
        if objective != expected_objective:
            raise ValueError('Learning objective does not match the algorithm reward/value layout')
        critic_dim = 1 if team else 5 if objective.get('critic_reward_layout') == 'mixed_by_reservoir' else 6
    if architecture is None:
        if requested:
            raise ValueError('Shared MAPPO requires explicit actor_architecture metadata')
        return  # Older labeled MAPPO and HAPPO logs used five independent actors.
    if not isinstance(architecture, dict):
        raise ValueError('actor_architecture must be a mapping')
    shared = architecture.get('parameter_sharing')
    if not isinstance(shared, bool) or (algorithm == 'happo' and shared):
        raise ValueError('Only MAPPO may use an ID-conditioned shared actor')
    if requested is not None and requested != shared:
        raise ValueError('Actor architecture disagrees with share_param')
    raw_dim, _ = _observation_dimensions(run)
    expected = dict(network_count=1 if shared else 5, optimizer_count=1 if shared else 5,
        raw_observation_dim=raw_dim, network_observation_dim=raw_dim+5 if shared else raw_dim,
        identity_features='one_hot_reservoir_id' if shared else 'implicit_in_separate_actor',
        critic_output_dim=critic_dim)
    if any(architecture.get(key) != value for key, value in expected.items()):
        raise ValueError('Actor architecture dimensions/counts do not match the cascade observations')


def validate_comparison(payload):
    """Validate identity and each legacy history; return a normalized outer payload."""
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('Expected comparison schema '+SCHEMA)
    runs = payload.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('Comparison requires at least one algorithm run')
    normalized, names, reference, observation_reference = [], set(), None, None
    run_ids = set()
    for entry in runs:
        if not isinstance(entry, dict) or not isinstance(entry.get('algorithm'), str):
            raise ValueError('Every run requires an algorithm name and legacy payload')
        algorithm = entry['algorithm'].lower()
        if algorithm not in COLORS or algorithm in names:
            raise ValueError('Algorithms must be unique HAPPO/MAPPO entries')
        names.add(algorithm)
        run = entry.get('payload')
        if not isinstance(run, dict):
            raise ValueError('Every algorithm run requires a legacy plotting payload')
        if 'algorithm' in run and str(run['algorithm']).lower() != algorithm:
            raise ValueError('Algorithm label disagrees with its plotting payload')
        context, signature = _context(run)
        observation_dimensions = _observation_dimensions(run)
        if observation_reference is not None and observation_dimensions != observation_reference:
            raise ValueError('Comparison actor/critic observation layouts differ between algorithms')
        observation_reference = observation_dimensions
        _validate_architecture(algorithm, run)
        identity = _canonical(context)
        if reference is not None and identity != reference:
            raise ValueError('Comparison settings or input data_hashes differ between algorithms')
        if run.get('source_run_id') is not None and run['source_run_id'] in run_ids:
            raise ValueError('HAPPO and MAPPO results must have different run identities')
        if run.get('source_run_id') is not None:
            run_ids.add(run['source_run_id'])
        reference = identity
        legacy.available_histories(run, 'histories')
        for rows in run.get('histories', []):
            steps = [legacy.positive_integer(r['training_steps'], 'training_steps') for r in rows]
            if any(b <= a for a, b in zip(steps, steps[1:])):
                raise ValueError('Training steps must increase without duplicates')
            if any(not np.isfinite(float(r[key])) for r in rows
                   for key in ('episode_reward', 'episode_energy_TWh')):
                raise ValueError('Training curve metrics must be finite')
            for row in rows:
                legacy.learning_return_value(run, row, 'training')
        _, histories, _ = legacy.periodic_histories(run)
        for rows in histories:
            for row in rows:
                legacy.validate_periodic_row(row, signature)
                if 'gamma' in row and float(row['gamma']) != float(run['gamma']):
                    raise ValueError('Periodic gamma differs from plot gamma')
        if not legacy.violation_metrics_available(run):
            raise ValueError('Three comparison figures require actual periodic violation counts')
        normalized.append(dict(algorithm=algorithm, payload=run))
    return dict(payload, runs=normalized)


def load_comparison(happo_log=None, mappo_log=None):
    """Load separate logs: legacy integer seed identifiers remain isolated."""
    paths = [Path(path).resolve() for path in (happo_log, mappo_log) if path is not None]
    if len(set(paths)) != len(paths):
        raise ValueError('The same log cannot represent two different algorithms')
    runs = []
    for algorithm, path in (('happo', happo_log), ('mappo', mappo_log)):
        if path is None:
            continue
        run = legacy.load_episode_log(path)
        metadata = None
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('record_type') == 'run_configuration':
                    metadata = row
                    break
        if metadata is None:
            raise ValueError('A run_configuration is required for comparison')
        recorded = metadata.get('algorithm', metadata.get('algorithm_name'))
        if recorded is None and algorithm == 'mappo':
            raise ValueError('A MAPPO log requires explicit algorithm metadata; unlabeled legacy logs are HAPPO')
        if recorded is not None and str(recorded).lower() != algorithm:
            raise ValueError('The supplied '+algorithm+' log records a different algorithm')
        run['comparison_context'] = context_from_configuration(metadata)
        if 'learning_objective' in metadata:
            run['learning_objective'] = metadata['learning_objective']
        if 'actor_architecture' in metadata:
            run['actor_architecture'] = copy.deepcopy(metadata['actor_architecture'])
        if 'run_id' in metadata:
            run['source_run_id'] = metadata['run_id']
        if 'provenance' in metadata:
            run['source_provenance'] = copy.deepcopy(metadata['provenance'])
        runs.append(dict(algorithm=algorithm, payload=run))
    return validate_comparison(dict(schema=SCHEMA, runs=runs))


def _label(algorithm, count, run, return_curve=False):
    total = len(run['comparison_context']['shared_settings']['seeds'])
    method = algorithm.upper()
    if run['comparison_context']['shared_settings'].get('algo',{}).get('happo_credit_assignment') == 'counterfactual':
        method += ' counterfactual'
    label = method+(' mean' if count == total else
                   ' available-seed mean (n={})'.format(count))
    if return_curve:
        label += ' ('+legacy.learning_return_label(run).lower()+')'
    return label


def _line_and_band(ax, grid, aligned, algorithm, run, return_curve=False):
    valid = np.isfinite(aligned).all(axis=0)
    mean = np.full(len(grid), np.nan)
    mean[valid] = aligned[:, valid].mean(axis=0)
    color = COLORS[algorithm]
    ax.plot(grid, mean, color=color, lw=2,
            linestyle='-' if algorithm == 'happo' else '--',
            label=_label(algorithm, len(aligned), run, return_curve))
    if len(aligned) > 1:
        sd = np.full(len(grid), np.nan)
        sd[valid] = aligned[:, valid].std(axis=0, ddof=1)
        ax.fill_between(grid, mean-sd, mean+sd, color=color, alpha=.15,
                        label=algorithm.upper()+' +/-1 SD')


def _training(ax, algorithm, run, metric):
    _, histories = legacy.available_histories(run, 'histories')
    if not histories:
        return
    common = set(r['training_steps'] for r in histories[0])
    for rows in histories[1:]:
        common.intersection_update(r['training_steps'] for r in rows)
    grid = np.array(sorted(common), dtype=float)
    aligned = np.array([[{r['training_steps']: (legacy.learning_return_value(run, r, 'training')
                                             if metric == 'learning_return' else r[metric]) for r in rows}[step]
                         for step in grid] for rows in histories], dtype=float)
    _line_and_band(ax, grid, aligned, algorithm, run, metric == 'learning_return')


def _periodic(ax, algorithm, run, metric, percent=False):
    seeds, histories, grids = legacy.periodic_histories(run)
    grid = np.unique(np.concatenate(grids))
    aligned = np.full((len(seeds), len(grid)), np.nan)
    for i, (rounds, rows) in enumerate(zip(grids, histories)):
        aligned[i, np.searchsorted(grid, rounds)] = [
            legacy.learning_return_value(run, row, 'evaluation') if metric == 'learning_return' else
            np.nan if row[metric] is None else float(row[metric])*(100. if percent else 1.)
            for row in rows]
    _line_and_band(ax, grid, aligned, algorithm, run, metric == 'learning_return')
    missing = ~np.isfinite(aligned).all(axis=0)
    if missing.any():
        y = .025 if algorithm == 'happo' else .055
        ax.scatter(grid[missing], np.full(int(missing.sum()), y),
                   transform=ax.get_xaxis_transform(), marker='x', color=COLORS[algorithm],
                   label=algorithm.upper()+' unavailable evaluation', clip_on=False)


def make_figures(payload):
    """Create training, evaluator-return and ecological-flow violation figures."""
    payload = validate_comparison(payload)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    training, training_axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    evaluator, evaluator_ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    violations, violation_ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    violation_axes = (violation_ax,)
    for entry in payload['runs']:
        algorithm, run = entry['algorithm'], entry['payload']
        for ax, metric in zip(training_axes, ('learning_return', 'episode_energy_TWh')):
            _training(ax, algorithm, run, metric)
        _periodic(evaluator_ax, algorithm, run, 'learning_return')
        for ax, term in zip(violation_axes, legacy.VIOLATION_TERMS):
            _periodic(ax, algorithm, run, term+'_violation_rate', percent=True)
    shared = payload['runs'][0]['payload']['comparison_context']['shared_settings']
    kinds = {legacy.learning_return_kind(entry['payload']) for entry in payload['runs']}
    return_label = (legacy.learning_return_label(payload['runs'][0]['payload']) if len(kinds) == 1
                    else 'Learning-objective return (see legend)' if 'legacy_team' not in kinds
                    else 'Return (see objective labels in legend)')
    for ax, label in zip(training_axes, (
            'Mean collected '+return_label.lower()+' (gamma={:g})'.format(float(shared['gamma'])),
            'Mean collected episode generation (TWh)')):
        ax.set(xlabel='Collected daily environment transitions', ylabel=label,
               title='{}-day stochastic training episodes (in-sample)'.format(shared['episode_horizon_days']))
    evaluator_ax.set(ylabel=return_label+'\n(mean of three fixed training windows)',
                     title=('Legacy Return' if kinds == {'legacy_team'} else 'Learning Return')+
                         ' vs. Evaluation Round')
    for ax, title in zip(violation_axes, ('Ecological-flow violation rate',)):
        ax.set(ylabel='Violation rate (%)', title=title)
    for ax in (evaluator_ax, *violation_axes):
        ax.set(xlabel='Evaluation round', xlim=(0, shared['episodes_per_seed']//shared['parallel_environments']))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in (*training_axes, evaluator_ax, *violation_axes):
        ax.grid(alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    violations.suptitle('Fixed-scenario evaluation: violation rates')
    objectives = {entry['algorithm']: entry['payload'].get('learning_objective', {}).get('advantage_source', 'individual_mixed')
                  for entry in payload['runs']}
    if len(set(objectives.values())) > 1:
        note = 'Different learning objectives: HAPPO shared team; MAPPO individual mixed'
        training.suptitle(note, fontsize=10)
        evaluator.suptitle(note, fontsize=10)
        violations.suptitle('Fixed-scenario violation rates\n'+note, fontsize=10)
    return dict(training=training, evaluator=evaluator, violations=violations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--happo-log', type=Path)
    parser.add_argument('--mappo-log', type=Path)
    parser.add_argument('--output', type=Path, help='Training PNG; also derives the other two paths')
    parser.add_argument('--evaluator-output', type=Path)
    parser.add_argument('--violations-output', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.no_show and all(p is None for p in (args.output, args.evaluator_output, args.violations_output)):
        parser.error('--no-show requires an output path')
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg', force=True)
    try:
        payload = (load_comparison(args.happo_log, args.mappo_log)
                   if args.happo_log is not None or args.mappo_log is not None else json.load(sys.stdin))
        outputs = dict(training=args.output,
                       evaluator=args.evaluator_output or (legacy.evaluator_output_path(args.output) if args.output else None),
                       violations=args.violations_output or (legacy.violation_output_path(args.output) if args.output else None))
        targets = [p.resolve() for p in outputs.values() if p is not None]
        sources = {p.resolve() for p in (args.happo_log, args.mappo_log) if p is not None}
        if len(set(targets)) != len(targets) or sources.intersection(targets):
            raise ValueError('Figure outputs must be distinct and cannot overwrite input logs')
        for output in outputs.values():
            if output is not None and output.exists():
                raise ValueError('Figure output already exists: '+str(output))
        figures = make_figures(payload)
        for key, figure in figures.items():
            if outputs[key] is not None:
                outputs[key].parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(str(outputs[key]), dpi=180)
                print('Saved '+key+' comparison to '+str(outputs[key].resolve()))
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    import matplotlib.pyplot as plt
    if args.no_show:
        for figure in figures.values():
            plt.close(figure)
    else:
        plt.show()


if __name__ == '__main__':
    main()
