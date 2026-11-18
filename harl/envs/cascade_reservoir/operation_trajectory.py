"""Compact, observational rollout covering five complete source calendar years.

The fixed 1,825-day evaluator retains leap days and can stop before the fifth
December 31. This separate final-policy rollout supplies the calendar-complete
operation plot without changing collection horizons or evaluator statistics.
"""
from datetime import date, timedelta

import numpy as np

from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import _preserve_training_state
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade


SERIES_FIELDS = ('levels_start', 'levels_end', 'release', 'inflow')


def _calendar_dates(data, years):
    """Require the selected scenario's five complete, consecutive training years."""
    if len(years) != 5 or years != list(range(years[0], years[0]+5)):
        raise ValueError('The selected scenario must contain five consecutive source years')
    if not set(years).issubset(data.training_years):
        raise ValueError('All operation-plot source years must belong to the training split')
    days = []
    for year in years:
        first, stop = data.indices(year, year)
        source = tuple(data.dates[first:stop])
        start, end = date(year, 1, 1), date(year, 12, 31)
        expected = tuple(start+timedelta(days=i) for i in range((end-start).days+1))
        if source != expected:
            raise ValueError('The operation plot requires complete Jan-1 to Dec-31 source years')
        days.extend(source)
    return days


def capture_operation_trajectory(actors, data, physical_config, *, seed, episode,
                                 policy_update, training_steps, scenario, gamma):
    """Evaluate the frozen policy once; preserve RNGs and record physical failures.

    Short integration scenarios are explicitly unavailable rather than expanded
    into unrelated years. No samples, metrics or storage endpoints from this
    auxiliary rollout enter the training or fixed-window evaluator histories.
    """
    years = list(scenario['source_years'])
    series = dict(date=[], **{name: [] for name in SERIES_FIELDS})
    result = dict(schema_version=1, record_type='five_year_operation_trajectory',
        seed=int(seed), source_years=years, policy_update=int(policy_update),
        scenario_id=scenario['window_id']+'_calendar5y',
        source_scenario_id=scenario['window_id'], deterministic=True,
        status='unavailable', steps=0, expected_steps=0, series=series,
        calendar_policy='five complete source calendar years; leap days retained',
        level_timestamp='levels_end is the end-of-day level after the dated daily action',
        physical={})
    try:
        days = _calendar_dates(data, years)
    except ValueError as exc:
        result['reason'] = str(exc)
        return result
    result.update(expected_steps=len(days), start_date=days[0].isoformat(),
                  end_date=days[-1].isoformat())
    initial = np.asarray(data.initial_storage(years[0]), dtype=float)
    result['physical']['initial_level_m'] = np.asarray(data.levels(initial), dtype=float).tolist()

    def capture(transition):
        index = len(series['date'])
        stamp = str(transition['date'])
        if index >= len(days) or stamp != days[index].isoformat():
            raise ValueError('Operation trajectory does not match the selected source calendar')
        values = {}
        for name in SERIES_FIELDS:
            value = np.asarray(transition[name], dtype=float)
            if value.shape != (5,) or not np.isfinite(value).all():
                raise ValueError('Operation trajectory '+name+' must be a finite five-vector')
            values[name] = value.tolist()
        series['date'].append(stamp)
        for name, value in values.items():
            series[name].append(value)

    with _preserve_training_state(actors):
        try:
            evaluate_policy(actors, data, physical_config, seed=int(seed),
                episode=int(episode), training_steps=int(training_steps), gamma=gamma,
                source_years=years, episode_days=len(days), deterministic=True,
                transition_observer=capture)
        except InfeasibleCascade as exc:
            progress = getattr(exc, 'evaluation_progress', {})
            result.update(status='failed', failure=dict(error_type=type(exc).__name__,
                reason=str(exc), date=progress.get('date', days[len(series['date'])].isoformat()
                    if len(series['date']) < len(days) else days[-1].isoformat()),
                completed_steps=len(series['date'])))
        else:
            if len(series['date']) != len(days):
                raise ValueError('Completed operation trajectory must contain every source-calendar day')
            result['status'] = 'ok'
    result['steps'] = len(series['date'])
    return result
