"""Hard daily water-level ramp checks before release dispatch."""
import copy
import csv
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.train_cascade import load_configuration
from harl.envs.cascade_reservoir.batched_physics import batched_transition
from harl.envs.cascade_reservoir.daily_change_diagnostics import (
    DailyChangeRecorder, write_daily_change_csv)
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.level_ramp import (
    CONFLICT_FIELDS, RampConflictWriter, daily_level_ramp_limits)
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade


class DailyLevelRampTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.physical = load_configuration()
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical,
                               cls.config['experiment'])
        cls.limits = daily_level_ramp_limits(cls.physical)

    def environment(self, days=1, bins=11):
        return episode_environment(self.data, self.physical, [1965], days=days,
                                   action_bins=bins)

    def test_exact_constant_and_tgr_seasonal_limits_at_month_boundaries(self):
        cases = (
            (date(2020, 1, 1), [1., 2., 3., 4., .6], [1., 2., 3., 4., .6]),
            (date(2020, 5, 31), [1., 2., 3., 4., .6], [1., 2., 3., 4., .6]),
            (date(2020, 6, 1), [1., 2., 3., 4., 2.], [1., 2., 3., 4., 2.]),
            (date(2020, 8, 1), [1., 2., 3., 4., 3.], [1., 2., 3., 4., 2.]),
            (date(2020, 9, 30), [1., 2., 3., 4., 3.], [1., 2., 3., 4., 2.]),
            (date(2020, 10, 1), [1., 2., 3., 4., 3.], [1., 2., 3., 4., .6]),
            (date(2020, 10, 31), [1., 2., 3., 4., 3.], [1., 2., 3., 4., .6]),
            (date(2020, 11, 1), [1., 2., 3., 4., .6], [1., 2., 3., 4., .6]),
        )
        for day, expected_rise, expected_fall in cases:
            with self.subTest(day=day):
                rise, fall, season = self.limits.limits_for_date(day)
                np.testing.assert_array_equal(rise, expected_rise)
                np.testing.assert_array_equal(fall, expected_fall)
                if 8 <= day.month <= 10:
                    self.assertTrue(all(name.startswith('refill_') for name in season[4:]))

    def test_invalid_or_incomplete_ramp_configuration_is_rejected(self):
        physical = copy.deepcopy(self.physical)
        del physical['daily_level_ramp_limits']['constant']['WDD']['fall_m_per_day']
        with self.assertRaisesRegex(ValueError, 'fall_m_per_day'):
            daily_level_ramp_limits(physical)
        physical = copy.deepcopy(self.physical)
        physical['daily_level_ramp_limits']['seasonal']['THR']['flood_months'] = [7, 8, 9]
        with self.assertRaisesRegex(ValueError, 'partition'):
            daily_level_ramp_limits(physical)

    def test_effective_bounds_are_intersected_before_dispatch(self):
        env = self.environment(days=1)
        lower, upper = env._next_state_storage_bounds(env.index)
        level_lower, level_upper = env._next_state_level_bounds(env.index)
        effective_lower, effective_upper, metadata = self.limits.effective_storage_bounds(
            self.data, env.storage, self.data.dates[env.index], lower, upper,
            level_lower, level_upper)
        expected_level_lower = np.maximum(level_lower, metadata['current_level_m']-
                                         metadata['ramp_fall_limit_m_per_day'])
        expected_level_upper = np.minimum(level_upper, metadata['current_level_m']+
                                         metadata['ramp_rise_limit_m_per_day'])
        np.testing.assert_allclose(metadata['effective_level_lower_m'],
                                   expected_level_lower, rtol=0., atol=1e-12)
        np.testing.assert_allclose(metadata['effective_level_upper_m'],
                                   expected_level_upper, rtol=0., atol=1e-12)
        for index, name in enumerate(IDS):
            anchors = self.data.anchors[name]
            self.assertAlmostEqual(
                np.interp(expected_level_lower[index], anchors[:, 0], anchors[:, 1]),
                effective_lower[index], places=6)
            self.assertAlmostEqual(
                np.interp(expected_level_upper[index], anchors[:, 0], anchors[:, 1]),
                effective_upper[index], places=6)

    def test_requested_actions_are_mapped_inside_hard_ramp_limits_without_clipping(self):
        for commands in (np.zeros(5, dtype=int), np.full(5, 10, dtype=int)):
            with self.subTest(commands=commands.tolist()):
                env = self.environment(days=1)
                env.step(commands)
                row = env.last_transition
                self.assertTrue(row['level_ramp_within_limits'].all())
                np.testing.assert_array_equal(row['storage_lower_m3'],
                                              row['ramp_storage_lower_m3'])
                np.testing.assert_array_equal(row['storage_upper_m3'],
                                              row['ramp_storage_upper_m3'])
                self.assertTrue(np.all(row['level_change_m'] <=
                                       row['ramp_rise_limit_m_per_day']+1e-8))
                self.assertTrue(np.all(row['level_change_m'] >=
                                       -row['ramp_fall_limit_m_per_day']-1e-8))
                np.testing.assert_allclose(row['storage_end'],
                    row['storage_start']+(row['inflow']-row['release'])*env.physics.dt,
                    rtol=0., atol=1e-4)
                np.testing.assert_array_equal(env.storage, row['storage_end'])
                np.testing.assert_allclose(row['levels_end'],
                    self.data.levels(row['storage_end']), rtol=0., atol=1e-12)

    def test_ramp_conflict_raises_structured_error_and_leaves_state_unchanged(self):
        env = self.environment(days=1)
        limits = copy.copy(env.level_limits)
        level_lower = limits.level_lower.copy()
        storage_lower = limits.storage_lower.copy()
        current = env.physics._forebay_levels(env.storage)
        target = current[0]+env.level_ramp_limits.month_rise[env.data.dates[env.index].month, 0]+1.
        level_lower[env.index, 0] = target
        storage_lower[env.index, 0] = np.interp(
            target, self.data.anchors[IDS[0]][:, 0], self.data.anchors[IDS[0]][:, 1])
        limits.level_lower, limits.storage_lower = level_lower, storage_lower
        env.level_limits = limits
        storage, index, previous = env.storage.copy(), env.index, env.last_transition
        with self.assertRaises(InfeasibleCascade) as raised:
            env.step(np.zeros(5, dtype=int))
        self.assertTrue(raised.exception.conflicts)
        conflict = raised.exception.conflicts[0]
        self.assertEqual(conflict['reservoir'], 'WDD')
        self.assertEqual(conflict['reason'], 'ramp_rise_below_required_next_level')
        self.assertEqual(conflict['date'], env.data.dates[index].isoformat())
        np.testing.assert_array_equal(env.storage, storage)
        self.assertEqual(env.index, index)
        self.assertIs(env.last_transition, previous)

    def test_scalar_and_batched_dispatch_match_with_effective_ramp_bounds(self):
        env = self.environment(days=1)
        env.step(np.arange(5, dtype=int))
        row = env.last_transition
        batch = batched_transition(env.physics, row['storage_start'][None],
            row['external'][None], row['requested_normalized_actions'][None],
            row['storage_lower_m3'][None], row['storage_upper_m3'][None])
        np.testing.assert_allclose(batch['release'][0], row['release'], rtol=0., atol=1e-12)
        np.testing.assert_allclose(batch['storage_end'][0], row['storage_end'], rtol=0., atol=1e-6)

    def test_daily_csv_and_recorder_retain_limits_and_first_difference_rule(self):
        recorder = DailyChangeRecorder(7)
        env = self.environment(days=2)
        for commands in (np.zeros(5, dtype=int), np.full(5, 10, dtype=int)):
            env.step(commands)
            recorder.capture(env.last_transition)
        payload = recorder.finish(expected_steps=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'daily.csv'
            write_daily_change_csv(path, payload)
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row['delta_level_m'] == '' for row in rows[:5]))
        self.assertFalse(any(row['within_ramp_limit'] == '' for row in rows[5:]))
        self.assertTrue(all(row['within_ramp_limit'] == '1' for row in rows[5:]))

    def test_conflict_csv_exists_with_header_and_records_structured_evidence(self):
        conflict = dict(date='2020-08-01', reservoir='THR', reservoir_index=4,
            reason='ramp_rise_below_required_next_level', current_level_m=160.,
            rise_limit_m_per_day=3., fall_limit_m_per_day=2.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'happo_seed2_ramp_conflicts.csv'
            with RampConflictWriter(path, 2) as writer:
                writer.write([conflict], 'training', window_start_year=2020,
                             episode_step=214)
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(tuple(rows[0]), CONFLICT_FIELDS)
        self.assertEqual(rows[0]['seed'], '2')
        self.assertEqual(rows[0]['phase'], 'training')
        self.assertEqual(rows[0]['reservoir'], 'THR')
        self.assertEqual(rows[0]['reason'], 'ramp_rise_below_required_next_level')


if __name__ == '__main__':
    unittest.main(verbosity=2)
