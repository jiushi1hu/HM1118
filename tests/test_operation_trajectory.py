"""Calendar-complete final-policy plot data is observational and never invented."""
from datetime import date, timedelta
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir import operation_trajectory as operation
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade


class CalendarData:
    def __init__(self, years):
        self.training_years = list(years)
        first, stop = date(years[0], 1, 1), date(years[-1]+1, 1, 1)
        self.dates = tuple(first+timedelta(days=i) for i in range((stop-first).days))

    def indices(self, first, last):
        positions = [i for i, day in enumerate(self.dates) if first <= day.year <= last]
        return positions[0], positions[-1]+1

    def initial_storage(self, year):
        return np.arange(5, dtype=float)+100.

    def levels(self, storage):
        return np.asarray(storage)*.5


class OperationTrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.years = list(range(2016, 2021))
        self.data = CalendarData(self.years)
        self.actors = []
        for _ in range(5):
            module = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
            module.train()
            module[1].eval()  # Preserve individual submodule modes as well.
            self.actors.append(SimpleNamespace(actor=module))

    def capture(self, years=None):
        return operation.capture_operation_trajectory(self.actors, self.data, {},
            seed=2, episode=300, policy_update=50, training_steps=547500,
            scenario=dict(window_id='median_2016',
                          source_years=self.years if years is None else years), gamma=1.)

    def emit(self, count, observer):
        for i, day in enumerate(self.data.dates[:count]):
            observer(dict(date=day.isoformat(), levels_start=np.arange(5)+i,
                levels_end=np.arange(5)+i+.5, release=np.arange(5)+200.,
                inflow=np.arange(5)+300.))

    def test_two_leap_days_and_all_five_actual_dec31_end_levels_are_retained(self):
        def evaluate(*args, **options):
            self.assertEqual(options['source_years'], self.years)
            self.assertEqual(options['episode_days'], 1827)
            self.assertTrue(options['deterministic'])
            self.emit(options['episode_days'], options['transition_observer'])
        with mock.patch.object(operation, 'evaluate_policy', side_effect=evaluate):
            result = self.capture()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual((result['steps'], result['expected_steps']), (1827, 1827))
        stamps = result['series']['date']
        self.assertEqual((stamps[0], stamps[-1]), ('2016-01-01', '2020-12-31'))
        self.assertIn('2016-02-29', stamps)
        self.assertIn('2020-02-29', stamps)
        endpoints = [i for i, stamp in enumerate(stamps) if stamp.endswith('-12-31')]
        self.assertEqual(len(endpoints), 5)
        for index in endpoints:
            self.assertEqual(result['series']['levels_end'][index][0], index+.5)
        self.assertEqual(result['policy_update'], 50)

    def test_rng_draws_and_all_module_modes_are_restored_without_parameter_changes(self):
        def evaluate(*args, **options):
            random.random()
            np.random.random(4)
            torch.rand(4)
            for actor in self.actors:
                actor.actor.eval()
            self.emit(options['episode_days'], options['transition_observer'])
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        modes = [[child.training for child in actor.actor.modules()] for actor in self.actors]
        weights = [actor.actor[0].weight.detach().clone() for actor in self.actors]
        with mock.patch.object(operation, 'evaluate_policy', side_effect=evaluate):
            self.capture()
        self.assertEqual(random.getstate(), python_state)
        current = np.random.get_state()
        self.assertEqual(current[0], numpy_state[0])
        np.testing.assert_array_equal(current[1], numpy_state[1])
        self.assertEqual(current[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(modes, [[child.training for child in actor.actor.modules()]
                                 for actor in self.actors])
        for weight, actor in zip(weights, self.actors):
            self.assertTrue(torch.equal(weight, actor.actor[0].weight))

    def test_physical_failure_retains_only_executed_days_with_no_endpoint_padding(self):
        def evaluate(*args, **options):
            self.emit(3, options['transition_observer'])
            raise InfeasibleCascade('Synthetic fourth-day failure')
        with mock.patch.object(operation, 'evaluate_policy', side_effect=evaluate):
            result = self.capture()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual((result['steps'], result['expected_steps']), (3, 1827))
        self.assertEqual(result['series']['date'][-1], '2016-01-03')
        self.assertEqual(result['failure']['date'], '2016-01-04')
        self.assertEqual(result['failure']['completed_steps'], 3)

    def test_short_nonconsecutive_split_or_incomplete_calendars_are_unavailable(self):
        for years in ([2016], [2016, 2017, 2018, 2019, 2021]):
            with self.subTest(years=years), mock.patch.object(operation, 'evaluate_policy') as evaluate:
                result = self.capture(list(years))
                self.assertEqual(result['status'], 'unavailable')
                self.assertEqual(result['steps'], 0)
                self.assertEqual(result['series']['date'], [])
                evaluate.assert_not_called()
        self.data.training_years = self.years[:-1]
        with mock.patch.object(operation, 'evaluate_policy') as evaluate:
            self.assertEqual(self.capture()['status'], 'unavailable')
            evaluate.assert_not_called()
        self.data.training_years = self.years
        self.data.dates = self.data.dates[:-1]
        with mock.patch.object(operation, 'evaluate_policy') as evaluate:
            self.assertEqual(self.capture()['status'], 'unavailable')
            evaluate.assert_not_called()

    def test_programming_errors_propagate_instead_of_being_labeled_physical_failures(self):
        with mock.patch.object(operation, 'evaluate_policy', side_effect=RuntimeError('Bad configuration')):
            with self.assertRaisesRegex(RuntimeError, 'Bad configuration'):
                self.capture()


if __name__ == '__main__':
    unittest.main()
