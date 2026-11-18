"""Free episode-end storage retains daily physical constraints and actor control."""
import copy
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.batched_physics import batched_transition
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.dispatch_kernel import dispatch_release
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics


class FreeEndStorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load(
            (ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical)

    def environment(self, days=1, limits=True, data=None):
        return episode_environment(self.data if data is None else data,
            dict(self.physical, use_daily_level_limits=limits), [1965],
            days=days, action_bins=11)

    def test_final_day_commands_control_flow_and_allow_storage_above_or_below_initial(self):
        for limits in (False, True):
            with self.subTest(daily_limits=limits):
                retaining, releasing = self.environment(limits=limits), self.environment(limits=limits)
                initial = retaining.storage.copy()
                left, right = retaining.step(np.zeros(5)), releasing.step(np.full(5, 10))
                self.assertTrue(left[3].all() and right[3].all())
                self.assertEqual(retaining.last_transition['controls'][0], 1.)
                self.assertEqual(releasing.last_transition['controls'][0], 1.)
                self.assertLess(retaining.last_transition['release'][0], releasing.last_transition['release'][0])
                self.assertGreater(retaining.storage[0], initial[0])
                self.assertLess(releasing.storage[0], initial[0])
                np.testing.assert_array_equal(retaining.storage, retaining.last_transition['storage_end'])
                np.testing.assert_array_equal(releasing.storage, releasing.last_transition['storage_end'])
                for env in (retaining, releasing):
                    self.assertFalse(any(name.startswith('terminal') for name in env.last_transition))
                    self.assertFalse(any(name.startswith('terminal') for name in env.observation_metadata()))

    def test_identical_physical_context_dispatches_identically_at_different_episode_horizons(self):
        for limits in (False, True):
            with self.subTest(daily_limits=limits):
                short, long = self.environment(limits=limits), self.environment(days=30, limits=limits)
                commands = np.arange(4, 9)
                short.step(commands)
                long.step(commands)
                for name in ('release', 'inflow', 'storage_end', 'controls', 'power', 'energy', 'low', 'high'):
                    np.testing.assert_array_equal(short.last_transition[name], long.last_transition[name])

    def test_daily_envelopes_capacities_and_independent_prefix_conservation_are_preserved(self):
        env = self.environment(days=20)
        for step in range(20):
            env.step((3*np.arange(5)+7*step) % 11)
            row = env.last_transition
            dt = env.physics.dt
            start, end = row['storage_start'], row['storage_end']
            np.testing.assert_allclose(np.cumsum(end-start),
                dt*(np.cumsum(row['external'])-row['release']), rtol=0., atol=1e-4)
            np.testing.assert_array_equal(row['external'], env.data.external[step])
            tolerance = np.maximum(1., 1e-10*np.abs(start))
            self.assertTrue(np.all(end >= row['storage_lower_m3']-tolerance))
            self.assertTrue(np.all(end <= row['storage_upper_m3']+tolerance))
            self.assertTrue(np.all(row['release'] >= -1e-8))
            self.assertTrue(np.all(row['release'] <= self.data.release_cap+1e-8))
            self.assertTrue(np.all(row['power'] <= self.data.power_cap+1e-7))
            np.testing.assert_array_equal(row['low'], row['physical_low_m3s'])
            np.testing.assert_array_equal(row['high'], row['physical_high_m3s'])
            np.testing.assert_array_equal(row['controls'] == 0., row['high']-row['low'] <= 1e-8)

    def test_issued_forecast_changes_observations_without_changing_fixed_command_dispatch(self):
        changed = copy.copy(self.data)
        changed.inflow_forecast = self.data.inflow_forecast.copy()
        changed.inflow_forecast[0] += 1000.*np.arange(1., 6.)
        original, altered = self.environment(), self.environment(data=changed)
        self.assertFalse(np.array_equal(original.actor_observations(), altered.actor_observations()))
        self.assertFalse(np.array_equal(original.state(), altered.state()))
        original.step(np.arange(5))
        altered.step(np.arange(5))
        for name in ('release', 'inflow', 'storage_end', 'controls', 'power', 'mixed_rewards'):
            np.testing.assert_array_equal(original.last_transition[name], altered.last_transition[name])

    def test_future_observed_inflow_does_not_affect_earlier_physical_decisions(self):
        changed = copy.copy(self.data)
        changed.external = self.data.external.copy()
        changed.external[2:] += 10000.
        original, altered = self.environment(days=4), self.environment(days=4, data=changed)
        for step in range(2):
            np.testing.assert_array_equal(original.state(), altered.state())
            np.testing.assert_array_equal(original.actor_observations(), altered.actor_observations())
            original.step((np.arange(5)+step) % 11)
            altered.step((np.arange(5)+step) % 11)
            for name in ('release', 'inflow', 'storage_end', 'controls', 'power', 'mixed_rewards'):
                np.testing.assert_array_equal(original.last_transition[name], altered.last_transition[name])

    def test_physical_kernel_preserves_quadratic_grid_and_only_forces_collapsed_ranges(self):
        storage = np.full((1, 5), 10.)
        external = np.array([[2., 1., 1., 1., 1.]])
        actions = np.array([[0., .25, .5, .75, 1.]])
        result = dispatch_release(storage, external, actions, np.zeros((1, 5)),
                                  np.full((1, 5), 20.), np.full(5, 10.), 1.)
        np.testing.assert_array_equal(result['release'][0], [0., .625, 2.5, 5.625, 10.])
        np.testing.assert_array_equal(result['controls'], np.ones((1, 5)))
        for commands in (np.zeros((1, 5)), np.ones((1, 5))):
            forced = dispatch_release(storage, external, commands, storage, storage, np.full(5, 10.), 1.)
            np.testing.assert_array_equal(forced['release'][0], [2., 3., 4., 5., 6.])
            np.testing.assert_array_equal(forced['controls'], np.zeros((1, 5)))

    def test_removed_scheduling_keyword_is_not_a_hidden_disabled_code_path(self):
        for function in (dispatch_release, ReservoirPhysics.transition, batched_transition):
            self.assertNotIn('terminal', inspect.signature(function).parameters)
        physics = ReservoirPhysics(self.data, self.physical)
        with self.assertRaises(TypeError):
            physics.transition(self.data.initial_storage(1965), self.data.external[0],
                               np.zeros(5), terminal={})


if __name__ == '__main__':
    unittest.main(verbosity=2)
