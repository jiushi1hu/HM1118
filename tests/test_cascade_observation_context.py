"""Issued forecasts remain visible while endpoint storage is unconstrained."""
import copy
from datetime import date
from pathlib import Path
import unittest
import numpy as np
import yaml
from harl.envs.cascade_reservoir.cascade_reservoir_env import (
    ACTOR_OBSERVATION_LAYOUT, CRITIC_STATE_LAYOUT, critic_state_size)
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.episodes import episode_environment
ROOT = Path(__file__).resolve().parents[1]

class CascadeObservationContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical)

    def environment(self, limits, source=None, years=None, days=4):
        return episode_environment(self.data if source is None else source,
            dict(self.physical, use_daily_level_limits=limits), [1965] if years is None else years,
            days=days, action_bins=11)

    def assert_context(self, env):
        forecast = ((env.data.inflow_forecast[env.index]-self.data.flow_mean)/self.data.flow_std).astype(np.float32)
        np.testing.assert_array_equal(env.state()[-5:], forecast)
        np.testing.assert_array_equal(env.actor_observations()[:, -1], forecast)

    def test_spaces_metadata_and_preserved_physical_prefix(self):
        for limits, actor_size, state_size in ((True, 16, 43), (False, 12, 23)):
            with self.subTest(limits=limits):
                env = self.environment(limits)
                obs, shared, available = env.reset()
                self.assertEqual(critic_state_size(limits), state_size)
                self.assertEqual(obs.shape, (5, actor_size))
                self.assertEqual(shared.shape, (5, state_size))
                self.assertIsNone(available)
                metadata = env.observation_metadata()
                self.assertEqual(metadata['actor_observation_layout'], ACTOR_OBSERVATION_LAYOUT)
                self.assertEqual(metadata['critic_state_layout'], CRITIC_STATE_LAYOUT)
                self.assertEqual(metadata['end_storage_policy'], 'unconstrained_within_daily_operating_bounds')
                self.assertEqual(env.observation_features[-1], 'next7d_mean_inflow_forecast_standardized')
                np.testing.assert_array_equal(env.state()[:5], ((env.storage-self.data.vmin)/(self.data.vmax-self.data.vmin)).astype(np.float32))
                np.testing.assert_array_equal(obs[:, :3], env.state()[:15].reshape(3, 5).T)
                np.testing.assert_array_equal(shared, np.repeat(env.state()[None], 5, axis=0))
                self.assert_context(env)

    def test_forecast_changes_only_corresponding_actor_and_critic_input(self):
        changed = copy.copy(self.data)
        changed.inflow_forecast = self.data.inflow_forecast.copy()
        changed.inflow_forecast[0, 2] += 3.*self.data.flow_std[2]
        for limits in (False, True):
            left, right = self.environment(limits), self.environment(limits, changed)
            np.testing.assert_array_equal(left.state()[:-5], right.state()[:-5])
            np.testing.assert_array_equal(left.actor_observations()[:, :-1], right.actor_observations()[:, :-1])
            np.testing.assert_array_equal(np.flatnonzero(left.state() != right.state()), [len(left.state())-3])
            self.assertAlmostEqual(float(right.state()[-3]-left.state()[-3]), 3., places=5)
            self.assert_context(right)

    def test_forecast_does_not_impose_dispatch_or_endpoint_restrictions(self):
        changed = copy.copy(self.data)
        changed.inflow_forecast = self.data.inflow_forecast+100000.
        for limits in (False, True):
            left, right = self.environment(limits), self.environment(limits, changed)
            for step in range(4):
                command = (np.arange(5)+step) % 11
                left.step(command)
                right.step(command)
                for key in ('release', 'storage_end', 'controls', 'power', 'low', 'high'):
                    np.testing.assert_array_equal(left.last_transition[key], right.last_transition[key])
            np.testing.assert_array_equal(left.state(), np.zeros(left.state_size))
            np.testing.assert_array_equal(left.actor_observations(), np.zeros((5, left.observation_size)))

    def test_forecasts_keep_source_dates_at_leaps_and_nonconsecutive_joins(self):
        indices = {day: i for i, day in enumerate(self.data.dates)}
        for limits in (False, True):
            env = self.environment(limits, years=[1968, 1970, 2000], days=1097)
            for day in (date(1968, 2, 29), date(1968, 12, 31), date(1970, 1, 1), date(2000, 2, 29)):
                env.index = env.data.dates.index(day)
                np.testing.assert_array_equal(env.data.inflow_forecast[env.index], self.data.inflow_forecast[indices[day]])
                self.assert_context(env)

    def test_future_actual_inflows_do_not_change_current_inputs_or_dispatch(self):
        changed = copy.copy(self.data)
        changed.external = self.data.external.copy()
        changed.external[2:] += 10000.
        for limits in (False, True):
            left, right = self.environment(limits), self.environment(limits, changed)
            for step in range(2):
                np.testing.assert_array_equal(left.state(), right.state())
                command = (np.arange(5)+step) % 11
                left.step(command)
                right.step(command)
                np.testing.assert_array_equal(left.last_transition['release'], right.last_transition['release'])
            self.assertFalse(np.array_equal(left.state()[5:10], right.state()[5:10]))

if __name__ == '__main__':
    unittest.main()
