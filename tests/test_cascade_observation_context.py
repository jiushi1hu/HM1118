"""Exact 22-feature actor and 110-feature concatenated critic observations."""
import calendar
import copy
from datetime import date
from pathlib import Path
import unittest

import numpy as np
import yaml

from harl.envs.cascade_reservoir.cascade_reservoir_env import (
    ACTOR_OBSERVATION_FEATURES, ACTOR_OBSERVATION_LAYOUT, ACTOR_OBSERVATION_SIZE,
    CRITIC_STATE_LAYOUT, CRITIC_STATE_SIZE, critic_state_size)
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import episode_environment

ROOT = Path(__file__).resolve().parents[1]


class CascadeObservationContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load(
            (ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text(encoding='utf-8'))
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical)

    def environment(self, limits, source=None, years=None, days=4):
        return episode_environment(self.data if source is None else source,
            dict(self.physical, use_daily_level_limits=limits), [1965] if years is None else years,
            days=days, action_bins=11)

    def expected_observations(self, env):
        """Compute the documented layout independently of the environment helper."""
        data, index = env.data, env.index
        storage = env.storage
        levels = np.asarray([
            np.interp(storage[i], data.anchors[name][:, 1], data.anchors[name][:, 0])
            for i, name in enumerate(IDS)])
        level_min, level_max = data.levels(data.vmin), data.levels(data.vmax)
        level_span = level_max-level_min
        storage_span = data.vmax-data.vmin
        total_inflow = data.external[index].copy()
        total_inflow[1:] += env.previous_release[:-1]
        forecast = (data.inflow_forecast[index]-data.flow_mean)/data.flow_std
        next_lower, next_upper = env._next_state_level_bounds(index)
        day = data.dates[index]
        phase = 2.*np.pi*(day.timetuple().tm_yday-1)/(
            366 if calendar.isleap(day.year) else 365)
        result = np.zeros((5, 22), dtype=np.float64)
        for i in range(5):
            result[i] = [
                (levels[i]-level_min[i])/level_span[i],
                total_inflow[i]/data.release_cap[i],
                forecast[i], np.sin(phase), np.cos(phase),
                (next_upper[i]-level_min[i])/level_span[i],
                (next_lower[i]-level_min[i])/level_span[i],
                env.previous_requested_action[i],
                (storage[i]-data.vmin[i])/storage_span[i],
                np.clip((data.vmax[i]-storage[i])/storage_span[i], 0., 1.),
                env.previous_release[i]/data.release_cap[i],
                env.previous_turbine[i]/data.turbine_cap[i],
                env.previous_power[i]/data.power_cap[i],
                env.previous_realized_action[i],
                (levels[i-1]-level_min[i-1])/level_span[i-1] if i else 0.,
                env.previous_release[i-1]/data.release_cap[i-1] if i else 0.,
                forecast[i-1] if i else 0.,
                (levels[i+1]-level_min[i+1])/level_span[i+1] if i < 4 else 0.,
                (np.clip((data.vmax[i+1]-storage[i+1])/storage_span[i+1], 0., 1.)
                 if i < 4 else 1.),
                storage.sum()/data.vmax.sum(),
                env.previous_power.sum()/data.power_cap.sum(),
                1. if 6 <= day.month <= 9 else 0.,
            ]
        return result.astype(np.float32)

    def assert_context(self, env):
        forecast = ((env.data.inflow_forecast[env.index]-self.data.flow_mean)/
                    self.data.flow_std).astype(np.float32)
        np.testing.assert_array_equal(env.actor_observations()[:, 2], forecast)
        np.testing.assert_array_equal(env.state()[2::22], forecast)
        for i in range(1, 5):
            self.assertAlmostEqual(float(env.actor_observations()[i, 16]),
                                   float(forecast[i-1]), places=6)

    def test_spaces_metadata_and_exact_feature_layout(self):
        self.assertEqual(ACTOR_OBSERVATION_SIZE, 22)
        self.assertEqual(CRITIC_STATE_SIZE, 110)
        for limits in (True, False):
            with self.subTest(limits=limits):
                env = self.environment(limits)
                obs, shared, available = env.reset()
                self.assertEqual(critic_state_size(limits), 110)
                self.assertEqual(obs.shape, (5, 22))
                self.assertEqual(shared.shape, (5, 110))
                self.assertIsNone(available)
                metadata = env.observation_metadata()
                self.assertEqual(metadata['actor_observation_layout'], ACTOR_OBSERVATION_LAYOUT)
                self.assertEqual(metadata['critic_state_layout'], CRITIC_STATE_LAYOUT)
                self.assertEqual(tuple(metadata['actor_observation_features']),
                                 ACTOR_OBSERVATION_FEATURES)
                self.assertEqual(metadata['actor_observation_size'], 22)
                self.assertEqual(metadata['critic_state_size'], 110)
                self.assertEqual(metadata['end_storage_policy'],
                                 'unconstrained_within_daily_operating_bounds')
                np.testing.assert_array_equal(env.state(), obs.reshape(-1))
                np.testing.assert_array_equal(shared, np.repeat(env.state()[None], 5, axis=0))
                np.testing.assert_array_equal(obs[0, 14:17], np.zeros(3, dtype=np.float32))
                np.testing.assert_array_equal(obs[4, 17:19], np.array([0., 1.], dtype=np.float32))
                np.testing.assert_allclose(obs, self.expected_observations(env), rtol=1e-6, atol=1e-6)
                self.assert_context(env)

    def test_observation_values_follow_local_upstream_downstream_and_system_definitions(self):
        env = self.environment(True)
        np.testing.assert_allclose(env.actor_observations(), self.expected_observations(env),
                                   rtol=1e-6, atol=1e-6)
        for step in range(3):
            env.step((np.arange(5)+step) % 11)
            np.testing.assert_allclose(env.actor_observations(), self.expected_observations(env),
                                       rtol=1e-6, atol=1e-6)

    def test_forecast_changes_only_its_local_and_direct_downstream_observation_slots(self):
        changed = copy.copy(self.data)
        changed.inflow_forecast = self.data.inflow_forecast.copy()
        changed.inflow_forecast[0, 2] += 3.*self.data.flow_std[2]
        for limits in (False, True):
            left, right = self.environment(limits), self.environment(limits, changed)
            left_obs, right_obs = left.actor_observations(), right.actor_observations()
            # Actor XLD owns the forecast; actor XJB receives it as its direct upstream forecast.
            np.testing.assert_array_equal(np.flatnonzero(left_obs != right_obs), [2+22*2, 16+22*3])
            np.testing.assert_array_equal(np.flatnonzero(left.state() != right.state()), [2+22*2, 16+22*3])
            self.assertAlmostEqual(float(right_obs[2, 2]-left_obs[2, 2]), 3., places=5)
            self.assertAlmostEqual(float(right_obs[3, 16]-left_obs[3, 16]), 3., places=5)
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
            np.testing.assert_array_equal(left.state(), np.zeros(110))
            np.testing.assert_array_equal(left.actor_observations(), np.zeros((5, 22)))

    def test_forecasts_keep_source_dates_at_leaps_and_nonconsecutive_joins(self):
        indices = {day: i for i, day in enumerate(self.data.dates)}
        for limits in (False, True):
            env = self.environment(limits, years=[1968, 1970, 2000], days=1097)
            for day in (date(1968, 2, 29), date(1968, 12, 31), date(1970, 1, 1), date(2000, 2, 29)):
                env.index = env.data.dates.index(day)
                np.testing.assert_array_equal(env.data.inflow_forecast[env.index],
                                              self.data.inflow_forecast[indices[day]])
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
                np.testing.assert_array_equal(left.last_transition['release'],
                                              right.last_transition['release'])
            # The next decision uses external inflow row 2, which changed.
            self.assertFalse(np.array_equal(left.actor_observations(), right.actor_observations()))
            left.step(np.arange(5) % 11)
            right.step(np.arange(5) % 11)
            self.assertFalse(np.array_equal(left.last_transition['release'],
                                            right.last_transition['release']))


if __name__ == '__main__':
    unittest.main()
