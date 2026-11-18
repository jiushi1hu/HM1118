"""Independent selected-episode checks detect corrupted numerical evidence."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.correctness_diagnostics import CorrectnessRecorder
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import episode_environment


def golden_transition():
    """Hand-calculated constant-storage chain, including inactive ecology."""
    storage = np.full(5, 43200000.)
    release = np.array([5., 10., 20., 30., 40.])
    power = np.array([4.4145, 8.829, 17.658, 26.487, 35.316])
    local_generation = np.array([.0044145, .008829, .017658, .026487, .035316])
    local_cost = np.array([.5, .5, 0., .25, .2])
    local = np.array([-.4955855, -.491171, .017658, -.223513, -.164684])
    mixed = np.array([-.2385223, -.23631505, .01809945, -.10248605, -.07307155])
    row = dict(date='2001-01-01', storage_start=storage.copy(), storage_end=storage.copy(),
        levels_start=np.full(5, 5.), levels_end=np.full(5, 5.),
        external=np.array([5., 5., 10., 10., 10.]), inflow=release.copy(),
        release=release.copy(), low=np.zeros(5), high=np.full(5, 10000.), controls=np.ones(5),
        power=power, energy=24.*power, turbine=release.copy(), spill=np.zeros(5),
        net_head_m=np.full(5, 100.), local_generation_rewards=local_generation,
        local_reward_penalties=local_cost, local_rewards=local, mixed_rewards=mixed,
        generation_reward=.0185409, global_reward=.0185409, reward_penalty=.3625,
        reward=-.3439591, system_reward=-.6693773)
    data = SimpleNamespace(power_cap=np.full(5, 1000.), release_cap=np.full(5, 10000.),
        turbine_cap=np.full(5, 1000.), vmin=np.zeros(5), vmax=np.full(5, 864000000.),
        head=np.full(5, 100.), initial_storage=lambda year: storage.copy(),
        anchors={name: np.array([[0., 0.], [100., 864000000.]]) for name in IDS},
        service_requirements=SimpleNamespace(ecological_flow_m3s=np.array([10., 20., 0., 40., 50.])))
    physical = dict(timestep_seconds=86400., efficiency=.9, water_density_kg_m3=1000.,
        gravity_m_s2=9.81, head_model='fixed',
        reward_penalties=dict(enabled=True, ecological_flow_weight=1.))
    return data, physical, row


class CascadeCorrectnessDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load(
            (ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())

    def test_hand_calculated_rewards_and_physical_accounting(self):
        data, physical, row = golden_transition()
        recorder = CorrectnessRecorder(data, physical, seed=3, max_steps=1)
        recorder.begin_episode(7, [2001])
        recorder.capture(row)
        result = recorder.finish()
        self.assertEqual(result['steps'], 1)
        self.assertEqual(result['episode'], 7)
        self.assertEqual(result['seed'], 3)
        self.assertEqual(result['summary']['hard_physical_reservoir_days'], 0)
        self.assertEqual(result['summary']['reward_mismatch_days'], 0)
        self.assertEqual(result['summary']['violation_counts']['generation_calculation'], 0)
        self.assertEqual(result['summary']['violation_counts']['level_lookup'], 0)
        self.assertAlmostEqual(result['independent']['reward'][0], -.3439591)
        self.assertAlmostEqual(result['independent']['system_reward'][0], -.6693773)
        np.testing.assert_allclose(result['independent']['mixed_rewards'][0], row['mixed_rewards'], atol=1e-12)
        np.testing.assert_array_equal(result['independent']['mass_balance_residual_m3'], np.zeros((1, 5)))
        self.assertEqual(result['independent']['cascade_mass_balance_residual_m3'], [0.])
        self.assertEqual(result['summary']['active_ecological_reservoir_days'], 4)
        self.assertEqual(result['summary']['ecological_violation_rate'], 1.)
        # Ecological deficits are soft events, not failures of hard dispatch.
        self.assertEqual(result['summary']['violation_counts']['ecological'], 4)
        self.assertEqual(result['storage']['storage_change_m3'], [0.]*5)
        self.assertEqual(result['storage']['net_storage_change_m3'], 0.)
        json.dumps(result, allow_nan=False)

    def test_corrupted_storage_inflow_and_reward_are_exposed_without_repair(self):
        data, physical, original = golden_transition()
        cases = (
            ('storage_end', 2, 100., 'mass_balance'),
            ('inflow', 3, 1., 'inflow_routing'),
            ('system_reward', None, .2, 'reward_mismatch'),
        )
        for field, reservoir, increment, flag in cases:
            with self.subTest(corruption=field):
                row = copy.deepcopy(original)
                if reservoir is None:
                    row[field] += increment
                else:
                    row[field][reservoir] += increment
                recorder = CorrectnessRecorder(data, physical, 1)
                recorder.begin_episode(1, [2001])
                recorder.capture(row)
                result = recorder.finish()
                self.assertGreater(result['summary']['violation_counts'][flag], 0)
                np.testing.assert_array_equal(result['series'][field][0], row[field])
                if field == 'storage_end':
                    self.assertAlmostEqual(result['independent']['mass_balance_residual_m3'][0][2], 100.)
                    self.assertAlmostEqual(result['independent']['cascade_mass_balance_residual_m3'][0], 100.)
                elif field == 'inflow':
                    self.assertAlmostEqual(result['independent']['inflow_routing_residual_m3s'][0][3], 1.)
                else:
                    self.assertAlmostEqual(result['reward_differences']['system_reward'][0], .2)

    def test_historical_scalar_physics_reconciles_both_heads_and_reward_modes(self):
        for head in ('fixed', 'storage_sensitivity'):
            for enabled in (False, True):
                with self.subTest(head=head, penalties=enabled):
                    physical = dict(self.physical, head_model=head,
                        reward_penalties=dict(enabled=enabled, ecological_flow_weight=1.))
                    data = CascadeData(ROOT/'data/cascade_reservoir_new', physical)
                    env = episode_environment(data, physical, [1965], days=4, action_bins=11)
                    recorder = CorrectnessRecorder(data, physical, 2, max_steps=4)
                    recorder.begin_episode(12, [1965])
                    storage = env.storage.copy()
                    for step in range(4):
                        env.step((np.arange(5)*3+step*7) % 11)
                        row = copy.deepcopy(env.last_transition)
                        recorder.capture(env.last_transition)
                        for name in ('storage_start', 'storage_end', 'power', 'mixed_rewards', 'release'):
                            np.testing.assert_array_equal(env.last_transition[name], row[name])
                    result = recorder.finish()
                    self.assertEqual(result['summary']['hard_physical_reservoir_days'], 0)
                    self.assertEqual(result['summary']['reward_mismatch_days'], 0)
                    self.assertEqual(result['summary']['violation_counts']['generation_calculation'], 0)
                    self.assertEqual(result['summary']['violation_counts']['level_lookup'], 0)
                    np.testing.assert_array_equal(result['storage']['initial_storage_m3'], storage)
                    np.testing.assert_array_equal(result['storage']['final_storage_m3'], env.storage)
                    np.testing.assert_array_equal(result['storage']['storage_change_m3'], env.storage-storage)
                    self.assertNotIn('met', result['storage'])
                    json.dumps(result, allow_nan=False)

    def test_final_day_inventory_change_is_neutral_and_preserves_physical_control(self):
        physical = dict(self.physical, use_daily_level_limits=False)
        data = CascadeData(ROOT/'data/cascade_reservoir_new', physical)
        env = episode_environment(data, physical, [1965], days=1, action_bins=11)
        initial = env.storage.copy()
        env.step(np.full(5, 10))
        longer = episode_environment(data, physical, [1965], days=2, action_bins=11)
        longer.step(np.full(5, 10))
        recorder = CorrectnessRecorder(data, physical, 1, max_steps=1)
        recorder.begin_episode(1, [1965])
        recorder.capture(env.last_transition)
        result = recorder.finish()
        self.assertEqual(result['summary']['hard_physical_reservoir_days'], 0)
        self.assertTrue(np.any(env.storage != initial))
        for field in ('release', 'storage_end', 'controls', 'low', 'high'):
            np.testing.assert_array_equal(env.last_transition[field], longer.last_transition[field])
        np.testing.assert_array_equal(result['storage']['initial_storage_m3'], initial)
        np.testing.assert_array_equal(result['storage']['final_storage_m3'], env.storage)
        np.testing.assert_array_equal(result['storage']['storage_change_m3'], env.storage-initial)
        self.assertAlmostEqual(result['storage']['total_storage_depletion_m3'], float((initial-env.storage).sum()))
        self.assertNotIn('met', result['storage'])
        self.assertNotIn('tolerance_m3', result['storage'])

    def test_bounded_replacement_and_detached_json_outputs(self):
        data, physical, row = golden_transition()
        recorder = CorrectnessRecorder(data, physical, 0, max_steps=1)
        with self.assertRaises(RuntimeError):
            recorder.capture(row)
        recorder.begin_episode(1, [2001])
        recorder.capture(row)
        with self.assertRaises(ValueError):
            recorder.capture(row)
        row['release'][0] = 999.
        result = recorder.finish()
        self.assertEqual(result['series']['release'][0][0], 5.)
        result['series']['release'][0][0] = -999.
        self.assertEqual(recorder.finish()['series']['release'][0][0], 5.)
        with self.assertRaises(RuntimeError):
            recorder.capture(row)
        recorder.begin_episode(2, [2001])
        self.assertEqual(recorder.finish()['steps'], 0)

    def test_unavailable_and_partial_evidence_is_json_compatible(self):
        data, physical, row = golden_transition()
        recorder = CorrectnessRecorder(data, physical, 0, max_steps=4)
        recorder.begin_episode(1, [2001])
        result = recorder.finish()
        self.assertEqual(result['steps'], 0)
        self.assertEqual(result['series'], {})
        self.assertIsNone(result['storage']['final_storage_m3'])
        self.assertIsNone(result['storage']['storage_change_m3'])
        self.assertIsNone(result['summary']['max_abs_reward_difference'])
        json.dumps(result, allow_nan=False)
        recorder.begin_episode(1, [2001])
        recorder.capture(row)
        partial = recorder.finish()
        self.assertEqual(partial['steps'], 1)
        np.testing.assert_array_equal(partial['storage']['final_storage_m3'], row['storage_end'])
        np.testing.assert_array_equal(partial['storage']['storage_change_m3'], np.zeros(5))
        json.dumps(partial, allow_nan=False)

    def test_disabled_services_and_zero_thresholds_have_no_division_or_fake_events(self):
        data, physical, row = golden_transition()
        data.service_requirements = None
        physical['reward_penalties']['enabled'] = False
        row.update(reward=row['generation_reward'], reward_penalty=0.,
            local_reward_penalties=np.zeros(5), local_rewards=row['local_generation_rewards'],
            mixed_rewards=.5*row['local_generation_rewards']+.5*row['global_reward'],
            system_reward=.5*sum(row['local_generation_rewards'])+.5*row['global_reward'])
        recorder = CorrectnessRecorder(data, physical, 1)
        recorder.begin_episode(1, [2001])
        recorder.capture(row)
        result = recorder.finish()
        self.assertIsNone(result['summary']['ecological_violation_rate'])
        self.assertEqual(result['summary']['violation_counts']['ecological'], 0)
        self.assertEqual(result['summary']['reward_mismatch_days'], 0)
        self.assertFalse(result['physical']['service_requirements_available'])
        np.testing.assert_array_equal(result['independent']['normalized_ecological_shortfall'], np.zeros((1, 5)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
