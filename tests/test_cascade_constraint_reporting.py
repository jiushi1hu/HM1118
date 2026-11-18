"""Physical completion, soft ecological service and neutral storage accounting."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.evaluation import episode_constraint_status
from harl.envs.cascade_reservoir.periodic_evaluation import evaluate_periodic_policy
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.envs.cascade_reservoir.storage_accounting import storage_change_summary


def shortfalls(count=0, active=5):
    return {'ecological_flow':dict(reservoir_day_count=count, active_reservoir_days=active)}


class ConstraintStatusTest(unittest.TestCase):
    def test_completed_physical_operation_is_success_even_with_ecological_deficits(self):
        row = episode_constraint_status(completed=True, physical_feasible=True,
                                        service_shortfalls=shortfalls(2))
        self.assertTrue(row['completed'])
        self.assertTrue(row['physical_feasible'])
        self.assertTrue(row['feasible'])
        self.assertTrue(row['required_constraints_met'])
        self.assertFalse(row['ecological_compliant'])
        self.assertEqual(set(row), {'completed','physical_feasible','ecological_compliant',
            'required_constraints_met','feasible','feasible_definition','constraint_status_definition'})

    def test_partial_and_unavailable_service_measurements_remain_unknown(self):
        partial = episode_constraint_status(completed=False, physical_feasible=False,
                                            service_shortfalls=shortfalls())
        self.assertFalse(partial['required_constraints_met'])
        self.assertIsNone(partial['ecological_compliant'])
        inactive = episode_constraint_status(completed=True, physical_feasible=True,
                                             service_shortfalls=shortfalls(active=0))
        self.assertIsNone(inactive['ecological_compliant'])
        self.assertTrue(inactive['required_constraints_met'])
        unknown = episode_constraint_status(completed=True, physical_feasible=None)
        self.assertIsNone(unknown['required_constraints_met'])


class StorageAccountingTest(unittest.TestCase):
    def test_increases_and_depletion_are_signed_inventory_not_success_criteria(self):
        initial = np.asarray([100.,200.,300.,400.,500.])
        final = initial+np.asarray([10.,-20.,0.,30.,-40.])
        result = storage_change_summary(initial, final)
        self.assertEqual(result['storage_change_m3'], [10.,-20.,0.,30.,-40.])
        self.assertEqual(result['net_storage_change_m3'], -20.)
        self.assertEqual(result['total_storage_depletion_m3'], 20.)
        self.assertEqual(result['total_storage_depletion_billion_m3'], 2e-8)
        self.assertEqual(result['initial_storage_m3'], initial.tolist())
        self.assertEqual(result['final_storage_m3'], final.tolist())
        self.assertNotIn('met', result)
        self.assertNotIn('tolerance_m3', result)

    def test_missing_endpoints_and_nonfinite_or_wrong_shapes_are_explicit(self):
        self.assertIsNone(storage_change_summary([1.]*5)['storage_change_m3'])
        for start,end in (([1.]*4,[2.]*5),([1.]*5,[float('nan')]*5),([float('inf')]*5,[2.]*5)):
            with self.subTest(start=start,end=end),self.assertRaises(ValueError):
                storage_change_summary(start,end)


class PeriodicConstraintStatusTest(unittest.TestCase):
    def measured(self, value):
        row = dict(episode_reward=value, episode_system_reward=value,
            episode_global_reward=value, episode_mixed_rewards=[value]*5,
            episode_local_rewards=[value]*5, energy_MWh=value, energy_TWh=value/1e6,
            episode_penalty_ecological_flow=0., violation_count=0,
            max_daily_bound_excess_m3=0., service_shortfalls=shortfalls())
        row.update({name+'_end_m3':1e9-100. for name in ('WDD','BHT','XLD','XJB','THR')})
        return row

    def evaluate(self, measurements):
        windows = [dict(window_id=str(i),source_years=[2001+i],episode_horizon_days=1,
            start_date=str(2001+i)+'-01-01',initial_storage_m3=[1e9]*5) for i in range(3)]
        with mock.patch('harl.envs.cascade_reservoir.periodic_evaluation._validated_windows',return_value=windows), \
                mock.patch('harl.envs.cascade_reservoir.periodic_evaluation.evaluate_policy',side_effect=measurements):
            return evaluate_periodic_policy([],SimpleNamespace(levels=lambda values:np.asarray(values)),
                {},seed=1,evaluation_round=1,training_steps=3,windows=windows,gamma=.995)

    def test_storage_depletion_does_not_disqualify_completed_physical_returns(self):
        row = self.evaluate([self.measured(10.),self.measured(20.),self.measured(30.)])
        self.assertEqual(row['status'],'ok')
        self.assertEqual(row['completed_window_count'],3)
        self.assertTrue(row['required_constraints_met'])
        self.assertEqual(row['evaluator_return'],20.)
        self.assertEqual(row['windows'][0]['storage_change_m3'],[-100.]*5)
        self.assertEqual(row['windows'][0]['total_storage_depletion_m3'],500.)
        self.assertFalse(any(key.startswith('constraint_qualified_') for key in row))

    def test_partial_failure_never_averages_surviving_windows(self):
        row = self.evaluate([self.measured(10.),InfeasibleCascade('failed day'),self.measured(30.)])
        self.assertEqual(row['completed_window_count'],2)
        self.assertFalse(row['completed'])
        self.assertFalse(row['physical_feasible'])
        self.assertFalse(row['required_constraints_met'])
        self.assertIsNone(row['evaluator_return'])


if __name__ == '__main__':
    unittest.main()
