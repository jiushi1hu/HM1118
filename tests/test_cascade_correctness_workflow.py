"""Correctness evidence from real training and final-policy evaluation."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import train_cascade as cli
from examples import train_cascade_compare as compare
from examples import train_cascade_parallel as parallel
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.envs.cascade_reservoir.reservoir_physics import InfeasibleCascade
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


class CorrectnessWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.physical = cli.load_configuration()
        cls.config['train'].update(episode_length=3, n_rollout_threads=2)
        cls.config['model'].update(hidden_sizes=[8, 8], action_bins=5)
        cls.config['algo'].update(ppo_epoch=1, critic_epoch=1)
        cls.config['experiment'].update(episodes=2, seeds=[1])
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical,
                               cls.config['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=3)

    def runner(self, enabled):
        runner = CascadeReservoirRunner(self.data, self.physical, self.config, 1,
            periodic_windows=self.windows, correctness_enabled=enabled)
        self.addCleanup(runner.close)
        return runner

    def test_observation_preserves_models_rng_and_actual_evaluation(self):
        with redirect_stdout(io.StringIO()):
            baseline = self.runner(False)
            baseline.run(2)
            expected_rng = torch.get_rng_state().clone()
            observed = self.runner(True)
            observed.run(2)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for first, second in zip(baseline.actor, observed.actor):
            for name, value in first.actor.state_dict().items():
                self.assertTrue(torch.equal(value, second.actor.state_dict()[name]))
        for name, value in baseline.critic.critic.state_dict().items():
            self.assertTrue(torch.equal(value, observed.critic.critic.state_dict()[name]))
        for name, value in baseline.value_normalizer.state_dict().items():
            self.assertTrue(torch.equal(value, observed.value_normalizer.state_dict()[name]))
        self.assertEqual(baseline.periodic_evaluation_history, observed.periodic_evaluation_history)
        evidence = observed.correctness_trajectory
        self.assertEqual((evidence['steps'], evidence['expected_steps'], evidence['policy_update']), (3, 3, 1))
        self.assertEqual(evidence['scenario_id'], self.windows[1]['window_id'])
        self.assertEqual(evidence['status'], 'ok')
        self.assertEqual(evidence['summary']['hard_physical_reservoir_days'], 0)
        self.assertLess(evidence['summary']['max_abs_reward_difference'], 1e-10)
        self.assertEqual(len(observed.correctness_update_history), 1)
        self.assertEqual(len(observed.correctness_update_history[0]['learning_metrics']), 5)
        # The selected trajectory is the final deterministic policy, rather than
        # a stochastic trajectory collected before its final optimizer update.
        window = observed.periodic_evaluation_history[-1]['windows'][1]
        np.testing.assert_allclose(np.sum(evidence['series']['energy']), window['generation_MWh'])

    def test_experiment_saves_reusable_evidence_and_dispatches_seven_figures(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)/'correctness'
            args = SimpleNamespace(episode_log=None, no_plot=True, skip_final_evaluation=False,
                check_data_only=False, plot_python='/plot/python', correctness_output=output)
            with mock.patch.object(parallel, 'PERIODIC_EVALUATION_DAYS', 3), \
                    mock.patch.object(parallel, 'select_periodic_windows', return_value=self.windows), \
                    mock.patch.object(parallel.subprocess, 'run') as render:
                parallel.run_parallel_experiment(self.data, self.physical,
                                                copy.deepcopy(self.config), args)
            payload = json.loads((output/'evidence.json').read_text())
            self.assertEqual(payload['schema'], 'cascade_correctness_v2')
            self.assertEqual(payload['configuration']['correctness_diagnostics']['figures'],
                ['mass_balance.png', 'water_levels.png', 'release_feasibility.png',
                 'storage_change.png', 'reward_reconciliation.png', 'happo_updates.png',
                 'operation_trajectories.png'])
            run = payload['runs'][0]
            self.assertEqual(run['seed'], 1)
            self.assertEqual(run['trajectory']['steps'], 3)
            self.assertIn('operation_trajectory', run)
            self.assertEqual(len(run['storage_records']), 2+2*3+1)
            self.assertEqual(run['storage_summary']['training']['record_count'], 2)
            self.assertEqual(run['storage_summary']['training_evaluation']['record_count'], 6)
            self.assertEqual({row['phase'] for row in run['storage_records']},
                             {'training', 'training_evaluation', 'evaluation'})
            self.assertTrue(all(row['available'] for row in run['storage_records']))
            commands = [call.args[0] for call in render.call_args_list]
            self.assertEqual(commands[0], ['/plot/python', str(ROOT/'examples/plot_cascade_correctness.py'),
                                            '--check-backend'])
            self.assertEqual(commands[-1], ['/plot/python', str(ROOT/'examples/plot_cascade_correctness.py'),
                '--input', str(output/'evidence.json'), '--output-dir', str(output), '--no-show'])

    def test_held_out_failure_preserves_available_evidence_and_still_raises(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)/'failed_evaluation'
            args = SimpleNamespace(episode_log=None, no_plot=True, skip_final_evaluation=False,
                check_data_only=False, plot_python='/plot/python', correctness_output=output)
            error = InfeasibleCascade('Synthetic held-out physical failure')
            with mock.patch.object(parallel, 'PERIODIC_EVALUATION_DAYS', 3), \
                    mock.patch.object(parallel, 'select_periodic_windows', return_value=self.windows), \
                    mock.patch.object(parallel.subprocess, 'run'), \
                    mock.patch.object(CascadeReservoirRunner, 'final_evaluation', side_effect=error):
                with self.assertRaisesRegex(InfeasibleCascade, 'Synthetic held-out'):
                    parallel.run_parallel_experiment(self.data, self.physical,
                                                    copy.deepcopy(self.config), args)
            payload = json.loads((output/'evidence.json').read_text())
            self.assertEqual(payload['evaluation_failure']['seed'], 1)
            self.assertEqual(payload['runs'][0]['trajectory']['steps'], 3)
            inventory = payload['runs'][0]['storage_records'][-1]
            self.assertFalse(inventory['available'])
            self.assertEqual(inventory['status'], 'failed')


class CorrectnessCLIValidationTest(unittest.TestCase):
    def test_existing_output_is_rejected_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()), \
                mock.patch.object(cli, 'CascadeData', side_effect=AssertionError('Unexpected data loading')):
            with self.assertRaises(SystemExit):
                cli.main(['--correctness-output', directory, '--no-plot'])

    def test_overlapping_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'new'
            for other in (target, target/'training.png', target.parent):
                with self.subTest(other=other), self.assertRaises(ValueError):
                    parallel.validate_correctness_output(target, [other])

    def test_mappo_is_rejected_before_any_experiment(self):
        config, _ = cli.load_configuration()
        args = SimpleNamespace(algorithms=['happo', 'mappo'], correctness_output=Path('/tmp/new_correctness'))
        with self.assertRaisesRegex(ValueError, 'require HAPPO'):
            compare.comparison_preflight(args, config)

    def test_comparison_cannot_place_its_figures_inside_correctness_directory(self):
        config, _ = cli.load_configuration()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'new'
            args = SimpleNamespace(algorithms=['happo'], correctness_output=target,
                episode_log=None, trajectory_diagnostics=False, save_checkpoints=False,
                no_plot=False, plot_output=target/'results')
            with self.assertRaisesRegex(ValueError, 'separate'):
                compare.comparison_preflight(args, config)


if __name__ == '__main__':
    unittest.main()
