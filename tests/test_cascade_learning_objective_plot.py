"""Return figures follow policy targets without relabeling legacy accounting."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import plot_cascade_parallel as parallel
from examples import plot_cascade_compare as compare


def windows():
    return [dict(window_id=name, window_start_year=year, source_years=[year],
                 episode_horizon_days=2, start_date=str(year)+'-01-01',
                 end_date=str(year)+'-01-02', initial_storage_m3=[1.]*5,
                 initial_level_m=[1.]*5, external_inflow_sha256='0'*64)
            for name, year in (('dry', 2000), ('median', 2001), ('wet', 2002))]


def objective(algorithm, counterfactual=False):
    team = algorithm == 'happo' and counterfactual
    return dict(
        advantage_source=('counterfactual_shared_team' if counterfactual else
                          'individual_mixed'),
        reward_formula=('0.5 * sum(local rewards) + 0.5 * global generation reward'
                        if team else '0.5 * own local reward + 0.5 * global generation reward'),
        critic_reward_layout='shared_team' if team else 'mixed_by_reservoir',
        actor_value_channels=[0]*5 if team else list(range(5)),
        normalization=('common_controllable_rms_without_centering' if counterfactual else
                       'once_over_common_rollout_rows' if team else
                       'per_reservoir_over_controllable_rows'))


def evaluation(seed, number, failed=False):
    rows = []
    for descriptor in windows():
        rows.append(dict(descriptor, status='failed' if failed else 'ok',
            evaluator_return=None if failed else 1000.*seed+number,
            system_evaluator_return=None if failed else 100.+10*seed+number,
            mixed_evaluator_returns=None if failed else [seed+number+i for i in range(1, 6)],
            ecological_flow_violation_count=None if failed else seed,
            ecological_flow_active_reservoir_days=None if failed else 10,
            ecological_flow_violation_rate=None if failed else seed/10.))
    return dict(seed=seed, evaluation_round=number, gamma=1.,
        status='failed' if failed else 'ok', window_count=3,
        successful_window_count=0 if failed else 3,
        evaluator_return=None if failed else 1000.*seed+number,
        system_evaluator_return=None if failed else 100.+10*seed+number,
        mixed_evaluator_returns=None if failed else [seed+number+i for i in range(1, 6)],
        ecological_flow_violation_count=None if failed else 3*seed,
        ecological_flow_active_reservoir_days=None if failed else 30,
        ecological_flow_violation_rate=None if failed else seed/10., windows=rows)


def run(algorithm, counterfactual=False):
    settings = dict(episode_horizon_days=2, parallel_environments=2,
        episodes_per_seed=4, gamma=1., action_bins=4, seeds=[1, 2],
        training_years=[2000, 2001, 2002], evaluation_source_years=[2003],
        window_sampling='coverage_balanced', physical_config={'head_model': 'fixed'},
        periodic_windows=windows(), evaluation_interval=1, device={'torch_threads': 1},
        train={'episode_length': 2, 'n_rollout_threads': 2},
        model={'action_bins': 4}, algo={'gamma': 1.})
    if counterfactual:
        settings['algo']['happo_credit_assignment'] = 'counterfactual'
    return dict(algorithm=algorithm, gamma=1., seeds=[1, 2], configured_seeds=[1, 2],
        episode_horizon_days=2, parallel_envs=2, updates_per_seed=2,
        learning_objective=objective(algorithm, counterfactual),
        comparison_context=dict(shared_settings=settings, data_hashes={'inflow.csv': 'a'*64}),
        histories=[[dict(training_steps=4*i, episode_reward=100.*seed+i,
                         episode_system_reward=10.*seed+i,
                         episode_mixed_rewards=[10.*(seed-1)+i+j for j in range(5)],
                         episode_energy_TWh=10.*seed+i)
                    for i in (1, 2)] for seed in (1, 2)],
        evaluation_histories=[[evaluation(seed, i) for i in (0, 1, 2)] for seed in (1, 2)])


@unittest.skipIf(plt is None, 'Matplotlib is available in the plotting interpreter')
class LearningReturnFigureTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_counterfactual_happo_plots_system_returns(self):
        for counterfactual in (True,):
            with self.subTest(counterfactual=counterfactual):
                payload = run('happo', counterfactual)
                before = copy.deepcopy(payload)
                training = parallel.make_figure(payload)
                evaluator = parallel.make_evaluator_figure(payload)
                np.testing.assert_allclose(training.axes[0].lines[0].get_ydata(), [16., 17.])
                np.testing.assert_allclose(evaluator.axes[0].lines[0].get_ydata(), [115., 116., 117.])
                self.assertIn('system', training.axes[0].get_ylabel().lower())
                self.assertIn('system', evaluator.axes[0].get_ylabel().lower())
                np.testing.assert_allclose(training.axes[1].lines[0].get_ydata(), [16., 17.])
                self.assertEqual(payload, before)

    def test_matched_happo_and_mappo_plot_mean_of_five_mixed_returns(self):
        for algorithm in ('happo', 'mappo'):
            with self.subTest(algorithm=algorithm):
                payload = run(algorithm)
                training = parallel.make_figure(payload)
                evaluator = parallel.make_evaluator_figure(payload)
                np.testing.assert_allclose(training.axes[0].lines[0].get_ydata(), [8., 9.])
                np.testing.assert_allclose(evaluator.axes[0].lines[0].get_ydata(), [4.5, 5.5, 6.5])
                self.assertIn('mixed', training.axes[0].get_ylabel().lower())
                self.assertIn('mixed', evaluator.axes[0].get_ylabel().lower())

    def test_mappo_scalar_components_reconstruct_mean_mixed_in_memory(self):
        payload = run('mappo')
        for history in payload['histories']:
            for row in history:
                mean_mixed = np.mean(row.pop('episode_mixed_rewards'))
                row['episode_global_reward'] = (5*mean_mixed-row['episode_system_reward'])/2
        training = parallel.make_figure(payload)
        np.testing.assert_allclose(training.axes[0].lines[0].get_ydata(), [8., 9.])

    def test_mixed_return_requires_exactly_five_finite_values(self):
        for values in ([1., 2., 3., 4.], [1., 2., 3., 4., np.nan], [[1., 2., 3., 4., 5.]]):
            with self.subTest(values=values):
                payload = run('mappo')
                payload['histories'][0][0]['episode_mixed_rewards'] = values
                with self.assertRaises(ValueError):
                    parallel.make_figure(payload)

    def test_selected_evaluation_aggregate_must_match_all_three_windows(self):
        for algorithm in ('happo', 'mappo'):
            with self.subTest(algorithm=algorithm):
                payload = run(algorithm)
                row = payload['evaluation_histories'][0][0]
                if algorithm == 'happo':
                    row['system_evaluator_return'] += 1.
                else:
                    row['mixed_evaluator_returns'][0] += 5.
                with self.assertRaises(ValueError):
                    parallel.make_evaluator_figure(payload)

    def test_comparison_lines_share_the_matched_learning_objective(self):
        figures = compare.make_figures(dict(schema=compare.SCHEMA, runs=[
            dict(algorithm=name, payload=run(name)) for name in ('happo', 'mappo')]))
        training, evaluator = figures['training'].axes[0], figures['evaluator'].axes[0]
        np.testing.assert_allclose(training.lines[0].get_ydata(), [8., 9.])
        np.testing.assert_allclose(training.lines[1].get_ydata(), [8., 9.])
        np.testing.assert_allclose(evaluator.lines[0].get_ydata(), [4.5, 5.5, 6.5])
        np.testing.assert_allclose(evaluator.lines[1].get_ydata(), [4.5, 5.5, 6.5])
        labels = ' '.join(training.get_legend_handles_labels()[1]).lower()
        self.assertIn('mixed', labels)
        np.testing.assert_allclose(figures['violations'].axes[0].lines[0].get_ydata(), [15.]*3)

    def test_declared_objective_never_uses_missing_legacy_only_returns(self):
        for algorithm in ('happo', 'mappo'):
            with self.subTest(algorithm=algorithm):
                payload = run(algorithm)
                for history in payload['histories']:
                    for row in history:
                        row.pop('episode_system_reward')
                        row.pop('episode_mixed_rewards')
                with self.assertRaises(ValueError):
                    parallel.make_figure(payload)
                for history in payload['evaluation_histories']:
                    for row in history:
                        row.pop('system_evaluator_return')
                        row.pop('mixed_evaluator_returns')
                        for window in row['windows']:
                            window.pop('system_evaluator_return')
                            window.pop('mixed_evaluator_returns')
                with self.assertRaises(ValueError):
                    parallel.make_evaluator_figure(payload)

    def test_unidentified_old_payload_is_explicitly_legacy(self):
        payload = run('happo')
        payload.pop('algorithm')
        payload.pop('learning_objective')
        training = parallel.make_figure(payload)
        evaluator = parallel.make_evaluator_figure(payload)
        np.testing.assert_allclose(training.axes[0].lines[0].get_ydata(), [151., 152.])
        np.testing.assert_allclose(evaluator.axes[0].lines[0].get_ydata(), [1500., 1501., 1502.])
        self.assertIn('legacy', training.axes[0].get_ylabel().lower())
        self.assertIn('legacy', evaluator.axes[0].get_ylabel().lower())

    def test_failed_seed_invalidates_learning_return_mean_for_that_round(self):
        payload = run('happo')
        payload['evaluation_histories'][1][1] = evaluation(2, 1, failed=True)
        figure = parallel.make_evaluator_figure(payload)
        values = figure.axes[0].lines[0].get_ydata()
        np.testing.assert_allclose(values[[0, 2]], [115., 117.])
        self.assertTrue(np.isnan(values[1]))


def log_rows(algorithm):
    payload = run(algorithm)
    settings = payload['comparison_context']['shared_settings']
    metadata = dict(record_type='run_configuration', algorithm=algorithm,
        episode_horizon_days=2, parallel_environments=2, updates_per_seed=2,
        config=dict(algo={'gamma': 1.}, experiment={'seeds': [1]}),
        learning_objective=payload['learning_objective'],
        actor_architecture={'critic_output_dim': 1 if algorithm == 'happo' else 5},
        periodic_evaluation={'windows': windows()}, training_years=settings['training_years'],
        evaluation_source_years=settings['evaluation_source_years'])
    episodes = [dict(record_type='training_episode', seed=1, episode=lane+1,
        completed=True, collecting_for_update=1, parallel_env_index=lane,
        parallel_envs=2, training_steps=4,
        learning_rewards=dict(episode_system_reward=10.+10*lane,
            episode_global_reward=2.+2*lane,
            episode_mixed_rewards=[lane*10.+j for j in range(1, 6)])) for lane in range(2)]
    update = dict(record_type='training_update', seed=1, update=1,
        training_steps=4, episode_horizon_days=2, parallel_envs=2,
        episode_reward_mean=999., episode_generation_TWh_mean=3.)
    return metadata, episodes, update


class LearningReturnLogTest(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episodes.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            return parallel.load_episode_log(path)

    def test_loader_joins_all_episode_lanes_and_preserves_objective_metadata(self):
        for algorithm in ('happo', 'mappo'):
            with self.subTest(algorithm=algorithm):
                metadata, episodes, update = log_rows(algorithm)
                loaded = self.load([metadata, *episodes, update])
                self.assertEqual(loaded['algorithm'], algorithm)
                self.assertEqual(loaded['learning_objective'], metadata['learning_objective'])
                self.assertEqual(loaded['actor_architecture'], metadata['actor_architecture'])
                row = loaded['histories'][0][0]
                self.assertEqual(row['episode_reward'], 999.)
                self.assertEqual(row['episode_system_reward'], 15.)
                self.assertEqual(row['episode_global_reward'], 3.)
                np.testing.assert_allclose(row['episode_mixed_rewards'], [6., 7., 8., 9., 10.])

    def test_loader_rejects_incomplete_or_duplicate_completed_batch(self):
        metadata, episodes, update = log_rows('happo')
        for lanes in ([episodes[0]], [episodes[0], episodes[0]]):
            with self.subTest(lanes=len(lanes)), self.assertRaises(ValueError):
                self.load([metadata, *lanes, update])

    def test_loader_rejects_distinct_episode_ids_reusing_one_worker_lane(self):
        metadata, episodes, update = log_rows('happo')
        episodes[1]['parallel_env_index'] = episodes[0]['parallel_env_index']
        with self.assertRaises(ValueError):
            self.load([metadata, *episodes, update])

    def test_loader_does_not_publish_pending_episode_without_update(self):
        metadata, episodes, update = log_rows('happo')
        pending = copy.deepcopy(episodes[0])
        pending.update(episode=3, collecting_for_update=2, training_steps=8)
        pending['learning_rewards']['episode_system_reward'] = 123456.
        loaded = self.load([metadata, *episodes, update, pending])
        self.assertEqual(len(loaded['histories'][0]), 1)
        self.assertEqual(loaded['histories'][0][0]['episode_system_reward'], 15.)


if __name__ == '__main__':
    unittest.main()
