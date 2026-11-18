"""Five-channel HAPPO mixed-reward GAE and sequential-update contracts."""
import copy
import csv
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import yaml

from harl.envs.cascade_reservoir.checkpoint import ReplayTraceWriter, load_checkpoint, save_checkpoint
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ha_runner import OnPolicyHARunner

ROOT = Path(__file__).resolve().parents[1]


class MixedGAERunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = yaml.safe_load(
            (ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text(encoding='utf-8'))
        cls.physical = yaml.safe_load(
            (ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text(encoding='utf-8'))
        cls.args['train'].update(episode_length=4, n_rollout_threads=2)
        cls.args['model'].update(hidden_sizes=[8, 8], action_bins=7)
        cls.args['algo'].update(ppo_epoch=1, critic_epoch=1,
                                happo_credit_assignment='mixed_gae')
        cls.args['experiment'].update(episodes=2, seeds=[1], diagnostics=False)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical,
                               cls.args['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=4)

    def runner(self):
        runner = CascadeReservoirRunner(self.data, self.physical, self.args, 1,
                                        periodic_windows=self.windows)
        self.addCleanup(runner.close)
        return runner

    def test_mixed_rewards_map_to_critic_channels_and_independent_gae(self):
        runner = self.runner()
        captured = []
        original = runner._critic_rewards

        def capture(transitions):
            captured.extend(transitions)
            return original(transitions)

        runner._critic_rewards = capture
        runner.collect_batch(2)
        cb = runner.critic_buffer
        expected_rewards = np.asarray(
            [[row['mixed_rewards'] for row in captured]], dtype=np.float32
        ).reshape(runner.length, runner.n_envs, 5)
        np.testing.assert_array_equal(cb.rewards, expected_rewards)
        self.assertEqual(cb.rewards.shape, (4, 2, 5))
        self.assertEqual(cb.returns.shape, (5, 2, 5))
        self.assertEqual(runner._learning_reward_metadata()['advantage_normalization'],
                         'per_reservoir_over_common_rollout_rows')

        values = cb.value_preds.copy()
        if runner.value_normalizer is not None:
            values = runner.value_normalizer.denormalize(values)
        gae = np.zeros((runner.n_envs, 5), dtype=np.float64)
        expected_returns = np.zeros_like(cb.returns, dtype=np.float64)
        for t in reversed(range(runner.length)):
            delta = (cb.rewards[t].astype(np.float64) +
                     cb.gamma*values[t+1].astype(np.float64)*cb.masks[t+1] -
                     values[t].astype(np.float64))
            gae = delta + cb.gamma*cb.gae_lambda*cb.masks[t+1]*gae
            expected_returns[t] = gae + values[t].astype(np.float64)
        np.testing.assert_allclose(cb.returns[:-1], expected_returns[:-1],
                                   rtol=2e-6, atol=2e-6)
        self.assertTrue(all(len(runner.last_rollout[name]) == 5 for name in
                            ('gae_target_mean_by_reservoir',
                             'value_prediction_mean_by_reservoir',
                             'gae_target_rmse_by_reservoir')))

    def test_mixed_checkpoint_replay_has_five_own_mixed_channels(self):
        runner = self.runner()
        runner.collect_batch(2)
        runner.update_count, runner.completed_episodes, runner.training_steps = 1, 2, 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mixed.pt'
            save_checkpoint(runner, path)
            policy = load_checkpoint(path)
            self.assertEqual(policy.critic_reward_layout, 'mixed_by_reservoir')
            self.assertEqual(policy.critic_output_dim, 5)
            self.assertEqual(policy.actor_value_channels, (0, 1, 2, 3, 4))
            result = policy.replay([1965])
            self.assertEqual(result['critic_output_dim'], 5)
            stream = io.StringIO()
            policy.replay([1965], writer=ReplayTraceWriter(stream))
            rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
            self.assertTrue(all(row['value_basis'] == 'own_mixed_reward' for row in rows))
            self.assertTrue(all(row['legacy_team_value'] == '' for row in rows))
            for i in range(5):
                own = rows[i::5]
                self.assertTrue(all(int(row['value_channel']) == i for row in own))
                np.testing.assert_allclose(
                    [float(row['learning_reward']) for row in own],
                    [float(row['mixed_reward']) for row in own])

    def test_team_gae_remains_scalar_for_the_same_environment(self):
        args = copy.deepcopy(self.args)
        args['algo']['happo_credit_assignment'] = 'team_gae'
        runner = CascadeReservoirRunner(self.data, self.physical, args, 1,
                                        periodic_windows=self.windows)
        self.addCleanup(runner.close)
        self.assertEqual(runner.critic_reward_layout, 'shared_team')
        self.assertEqual(runner.critic_output_dim, 1)
        self.assertEqual(runner.actor_value_channels, (0, 0, 0, 0, 0))


class MixedSequentialUpdateTest(unittest.TestCase):
    def test_each_actor_gets_its_channel_and_factor_accumulates_in_order(self):
        runner = OnPolicyHARunner()
        runner.num_agents, runner.fixed_order, runner.action_aggregation = 5, True, 'prod'
        runner.credit_assignment = 'mixed_gae'
        runner.algo_args = dict(
            train=dict(episode_length=2, n_rollout_threads=1),
            algo=dict(mask_inactive_factors=True, normalize_shared_advantages=True))
        raw = np.array([[[-4., -1., 0., 2., 5.]],
                        [[1., 3., -2., 0., -5.]]], dtype=np.float32)
        returns = np.concatenate([raw, np.zeros((1, 1, 5), dtype=np.float32)])
        runner.critic_buffer = SimpleNamespace(
            returns=returns, value_preds=np.zeros((3, 1, 5), dtype=np.float32))
        runner.value_normalizer = None
        captured = []

        class Buffer:
            def __init__(self):
                self.obs = np.zeros((3, 1, 2), dtype=np.float32)
                self.actions = np.zeros((2, 1, 1), dtype=np.float32)
                self.available_actions = None
                self.active_masks = np.ones((3, 1, 1), dtype=np.float32)
                self.factor = np.ones((2, 1, 1), dtype=np.float32)

            def update_factor(self, factor):
                self.factor = factor.copy()

        class Actor:
            def __init__(self, agent_id):
                self.agent_id = agent_id
                self.updated = False

            def evaluate_actions(self, observations, *unused, **kwargs):
                value = .2 if self.updated else 0.
                return torch.full((len(observations), 1), value), None, None

            def train(self, buffer, advantages, state_type, **options):
                self.test_case.assertEqual(state_type, 'EP')
                self.test_case.assertTrue(options['advantages_normalized'])
                captured.append((self.agent_id, advantages.copy(), buffer.factor.copy()))
                self.updated = True
                return {}

        runner.actor_buffer = [Buffer() for _ in range(5)]
        actors = [Actor(i) for i in range(5)]
        for actor in actors:
            actor.test_case = self
        runner.actor = actors
        runner.critic = SimpleNamespace(train=mock.Mock(return_value={'value_loss': 0.}))
        runner.train()

        expected = ((raw-raw.mean(axis=(0, 1), keepdims=True)) /
                    (raw.std(axis=(0, 1), keepdims=True)+1e-5))
        self.assertEqual([item[0] for item in captured], list(range(5)))
        for agent_id, advantages, _ in captured:
            np.testing.assert_allclose(advantages, expected[..., agent_id:agent_id+1],
                                       rtol=1e-6, atol=1e-6)
        factor = np.exp(.2)
        for position, (_, _, used_factor) in enumerate(captured):
            np.testing.assert_allclose(used_factor, np.full((2, 1, 1), factor**position),
                                       rtol=1e-6, atol=1e-6)
        runner.critic.train.assert_called_once_with(runner.critic_buffer, None,
                                                    update_normalizer_once=True)


if __name__ == '__main__':
    unittest.main()
