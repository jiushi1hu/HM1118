"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


def _correctness_statistics(values):
    """Keep only bounded summaries, including explicit empty controllable sets."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return dict(count=0, min=None, median=None, p95=None, max=None)
    return dict(count=int(values.size), min=float(values.min()),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(values.max()))


@torch.no_grad()
def _correctness_distribution(actor, buffer):
    """Reconstruct full categorical log probabilities in a separate float64 path.

    The independent HAPPO actor is still its collection policy when this is
    called before its update. This observer neither samples nor advances Adam.
    """
    distribution = actor.rollout_distribution(buffer)
    return torch.log_softmax(distribution.logits.detach().to(dtype=torch.float64),
                             dim=-1).cpu().numpy()


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self):
        """Train the model."""
        actor_train_infos = [None] * self.num_agents

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        mode = getattr(self, 'credit_assignment', 'team_gae')
        counterfactual = mode == 'counterfactual'
        mixed = mode == 'mixed_gae'
        if counterfactual or not mixed:
            if advantages.shape[-1] != 1:
                raise ValueError('team_gae and counterfactual HAPPO require one shared-team value channel')
        elif advantages.shape[-1] != self.num_agents:
            raise ValueError('mixed_gae HAPPO requires one mixed-return value channel per reservoir')

        # team_gae normalizes one common team signal; mixed_gae normalizes each
        # reservoir channel over the same rollout rows. Counterfactual credit is
        # already frozen and normalized by its estimator.
        if counterfactual:
            credits = getattr(self, 'counterfactual_advantages', None)
            expected = advantages.shape[:2]+(self.num_agents,)
            if credits is None or credits.shape != expected or not np.isfinite(credits).all():
                raise ValueError('Counterfactual HAPPO requires frozen finite credit for every actor and rollout row')
        else:
            advantages = ((advantages - advantages.mean(axis=(0, 1), keepdims=True)) /
                          (advantages.std(axis=(0, 1), keepdims=True) + 1e-5))

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
        correctness = getattr(self, 'correctness_enabled', False)
        if correctness:
            correctness_actors = []
            prefix_log_ratio = np.zeros(factor.shape, dtype=np.float64)
        for position, agent_id in enumerate(agent_order):
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor
            if correctness:
                buffer = self.actor_buffer[agent_id]
                old_distribution_logp = _correctness_distribution(self.actor[agent_id], buffer)
                # Read the actual saved actor-entry F, not the local product
                # variable that the implementation will multiply below.
                used_factor = np.asarray(buffer.factor, dtype=np.float64)
                expected_factor = np.exp(prefix_log_ratio)
                error = np.abs(used_factor-expected_factor)
                correctness_actor = dict(agent_id=int(agent_id), update_position=position,
                    rows=int(used_factor.size),
                    used_factor=_correctness_statistics(used_factor),
                    expected_factor=_correctness_statistics(expected_factor),
                    factor_max_abs_error=float(error.max()),
                    factor_max_relative_error=float((error/np.maximum(
                        np.abs(expected_factor), np.finfo(np.float64).tiny)).max()))

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update actor
            train_options = {'advantages_normalized': True}
            if getattr(self, 'diagnostics_enabled', False):
                train_options['diagnostics'] = True
            actor_advantages = (credits[...,agent_id:agent_id+1] if counterfactual else
                                advantages[...,agent_id:agent_id+1] if mixed else advantages)
            # Only the sequential likelihood factor and each actor's control
            # mask differ; the frozen normalized advantage is consumed unchanged.
            actor_train_info = self.actor[agent_id].train(
                self.actor_buffer[agent_id],
                actor_advantages.copy(), "EP",
                **train_options
            )

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update factor for next agent
            completed_ratio = _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            if self.algo_args["algo"].get("mask_inactive_factors", False):
                completed_ratio = np.where(
                    self.actor_buffer[agent_id].active_masks[:-1] > 0,
                    completed_ratio, 1.0,
                )
            if not np.isfinite(completed_ratio).all():
                raise FloatingPointError("Non-finite HAPPO importance ratio")
            if correctness:
                new_distribution_logp = _correctness_distribution(self.actor[agent_id], buffer)
                commands = buffer.actions.reshape(-1).astype(np.int64)
                rows = np.arange(commands.size)
                independent_log_ratio = (new_distribution_logp[rows, commands]-
                                         old_distribution_logp[rows, commands]).reshape(factor.shape)
                active = buffer.active_masks[:-1] > 0.
                # Forced placeholders do not have a collection likelihood.
                # Marginalize their unused command as a unit factor instead.
                masked_log_ratio = np.where(active, independent_log_ratio, 0.)
                prefix_log_ratio += masked_log_ratio
                old_probability = np.exp(old_distribution_logp)
                new_probability = np.exp(new_distribution_logp)
                kl = np.sum(old_probability*(old_distribution_logp-new_distribution_logp), axis=-1)
                entropy = -np.sum(new_probability*new_distribution_logp, axis=-1)
                selected = active.reshape(-1)
                forced = completed_ratio[~active]
                retained_ratio = np.exp(independent_log_ratio[active])
                correctness_actor.update(controllable_rows=int(active.sum()),
                    forced_rows=int((~active).sum()),
                    final_sampled_ratio=_correctness_statistics(retained_ratio),
                    final_masked_ratio=_correctness_statistics(np.exp(masked_log_ratio)),
                    forced_factor_ratio_max_abs_error=(float(np.abs(forced-1.).max())
                                                       if forced.size else 0.),
                    final_policy_kl=(float(np.maximum(kl[selected], 0.).mean()) if selected.any() else None),
                    final_clip_fraction=(float(np.mean(
                        (retained_ratio < 1.-self.actor[agent_id].clip_param) |
                        (retained_ratio > 1.+self.actor[agent_id].clip_param)))
                        if retained_ratio.size else None),
                    final_entropy=(float(entropy[selected].mean()) if selected.any() else None))
                correctness_actors.append(correctness_actor)
            factor = factor * completed_ratio
            if not np.isfinite(factor).all():
                raise FloatingPointError("Non-finite accumulated HAPPO factor")
            actor_train_info["factor_max"] = float(factor.max())
            actor_train_info["factor_min"] = float(factor.min())
            actor_train_infos[agent_id] = actor_train_info

        # update critic
        # GAE and the shared actor advantage were computed with the rollout's
        # original ValueNorm statistics. Update them once only after those
        # signals have been consumed, then freeze the scale for all critic epochs.
        critic_train_info = self.critic.train(
            self.critic_buffer, self.value_normalizer, update_normalizer_once=True
        )

        if correctness:
            self.last_correctness_happo = dict(update=getattr(self, 'update_count', 0)+1,
                agent_order=[int(agent_id) for agent_id in agent_order],
                factor_oracle='float64 exp(sum of independently reconstructed masked categorical log ratios)',
                ratio_scope='final post-update / collection likelihood on controllable sampled commands',
                actors=correctness_actors)

        return actor_train_infos, critic_train_info
