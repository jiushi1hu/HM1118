"""The feasibility-mode switch changes propagation only, never constraints."""
import unittest

import numpy as np
import yaml

from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.dispatch_kernel import (
    FEASIBILITY_MODES, InfeasibleCascade, _backward, dispatch_release,
    validate_feasibility_mode)
from harl.envs.cascade_reservoir.feasibility_ablation import FeasibilityRecorder
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


class FeasibilityModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = yaml.safe_load(
            (ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text(encoding='utf-8'))
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical)
        limits = cls.data.load_daily_level_limits()
        # A historical action date with its real next-state storage envelope.
        index = cls.data.indices(1998, 1998)[0]+180
        cls.external = cls.data.external[index:index+1].copy()
        cls.lower = limits.storage_lower[index:index+1].copy()
        cls.upper = limits.storage_upper[index:index+1].copy()
        cls.storage = ((cls.lower+cls.upper)/2.)

    def test_modes_are_validated(self):
        self.assertEqual(set(FEASIBILITY_MODES), {'backward', 'forward_local'})
        for mode in FEASIBILITY_MODES:
            self.assertEqual(validate_feasibility_mode(mode), mode)
        for bad in ('forward', None, 1, 'BACKWARD'):
            with self.assertRaises(ValueError):
                validate_feasibility_mode(bad)

    def test_both_modes_dispatch_inside_the_available_interval(self):
        actions = np.full((1, 5), .7)
        for mode in FEASIBILITY_MODES:
            out = dispatch_release(self.storage, self.external, actions, self.lower,
                self.upper, self.data.release_cap, 86400., feasibility_mode=mode)
            self.assertEqual(out['feasibility_mode'], mode)
            self.assertFalse(out['empty_interval'].any())
            self.assertFalse(out['forced_interval'].any())
            self.assertTrue((out['release'] <= self.data.release_cap+1e-8).all())
            self.assertGreaterEqual(float(out['release'].min()), -1e-8)

    def test_backward_propagation_only_narrows_and_does_bind(self):
        """Compatibility propagation may only narrow the interval."""
        rng = np.random.default_rng(7)
        limits = self.data.load_daily_level_limits()
        index = rng.integers(0, len(self.data.dates), size=4000)
        storage = self.data.vmin+rng.random((4000, 5))*(self.data.vmax-self.data.vmin)
        lower, upper = limits.storage_lower[index].copy(), limits.storage_upper[index].copy()
        external = self.data.external[index].copy()
        actions = np.full((4000, 5), .5)
        _, _, conflict = _backward(storage, external, lower, upper,
            np.broadcast_to(self.data.release_cap, (4000, 5)), 86400.)
        consistent = ~conflict.any(axis=1)
        self.assertGreater(int(consistent.sum()), 100)
        storage, lower, upper = storage[consistent], lower[consistent], upper[consistent]
        external, actions = external[consistent], actions[consistent]
        self.assertGreater(storage.shape[0], 100)
        propagated_lower, propagated_upper, _ = _backward(storage, external, lower,
            upper, np.broadcast_to(self.data.release_cap, storage.shape), 86400.)
        local_low = external+(storage-upper)/86400.
        local_high = external+(storage-lower)/86400.
        forward_low = np.maximum(0., local_low)
        forward_high = np.minimum(self.data.release_cap, local_high)
        back_low = np.maximum(propagated_lower, local_low)
        back_high = np.minimum(propagated_upper, local_high)
        self.assertTrue((back_low >= forward_low-1e-9).all())
        self.assertTrue((back_high <= forward_high+1e-9).all())
        narrowed = ((back_high < forward_high-1e-6) |
                    (back_low > forward_low+1e-6))
        self.assertGreater(int(narrowed.any(axis=1).sum()), 0)

    def test_forward_mode_raises_instead_of_relaxing_an_empty_interval(self):
        storage = np.broadcast_to(self.data.vmax, (1, 5)).copy()
        external = np.zeros((1, 5))
        # A start at the global minimum with end bounds far above it requires an
        # inflow larger than any admissible release, so no interval exists.
        storage = np.broadcast_to(self.data.vmin, (1, 5)).copy()
        lower = np.broadcast_to(self.data.vmax - 1e3, (1, 5)).copy()
        upper = np.broadcast_to(self.data.vmax, (1, 5)).copy()
        with self.assertRaises(InfeasibleCascade):
            dispatch_release(storage, external, np.full((1, 5), .5), lower, upper,
                self.data.release_cap, 86400., feasibility_mode='forward_local')

    def test_physics_honours_the_configured_mode(self):
        for mode in FEASIBILITY_MODES:
            physics = ReservoirPhysics(self.data, dict(self.physical, feasibility_mode=mode))
            self.assertEqual(physics.feasibility_mode, mode)
            result = physics.transition(
                self.data.vmin+(self.data.vmax-self.data.vmin)*.5,
                self.data.external[0], np.full(5, .5))
            self.assertEqual(result['feasibility_mode'], mode)
            self.assertEqual(result['forced_actor_days'], 0)
            self.assertEqual(result['empty_interval_days'], 0)
            self.assertEqual(result['release_interval_fraction'].shape, (5,))

    def test_recorder_rejects_an_executed_empty_interval(self):
        recorder = FeasibilityRecorder('forward_local')
        transition = dict(date='1998-01-01', forced_interval=np.zeros(5, bool),
            empty_interval=np.ones(5, bool), release=np.zeros(5),
            release_interval_fraction=np.zeros(5), levels_start=np.ones(5)*100.,
            system_residual=0.)
        with self.assertRaises(ValueError):
            recorder.capture(transition)


if __name__ == '__main__':
    unittest.main(verbosity=2)
