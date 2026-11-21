"""End-to-end HAPPO evidence, convergence, provenance and plot consistency."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

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

from examples import plot_cascade_correctness as plots
from examples import train_cascade_parallel as parallel


def convergence_lines(figure):
    """Return convergence lines, excluding reference lines with scalar y."""
    axis = figure.axes[3]
    return [np.asarray(line.get_ydata(), dtype=float) for line in axis.lines
            if np.ndim(line.get_ydata()) == 1 and np.size(line.get_ydata()) > 1]


def critic_lines(figure):
    return [[np.asarray(line.get_ydata(), dtype=float) for line in axis.lines
             if np.ndim(line.get_ydata()) == 1 and np.size(line.get_ydata()) > 1]
            for axis in figure.axes[:3]]


def load_fixture():
    from tests.test_cascade_correctness_plots import fixture
    return fixture()


@unittest.skipIf(plt is None, 'Matplotlib is required for export-to-plot tests')
class ExportedEvidencePlotTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_transport_hashes_accept_posix_and_windows_source_keys(self):
        modules = ('examples/train_cascade_parallel.py',
                   'harl/envs/cascade_reservoir/correctness_diagnostics.py',
                   'examples/plot_cascade_correctness.py')
        for separator in ('/', '\\'):
            with self.subTest(separator=separator):
                hashes = {name.replace('/', separator):str(index)*64
                          for index, name in enumerate(modules)}
                selected = parallel._record_transport_hashes(hashes)
                self.assertEqual(tuple(selected), modules)

    def test_critic_histories_are_read_from_update_records(self):
        payload = load_fixture()
        run = payload['runs'][0]
        run['critic_quality'] = dict(available=False, channels=[])
        figures = plots.make_figures(payload)
        lines = critic_lines(figures['training_reliability'])
        self.assertTrue(all(lines), 'All Critic panels must contain update-record histories')
        expected = [record['critic_quality']['channels'][0]['critic_mc_post']['explained_variance']
                    for record in run['update_records']]
        fitted = next(line for line in figures['training_reliability'].axes[0].lines
                      if line.get_label().startswith('Mixed reward (WDD)'))
        np.testing.assert_allclose(fitted.get_ydata(), expected)
        self.assertNotIn('Unavailable: no Critic fit history',
                         ' '.join(text.get_text() for ax in figures['training_reliability'].axes
                                  for text in ax.texts))

    def test_team_gae_convergence_uses_system_return_not_legacy_return(self):
        payload = load_fixture()
        records = payload['runs'][0]['evaluation_records']
        for number, record in enumerate(records):
            record.update(evaluator_return=1000.+number,
                          system_evaluator_return=10.+number,
                          mixed_evaluator_returns=[100.+number+channel for channel in range(5)],
                          learning_evaluator_return=10.+number)
        figure = plots.make_figures(payload)['training_reliability']
        values = convergence_lines(figure)[0]
        np.testing.assert_allclose(values, [10., 11., 12., 13., 14., 15.])
        self.assertFalse(np.isclose(values, [1000., 1001., 1002., 1003., 1004., 1005.]).all())

    def test_mixed_gae_convergence_uses_mean_mixed_return(self):
        payload = load_fixture()
        payload['learning_objective'] = dict(advantage_source='individual_mixed',
            critic_reward_layout='mixed_by_reservoir', actor_value_channels=list(range(5)))
        payload['configuration']['learning_objective'] = payload['learning_objective']
        expected = []
        for number, record in enumerate(payload['runs'][0]['evaluation_records']):
            mixed = [100.+number+channel for channel in range(5)]
            record.update(evaluator_return=999., system_evaluator_return=777.,
                          mixed_evaluator_returns=mixed,
                          learning_evaluator_return=float(np.mean(mixed)),
                          learning_return_basis='individual_mixed')
            expected.append(np.mean(mixed))
        figure = plots.make_figures(payload)['training_reliability']
        np.testing.assert_allclose(convergence_lines(figure)[0], expected)

    def test_declared_objective_never_falls_back_to_legacy_evaluator_return(self):
        for source in ('shared_team', 'individual_mixed'):
            with self.subTest(source=source):
                payload = load_fixture()
                payload['learning_objective'] = dict(advantage_source=source)
                payload['configuration']['learning_objective'] = payload['learning_objective']
                for record in payload['runs'][0]['evaluation_records']:
                    record.pop('system_evaluator_return', None)
                    record.pop('mixed_evaluator_returns', None)
                    record.pop('learning_evaluator_return', None)
                with self.assertRaisesRegex(ValueError, 'Missing'):
                    plots.make_figures(payload)

    def test_feedback_values_and_manifest_bind_exported_data_to_rendered_pngs(self):
        payload = load_fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            evidence_path = directory/'evidence.json'
            evidence_path.write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
            exported = json.loads(evidence_path.read_text(encoding='utf-8'))
            figures = plots.make_figures(exported)
            expected_figures = plots.make_figures(payload)
            np.testing.assert_allclose(convergence_lines(figures['training_reliability'])[0],
                                       convergence_lines(expected_figures['training_reliability'])[0])
            for actual_axes, expected_axes in zip(critic_lines(figures['training_reliability']),
                                                  critic_lines(expected_figures['training_reliability'])):
                for actual, expected in zip(actual_axes, expected_axes):
                    np.testing.assert_allclose(actual, expected)
            targets = [directory/(name+'.png') for name in plots.FIGURE_KEYS]
            for key, path in zip(plots.FIGURE_KEYS, targets):
                figures[key].savefig(path)
            manifest = parallel.write_artifact_manifest(directory/'manifest.json',
                payload['provenance'], [evidence_path]+targets, root=directory)
            self.assertEqual(manifest['run_id'], payload['run_id'])
            self.assertEqual({row['path'] for row in manifest['artifacts']},
                             {'evidence.json'}|{path.name for path in targets})
            parallel.verify_artifact_manifest(directory/'manifest.json')
            targets[0].write_bytes(targets[0].read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch|size mismatch'):
                parallel.verify_artifact_manifest(directory/'manifest.json')

    def test_run_id_mismatch_and_existing_manifest_are_rejected(self):
        payload = load_fixture()
        payload['provenance']['run_id'] = 'd'*64
        with self.assertRaisesRegex(ValueError, 'run_id'):
            plots.make_figures(payload)
        payload = load_fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            target = directory/'evidence.json'
            target.write_text('source', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                parallel._write_json_atomic(target, {'value': 1})

    def test_run_manifest_names_are_exclusive_and_bound_to_the_output_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            self.assertEqual(parallel.run_artifact_manifest_path(None, directory/'curves'),
                             directory/'run_manifest.json')
            self.assertEqual(parallel.run_artifact_manifest_path(directory/'run.jsonl'),
                             directory/'run_manifest.json')
            self.assertNotEqual(parallel.run_artifact_manifest_path(directory/'happo.jsonl'),
                                parallel.run_artifact_manifest_path(directory/'mappo.jsonl'))
            provenance = load_fixture()['provenance']
            artifact = directory/'artifact.txt'
            artifact.write_text('content', encoding='utf-8')
            parallel.write_artifact_manifest(directory/'run_manifest.json', provenance, [artifact])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                parallel.write_artifact_manifest(directory/'run_manifest.json', provenance, [artifact])

    def test_correctness_save_verifies_renderer_manifest_without_overwriting_it(self):
        payload = load_fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            output = directory/'correctness'

            def render(command, **kwargs):
                self.assertTrue(command[1].endswith('plot_cascade_correctness.py'))
                self.assertEqual(plots.main(command[2:]), 0)

            with mock.patch.object(parallel, 'build_correctness_payload', return_value=payload), \
                    mock.patch.object(parallel.subprocess, 'run', side_effect=render):
                result = parallel.save_correctness_outputs([], payload['configuration'], [],
                                                           output, 'renderer.py')
            manifest = parallel.verify_artifact_manifest(output/'manifest.json')
            self.assertEqual(manifest['run_id'], payload['run_id'])
            self.assertEqual({row['path'] for row in manifest['artifacts']},
                             {'evidence.json'}|{name+'.png' for name in plots.FIGURE_KEYS})
            self.assertEqual(result['artifacts'], 9)

    def test_windows_source_hashes_and_run_manifest_authorize_a_valid_log(self):
        from tests.test_cascade_learning_objective_plot import log_rows
        metadata, episodes, update = log_rows('happo')
        platform_path = 'examples\\plot_cascade_parallel.py'
        provenance = dict(schema='cascade_run_provenance_v1', run_id='a'*64,
            source_code_hashes={platform_path:'d'*64},
            source_tree_sha256='b'*64, data_tree_sha256='c'*64)
        metadata.update(run_id=provenance['run_id'], provenance=provenance)
        completion = dict(record_type='run_complete', run_id=provenance['run_id'],
                          provenance=provenance)
        from examples import plot_cascade_parallel as legacy
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in
                [metadata, *episodes, update, completion]), encoding='utf-8')
            parallel.write_artifact_manifest(path.parent/'run_manifest.json', provenance, [path])
            loaded = legacy.load_episode_log(path)
            self.assertEqual(loaded['run_id'], provenance['run_id'])
            self.assertTrue(loaded['histories'])

    def test_completed_log_accepts_extended_provenance_and_remote_run_manifest(self):
        from tests.test_cascade_learning_objective_plot import log_rows
        metadata, episodes, update = log_rows('happo')
        provenance = dict(schema='cascade_run_provenance_v1', run_id='a'*64,
            source_code_hashes={'examples/plot_cascade_parallel.py':'d'*64},
            source_tree_sha256='b'*64, data_tree_sha256='c'*64)
        metadata.update(run_id=provenance['run_id'], provenance=provenance)
        from examples import plot_cascade_parallel as legacy
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path, figures = directory/'logs'/'run.jsonl', directory/'figures'
            path.parent.mkdir(); figures.mkdir()
            manifest_path = figures/'run_manifest.json'
            completed = dict(provenance, artifact_manifest=str(manifest_path),
                artifact_inventory='written after all outputs were closed',
                continuation_provenance='base source and data identity')
            completion = dict(record_type='run_complete', run_id=provenance['run_id'],
                              provenance=completed)
            path.write_text(''.join(json.dumps(row)+'\n' for row in
                [metadata, *episodes, update, completion]), encoding='utf-8')
            parallel.write_artifact_manifest(manifest_path, provenance, [path])
            loaded = legacy.load_episode_log(path)
            self.assertEqual(loaded['run_id'], provenance['run_id'])
            path.write_text(path.read_text(encoding='utf-8')+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch|size mismatch'):
                legacy.load_episode_log(path)

    def test_learning_log_loader_rejects_foreign_completion_and_stale_log(self):
        from tests.test_cascade_learning_objective_plot import log_rows
        metadata, episodes, update = log_rows('happo')
        provenance = dict(schema='cascade_run_provenance_v1', run_id='a'*64)
        metadata.update(run_id=provenance['run_id'], provenance=provenance)
        foreign = dict(record_type='run_complete', run_id='b'*64, provenance=provenance)
        from examples import plot_cascade_parallel as legacy
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in [metadata, *episodes, update, foreign]),
                            encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'run_complete provenance'):
                legacy.load_episode_log(path)

            # A manifest for a different run must not authorize this log.
            good = dict(record_type='run_complete', run_id=provenance['run_id'],
                        provenance=provenance)
            path.write_text(''.join(json.dumps(row)+'\n' for row in [metadata, *episodes, update, good]),
                            encoding='utf-8')
            parallel.write_artifact_manifest(path.with_name('run_manifest.json'),
                dict(provenance, source_tree_sha256='b'*64, data_tree_sha256='c'*64), [path])
            path.write_text(path.read_text(encoding='utf-8')+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch|size mismatch'):
                legacy.load_episode_log(path)


if __name__ == '__main__':
    unittest.main()
