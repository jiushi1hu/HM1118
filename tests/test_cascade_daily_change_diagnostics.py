"""Exact checks for automatic per-seed level and release change diagnostics."""
import csv
from contextlib import redirect_stdout
from datetime import date, timedelta
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import train_cascade_parallel as parallel
from examples import train_cascade as cli
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir import daily_change_diagnostics as diagnostics
from harl.envs.cascade_reservoir.level_ramp import CONFLICT_FIELDS
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from examples import plot_daily_change_diagnostics as plotting
except ImportError:  # The plotting interpreter may be separate from the test interpreter.
    plt = plotting = None


def trajectory(seed=3, days=None):
    if days is None:
        days = [date(2019, 12, 31), date(2020, 1, 1), date(2020, 2, 28),
                date(2020, 2, 29), date(2020, 3, 1), date(2001, 1, 1)]
    levels = np.asarray([[100., 200., 300., 400., 500.],
        [101., 198., 304., 397., 503.],
        [103., 197., 307., 399., 506.],
        [106., 195., 311., 402., 510.],
        [110., 192., 316., 406., 515.],
        [90., 190., 300., 410., 520.]], dtype=float)
    releases = np.asarray([[10., 20., 30., 40., 50.],
        [13., 18., 34., 37., 53.],
        [16., 17., 37., 39., 56.],
        [20., 15., 41., 42., 60.],
        [25., 12., 46., 46., 65.],
        [5., 50., 100., 150., 200.]], dtype=float)
    rise_limits = np.full((len(days), 5), 100.)
    fall_limits = np.full((len(days), 5), 100.)
    fall_limits[:, 0] = 10.
    return dict(schema_version=2, seed=seed, steps=len(days),
        dates=[day.isoformat() for day in days],
        levels_end=levels.tolist(), release=releases.tolist(),
        ramp_rise_limit_m_per_day=rise_limits.tolist(),
        ramp_fall_limit_m_per_day=fall_limits.tolist()), levels, releases


def read_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class DailyChangeDataTest(unittest.TestCase):
    def test_exact_differences_missing_first_step_and_cross_year_join(self):
        payload, levels, releases = trajectory()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'happo_seed3_daily_changes.csv'
            diagnostics.write_daily_change_csv(path, payload)
            rows = read_rows(path)
        self.assertEqual(tuple(rows[0]), diagnostics.DAILY_CHANGE_FIELDS)
        self.assertEqual(len(rows), len(levels)*len(diagnostics.IDS))
        for reservoir in range(5):
            self.assertEqual(rows[reservoir]['reservoir'], diagnostics.IDS[reservoir])
            self.assertEqual(rows[reservoir]['delta_level_m'], '')
            self.assertEqual(rows[reservoir]['abs_delta_level_m'], '')
            self.assertEqual(rows[reservoir]['delta_release_m3s'], '')
            self.assertEqual(rows[reservoir]['within_ramp_limit'], '')
            for day in range(1, len(levels)):
                row = rows[day*5+reservoir]
                self.assertAlmostEqual(float(row['delta_level_m']),
                    levels[day, reservoir]-levels[day-1, reservoir], places=12)
                self.assertAlmostEqual(float(row['delta_release_m3s']),
                    releases[day, reservoir]-releases[day-1, reservoir], places=12)
                self.assertEqual(float(row['rise_limit_m_per_day']), 100.)
                self.assertEqual(float(row['fall_limit_m_per_day']),
                                 10. if reservoir == 0 else 100.)
                self.assertEqual(row['within_ramp_limit'],
                    '0' if day == 5 and reservoir == 0 else '1')
        self.assertEqual(rows[5*5]['within_ramp_limit'], '0')
        join = rows[5*5]
        self.assertEqual(join['date'], '2001-01-01')
        self.assertEqual(join['year'], '2001')
        self.assertEqual(join['day_of_year'], '1')
        self.assertAlmostEqual(float(join['delta_level_m']), levels[5, 0]-levels[4, 0])
        leap = rows[3*5]
        self.assertEqual((leap['date'], leap['day_of_year']), ('2020-02-29', '60'))

    def test_statistics_use_all_finite_absolute_changes_per_reservoir(self):
        payload, levels, _ = trajectory()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'statistics.csv'
            diagnostics.write_level_change_statistics_csv(path, payload['seed'], levels)
            rows = read_rows(path)
        self.assertEqual(len(rows), 5)
        for index, row in enumerate(rows):
            values = np.abs(np.diff(levels[:, index]))
            self.assertEqual(row['reservoir'], diagnostics.IDS[index])
            self.assertEqual(int(row['sample_count']), len(values))
            self.assertAlmostEqual(float(row['mean_abs_delta_level_m']), np.mean(values))
            self.assertAlmostEqual(float(row['p95_abs_delta_level_m']), np.percentile(values, 95))
            self.assertAlmostEqual(float(row['max_abs_delta_level_m']), np.max(values))

    def test_invalid_first_change_and_missing_later_value_are_rejected(self):
        payload, _, _ = trajectory()
        payload['dates'] = payload['dates'][:-1]
        payload['levels_end'] = payload['levels_end'][:-1]
        payload['release'] = payload['release'][:-1]
        payload['ramp_rise_limit_m_per_day'] = payload['ramp_rise_limit_m_per_day'][:-1]
        payload['ramp_fall_limit_m_per_day'] = payload['ramp_fall_limit_m_per_day'][:-1]
        payload['steps'] -= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'changes.csv'
            diagnostics.write_daily_change_csv(path, payload)
            rows = read_rows(path)
        rows[0]['delta_level_m'] = '0'
        with tempfile.TemporaryDirectory() as directory:
            malformed = Path(directory)/'malformed.csv'
            with malformed.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=diagnostics.DAILY_CHANGE_FIELDS)
                writer.writeheader(); writer.writerows(rows)
            if plotting is not None:
                with self.assertRaisesRegex(ValueError, 'first daily difference'):
                    plotting.load_daily_change_csv(malformed)


class DailyChangeOutputTest(unittest.TestCase):
    def test_three_seeds_reserve_four_figures_and_two_csvs_each(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = parallel.daily_change_paths(None, Path(directory)/'run'/'summary.jsonl',
                [1, 2, 3], 'happo')
            self.assertEqual(set(paths), {1, 2, 3})
            for seed, files in paths.items():
                self.assertEqual(len(files), 7)
                self.assertTrue(all(path.parent == Path(directory)/'run' for path in files.values()))
                self.assertIn('happo_seed{}'.format(seed), files['csv'].name)
                self.assertEqual([name for name in files if name not in
                                  ('csv', 'statistics_csv', 'ramp_conflicts_csv')],
                    ['daily_level_change', 'level_change_heatmap',
                     'level_change_statistics', 'daily_release_change'])

    def test_orchestration_writes_csv_before_dispatch_and_keeps_no_plot_semantics(self):
        runners, expected = [], {}
        with tempfile.TemporaryDirectory() as directory:
            paths = parallel.daily_change_paths(Path(directory)/'run'/'curves.png', None,
                [1, 2, 3], 'happo')
            for seed in (1, 2, 3):
                payload, _, _ = trajectory(seed=seed)
                runners.append(SimpleNamespace(seed=seed, final_evaluation_daily_series=payload))
            with mock.patch.object(parallel.subprocess, 'run') as render:
                still_saved = parallel.save_daily_change_outputs(runners, paths, '/plot/python')
            self.assertEqual(render.call_count, 3)
            self.assertEqual(set(still_saved), {'1', '2', '3'})
            for call, seed in zip(render.call_args_list, (1, 2, 3)):
                command = call.args[0]
                self.assertEqual(command[:2], ['/plot/python', str(
                    ROOT/'examples/plot_daily_change_diagnostics.py')])
                self.assertIn('--input', command)
                self.assertTrue(paths[seed]['csv'].is_file())
                self.assertTrue(paths[seed]['statistics_csv'].is_file())


@unittest.skipIf(plotting is None, 'Matplotlib is available in the plotting interpreter')
class DailyChangeFigureTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_four_figures_per_seed_with_five_reservoir_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for seed in (1, 2, 3):
                payload, _, _ = trajectory(seed=seed)
                csv_path = directory/('happo_seed{}_daily_changes.csv'.format(seed))
                diagnostics.write_daily_change_csv(csv_path, payload)
                files = plotting.save_daily_change_figures(
                    csv_path, directory/'happo_seed{}'.format(seed))
                self.assertEqual(set(files), set(diagnostics.FIGURE_SUFFIXES))
                for path in files.values():
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 1000)
                series = plotting.load_daily_change_csv(csv_path)
                self.assertTrue(series['has_ramp_limits'])
                figures = [plotting.make_daily_level_change_figure(series),
                           plotting.make_level_change_heatmap_figure(series),
                           plotting.make_level_change_statistics_figure(series),
                           plotting.make_daily_release_change_figure(series)]
                for figure in figures[:1]+figures[3:]:
                    self.assertEqual(len(figure.axes), 5)
                self.assertTrue(all(any(line.get_label() == 'rise limit'
                                         for line in ax.lines) for ax in figures[0].axes))
                self.assertTrue(any(line.get_label() == 'outside hard ramp limit'
                                    for line in figures[0].axes[0].collections
                                    if hasattr(line, 'get_label')))
                self.assertEqual(sum(any(ax.images for ax in figure.axes)
                                     for figure in figures[1:2]), 1)
                self.assertEqual(sum(bool(ax.images) for ax in figures[1].axes), 5)
                self.assertEqual(len(figures[2].axes), 1)
                for figure in figures:
                    plt.close(figure)


@unittest.skipIf(plotting is None, 'Matplotlib is available in the plotting interpreter')
class AutomaticDailyChangeWorkflowTest(unittest.TestCase):
    def test_real_final_evaluation_automatically_writes_csvs_and_four_pngs(self):
        config, physical = cli.load_configuration()
        config['train'].update(episode_length=3, n_rollout_threads=2)
        config['model'].update(hidden_sizes=[8, 8], action_bins=5)
        config['algo'].update(ppo_epoch=1, critic_epoch=1)
        config['experiment'].update(episodes=2, seeds=[1], diagnostics=False,
            trajectory_diagnostics=False, save_checkpoints=False, checkpoint_interval=0)
        data = CascadeData(ROOT/'data/cascade_reservoir_new', physical, config['experiment'])
        windows = select_periodic_windows(data, episode_days=3)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run_dir = Path(directory)/'happo_run'
            args = SimpleNamespace(episode_log=run_dir/'summary.jsonl', plot_output=None,
                no_plot=True, skip_final_evaluation=False, check_data_only=False,
                plot_python=sys.executable, correctness_output=None)
            with mock.patch.object(parallel, 'PERIODIC_EVALUATION_DAYS', 3), \
                    mock.patch.object(parallel, 'select_periodic_windows', return_value=windows):
                result = parallel.run_parallel_experiment(data, physical, config, args)
            expected = set(parallel.daily_change_paths(None, args.episode_log, [1], 'happo')[1].values())
            self.assertEqual({path for path in run_dir.iterdir() if path in expected}, expected)
            for path in expected:
                self.assertTrue(path.is_file())
                if path.suffix == '.png':
                    self.assertGreater(path.stat().st_size, 1000)
            rows = read_rows(run_dir/'happo_seed1_daily_changes.csv')
            self.assertEqual(len(rows), 3*5)
            self.assertTrue(all(rows[index]['delta_level_m'] == '' for index in range(5)))
            self.assertFalse(any(rows[index]['delta_level_m'] == '' for index in range(5, len(rows))))
            conflict_rows = read_rows(run_dir/'happo_seed1_ramp_conflicts.csv')
            self.assertEqual(conflict_rows, [])
            with (run_dir/'happo_seed1_ramp_conflicts.csv').open(
                    encoding='utf-8', newline='') as stream:
                self.assertEqual(tuple(csv.DictReader(stream).fieldnames),
                                 CONFLICT_FIELDS)
            completed = [json.loads(line) for line in (run_dir/'summary.jsonl').read_text().splitlines()]
            run_complete = [row for row in completed if row.get('record_type') == 'run_complete'][0]
            self.assertTrue(run_complete['daily_change_diagnostics']['enabled'])
            self.assertEqual(result['daily_change_diagnostics']['files'], 
                             {str(seed):{name:str(path.resolve()) for name,path in files.items()}
                              for seed,files in parallel.daily_change_paths(
                                  None, args.episode_log, [1], 'happo').items()})


if __name__ == '__main__':
    unittest.main()
