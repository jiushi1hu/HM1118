"""Calendar and artist checks for five-year reservoir operation trajectories."""
import copy
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import plot_cascade_correctness as plots
from test_cascade_correctness_plots import fixture, plt, visible_text


def operation_fixture(start_year=2000):
    """Five complete calendar years, with different daily start and end levels."""
    payload = fixture()
    start, stop = date(start_year, 1, 1), date(start_year+5, 1, 1)
    dates = [start+timedelta(days=day) for day in range((stop-start).days)]
    count = len(dates)
    values = np.arange(count, dtype=float)[:, None]+100.*np.arange(5)[None, :]
    trajectory = dict(seed=1, source_years=list(range(start_year, start_year+5)),
        reservoir_ids=list(plots.IDS), steps=count, expected_steps=count,
        status='ok', scenario_id='median_'+str(start_year), policy_update=2,
        physical=dict(initial_level_m=(values[0]-10.).tolist()),
        series=dict(date=[day.isoformat() for day in dates],
            levels_start=(values-10.).tolist(), levels_end=values.tolist(),
            release=(1000.+2.*values).tolist()))
    payload['runs'][0]['operation_trajectory'] = trajectory
    return payload, dates, values


def artists(figure, gid):
    return [artist for ax in figure.axes for artist in
            list(ax.lines)+list(ax.collections) if artist.get_gid() == gid]


@unittest.skipIf(plt is None, 'Matplotlib is available in the plotting interpreter')
class CascadeOperationFigureTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_exact_five_year_axes_and_actual_december_31_end_levels(self):
        payload, dates, values = operation_fixture()
        original = copy.deepcopy(payload)
        figure = plots.make_operation_figure(payload)
        self.assertEqual(len(figure.axes), 10)  # Five level axes and five flow axes.
        for ax in figure.axes:
            np.testing.assert_allclose(ax.get_xlim(), [0., 5.], rtol=0., atol=1e-12)
        endpoints = [i for i, day in enumerate(dates) if (day.month, day.day) == (12, 31)]
        marks = artists(figure, 'december31_levels')
        self.assertEqual(len(marks), 5)
        for reservoir, scatter in enumerate(marks):
            offsets = np.asarray(scatter.get_offsets(), dtype=float)
            np.testing.assert_allclose(offsets[:, 0], np.arange(1., 6.))
            np.testing.assert_allclose(offsets[:, 1], values[endpoints, reservoir])
            self.assertEqual(len(scatter.get_facecolors()), 1)
            self.assertGreater(scatter.get_facecolors()[0, 3], 0.)
        self.assertEqual(payload, original)

    def test_leap_dates_are_preserved_and_release_matches_executed_daily_values(self):
        payload, dates, values = operation_fixture()
        figure = plots.make_operation_figure(payload)
        level = artists(figure, 'water_level')[0]
        release = artists(figure, 'release_discharge')[0]
        leap_index = dates.index(date(2000, 2, 29))
        following_year = dates.index(date(2001, 1, 1))
        # Levels are end-of-day states. Daily discharges are placed at day centers.
        self.assertAlmostEqual(level.get_xdata()[leap_index+1], 60./366.)
        self.assertAlmostEqual(level.get_xdata()[following_year+1], 1.+1./365.)
        self.assertAlmostEqual(release.get_xdata()[leap_index], 59.5/366.)
        np.testing.assert_array_equal(level.get_ydata()[1:], values[:, 0])
        np.testing.assert_array_equal(release.get_ydata(), 1000.+2.*values[:, 0])
        self.assertEqual(len(release.get_xdata()), 1827)

    def test_solid_levels_dashed_releases_and_reversed_flow_axes(self):
        payload, _, _ = operation_fixture()
        figure = plots.make_operation_figure(payload)
        levels = artists(figure, 'water_level')
        releases = artists(figure, 'release_discharge')
        self.assertEqual(len(levels), 5)
        self.assertEqual(len(releases), 5)
        for level, release in zip(levels, releases):
            self.assertEqual(level.get_linestyle(), '-')
            self.assertEqual(release.get_linestyle(), '--')
            self.assertFalse(level.axes.yaxis_inverted())
            self.assertTrue(release.axes.yaxis_inverted())
            self.assertIsNot(level.axes, release.axes)
            self.assertTrue(level.axes.get_shared_x_axes().joined(level.axes, release.axes))
        figure.canvas.draw()

    def test_each_seed_has_its_own_five_reservoir_panels(self):
        payload, _, _ = operation_fixture()
        second = copy.deepcopy(payload['runs'][0])
        second['seed'] = second['trajectory']['seed'] = second['operation_trajectory']['seed'] = 7
        second['operation_trajectory']['series']['levels_end'] = (
            np.asarray(second['operation_trajectory']['series']['levels_end'])+50.).tolist()
        payload['runs'].append(second)
        figure = plots.make_operation_figure(payload)
        self.assertEqual(len(figure.axes), 20)
        self.assertEqual(len(artists(figure, 'water_level')), 10)
        self.assertEqual(len(artists(figure, 'release_discharge')), 10)
        text = visible_text(figure)
        self.assertIn('seed 1', text)
        self.assertIn('seed 7', text)
        for reservoir in plots.IDS:
            self.assertIn(reservoir.lower(), text)

    def test_short_legacy_evidence_does_not_invent_five_years_or_year_end_markers(self):
        payload = fixture()
        figure = plots.make_operation_figure(payload)
        level = artists(figure, 'water_level')[0]
        release = artists(figure, 'release_discharge')[0]
        self.assertEqual(len(level.get_xdata()), 3)  # Initial state plus two executed days.
        self.assertEqual(len(release.get_xdata()), 2)
        self.assertLess(float(np.max(level.get_xdata())), 1.)
        self.assertEqual(sum(len(scatter.get_offsets()) for scatter in
                             artists(figure, 'december31_levels')), 0)
        np.testing.assert_allclose(level.axes.get_xlim(), [0., 5.])
        self.assertTrue(any(word in visible_text(figure) for word in
                            ('partial', 'incomplete', 'recorded', 'unavailable')))

    def test_partial_last_year_only_marks_recorded_december_31_dates(self):
        payload, dates, values = operation_fixture()
        trajectory = payload['runs'][0]['operation_trajectory']
        count = dates.index(date(2004, 12, 30))+1
        trajectory['steps'] = count
        trajectory['status'] = 'failed'
        for key in trajectory['series']:
            trajectory['series'][key] = trajectory['series'][key][:count]
        figure = plots.make_operation_figure(payload)
        marks = artists(figure, 'december31_levels')
        self.assertEqual(len(marks), 5)
        for scatter in marks:
            np.testing.assert_array_equal(scatter.get_offsets()[:, 0], [1., 2., 3., 4.])
        self.assertEqual(len(artists(figure, 'release_discharge')[0].get_xdata()), count)
        self.assertIn('failed', visible_text(figure))

    def test_unavailable_operation_capture_does_not_fall_back_to_other_trajectory(self):
        payload = fixture()
        payload['runs'][0]['operation_trajectory'] = dict(seed=1, steps=0,
            source_years=list(range(2000, 2005)), expected_steps=1827,
            reservoir_ids=list(plots.IDS), status='unavailable', series={})
        figure = plots.make_operation_figure(payload)
        self.assertEqual(len(artists(figure, 'water_level')), 0)
        self.assertEqual(len(artists(figure, 'release_discharge')), 0)
        self.assertIn('unavailable', visible_text(figure))


if __name__ == '__main__':
    unittest.main()
