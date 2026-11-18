"""Bounded, observational checks of one selected executed reservoir episode.

Raw transition values are retained alongside independently calculated balances
and rewards. Nothing is repaired, sampled, or fed back into optimization. These
checks establish numerical consistency of the supplied model, not calibration
of its hydraulic assumptions or horizon-wide dispatch optimality.
"""
import copy
import numpy as np

from harl.envs.cascade_reservoir.data_loader import IDS
from harl.envs.cascade_reservoir.storage_accounting import storage_change_summary


SCHEMA_VERSION = 2
BALANCE_RELATIVE_TOLERANCE = 64.*np.finfo(np.float64).eps
REWARD_ABSOLUTE_TOLERANCE = 1e-10
REWARD_RELATIVE_TOLERANCE = 1e-10

_VECTOR_FIELDS = (
    'storage_start', 'storage_end', 'levels_start', 'levels_end', 'external',
    'inflow', 'release', 'low', 'high', 'controls', 'power', 'energy', 'turbine',
    'spill', 'net_head_m', 'local_generation_rewards', 'local_reward_penalties',
    'local_rewards', 'mixed_rewards',
)
_SCALAR_REWARDS = ('reward', 'generation_reward', 'reward_penalty', 'global_reward', 'system_reward')
_VECTOR_REWARDS = ('local_generation_rewards', 'local_reward_penalties', 'local_rewards', 'mixed_rewards')
_OPTIONAL_VECTORS = (
    'storage_lower_m3', 'storage_upper_m3', 'next_state_level_lower_m',
    'next_state_level_upper_m', 'sampled_actions', 'actions', 'normalized_actions',
    'physical_low_m3s', 'physical_high_m3s',
)


def _vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (5,) or not np.isfinite(value).all():
        raise ValueError(name+' must be a finite five-vector')
    return value.copy()


def _scalar(value, name):
    value = np.asarray(value)
    if value.shape != () or not np.isfinite(value):
        raise ValueError(name+' must be a finite scalar')
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


def _json(value):
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tolerance(*terms):
    scale = sum(np.abs(np.asarray(term, dtype=np.longdouble)) for term in terms)
    return np.asarray(BALANCE_RELATIVE_TOLERANCE*np.maximum(scale, 1.), dtype=np.float64)


class CorrectnessRecorder:
    """Keep at most ``max_steps`` daily rows for one selected episode.

    ``begin_episode`` may replace an earlier episode, enabling a caller to keep
    only its selected evaluation or collection episode. ``finish`` returns detached JSON values.
    All violations are recorded; only malformed/nonfinite inputs raise.
    """

    def __init__(self, data, physical, seed, max_steps=1825):
        self.data = data
        self.seed = _integer(seed, 'seed')
        self.max_steps = _integer(max_steps, 'max_steps', 1)
        self.dt = _scalar(physical['timestep_seconds'], 'timestep_seconds')
        if self.dt <= 0.:
            raise ValueError('timestep_seconds must be positive')
        self.config = dict(physical)
        self.power_cap = _vector(data.power_cap, 'installed power capacity')
        self.release_cap = _vector(data.release_cap, 'outlet capacity')
        self.turbine_cap = _vector(data.turbine_cap, 'turbine capacity')
        self.vmin, self.vmax = _vector(data.vmin, 'minimum storage'), _vector(data.vmax, 'maximum storage')
        if ((self.power_cap <= 0.).any() or (self.release_cap <= 0.).any() or
                (self.turbine_cap <= 0.).any() or (self.vmin >= self.vmax).any()):
            raise ValueError('Capacities and storage spans must be positive')
        settings = physical.get('reward_penalties', {})
        self.penalties_enabled = settings.get('enabled', False)
        if not isinstance(self.penalties_enabled, bool):
            raise ValueError('Reward enabled flag must be a boolean')
        self.weight = _scalar(settings.get('ecological_flow_weight', 1.), 'ecological_flow_weight')
        if self.weight < 0.:
            raise ValueError('Ecological reward weight must be nonnegative')
        requirements = getattr(data, 'service_requirements', None)
        if requirements is None and self.penalties_enabled:
            raise ValueError('Independent ecological reward checks require loaded service requirements')
        self.ecology = (np.zeros(5) if requirements is None else
                        _vector(requirements.ecological_flow_m3s, 'ecological requirement'))
        if (self.ecology < 0.).any():
            raise ValueError('Ecological requirements must be nonnegative')
        self.services_available = requirements is not None
        self.started = False
        self.closed = False

    def _levels(self, storage):
        # Independent interpolation of the supplied anchors; no simulator level
        # helper is called. Outside-domain storage is separately flagged.
        return np.asarray([np.interp(storage[i], self.data.anchors[name][:, 1],
                                     self.data.anchors[name][:, 0])
                           for i, name in enumerate(IDS)])

    def begin_episode(self, episode, source_years):
        self.episode = _integer(episode, 'episode', 1)
        self.source_years = [_integer(year, 'source year', 1) for year in source_years]
        if not self.source_years:
            raise ValueError('An episode requires source years')
        initial = _vector(self.data.initial_storage(self.source_years[0]), 'initial storage')
        self.physical = dict(timestep_seconds=self.dt,
            head_model=self.config.get('head_model', 'fixed'),
            efficiency=float(self.config['efficiency']),
            power_cap_mw=self.power_cap.copy(), release_cap_m3s=self.release_cap.copy(),
            turbine_cap_m3s=self.turbine_cap.copy(), storage_min_m3=self.vmin.copy(),
            storage_max_m3=self.vmax.copy(), ecological_flow_m3s=self.ecology.copy(),
            service_requirements_available=self.services_available,
            reward_penalties_enabled=self.penalties_enabled, ecological_flow_weight=self.weight,
            local_global_alpha=.5, end_storage_policy='unconstrained_within_daily_operating_bounds',
            initial_storage_m3=initial.copy(), initial_level_m=self._levels(initial),
            balance_accumulator_dtype=str(np.dtype(np.longdouble)),
            balance_relative_tolerance=BALANCE_RELATIVE_TOLERANCE,
            reward_absolute_tolerance=REWARD_ABSOLUTE_TOLERANCE,
            reward_relative_tolerance=REWARD_RELATIVE_TOLERANCE)
        self.physical['bound_tolerance_definition'] = (
            'storage: max(1 m3, 1e-10*abs(start storage)); release: at least 1e-8 m3/s; '
            'installed power: at least 1e-7 MW; independent conservation uses separate 64-epsilon tolerances')
        self.rows, self.checks, self.differences, self.flags = [], [], [], []
        self.started, self.closed = True, False
        self.result = None

    def _rewards(self, power, release):
        shortfall = np.maximum(self.ecology-release, 0.)
        shortfall[shortfall <= 1e-10*np.maximum(self.ecology, 1.)] = 0.
        relative = np.zeros(5)
        for i in range(5):
            if self.ecology[i] > 0.:
                relative[i] = shortfall[i]/self.ecology[i]
        costs = (self.weight if self.penalties_enabled else 0.)*relative
        local_generation = np.asarray([power[i]/self.power_cap[i] for i in range(5)])
        generation = float(sum(power)/sum(self.power_cap))
        local = local_generation-costs
        penalty = float(sum(costs)/max(int(np.count_nonzero(self.ecology > 0.)), 1))
        return dict(generation_reward=generation, global_reward=generation,
            reward_penalty=penalty, reward=generation-penalty,
            local_generation_rewards=local_generation, local_reward_penalties=costs,
            local_rewards=local, mixed_rewards=.5*local+.5*generation,
            system_reward=float(.5*sum(local)+.5*generation),
            ecological_shortfall_m3s=shortfall, normalized_ecological_shortfall=relative)

    def capture(self, transition):
        if not self.started or self.closed:
            raise RuntimeError('Begin an episode before capturing its transitions')
        if len(self.rows) >= self.max_steps:
            raise ValueError('Selected correctness episode exceeds max_steps')
        row = {name: _vector(transition[name], name) for name in _VECTOR_FIELDS}
        row['date'] = str(transition['date'])
        for name in _SCALAR_REWARDS:
            default = transition['reward'] if name == 'generation_reward' else 0.
            row[name] = _scalar(transition.get(name, default), name)
        for name in _OPTIONAL_VECTORS:
            row[name] = None if name not in transition else _vector(transition[name], name)
        if not self.rows:
            self.physical['initial_storage_m3'] = row['storage_start'].copy()
            self.physical['initial_level_m'] = self._levels(row['storage_start'])
        start, end = [row[name].astype(np.longdouble) for name in ('storage_start', 'storage_end')]
        inflow, release, external = [row[name].astype(np.longdouble) for name in ('inflow', 'release', 'external')]
        dt = np.longdouble(self.dt)
        residual = np.asarray(end-start-dt*(inflow-release), dtype=np.float64)
        tolerance = _tolerance(start, end, dt*inflow, dt*release)
        cascade_residual = float(sum(end)-sum(start)-dt*(sum(external)-release[-1]))
        cascade_tolerance = float(_tolerance(sum(np.abs(start)), sum(np.abs(end)),
                                             dt*sum(np.abs(external)), dt*abs(release[-1])))
        routed = external.copy()
        routed[1:] += release[:-1]
        routing_residual = np.asarray(inflow-routed, dtype=np.float64)
        routing_tolerance = _tolerance(inflow, routed)
        continuity = (np.zeros(5) if not self.rows else
                      np.asarray(start-np.asarray(self.rows[-1]['storage_end'], dtype=np.longdouble), dtype=np.float64))
        continuity_tolerance = _tolerance(start, start if not self.rows else self.rows[-1]['storage_end'])
        lower = self.vmin if row['storage_lower_m3'] is None else row['storage_lower_m3']
        upper = self.vmax if row['storage_upper_m3'] is None else row['storage_upper_m3']
        # Distinguish strict independent conservation checks from the simulator's
        # explicit acceptance tolerances for physical bounds.
        storage_tolerance = np.maximum(1., 1e-10*np.abs(row['storage_start']))
        flow_tolerance = np.maximum(1e-8, _tolerance(row['release'], self.release_cap))
        power_tolerance = np.maximum(1e-7, _tolerance(row['power'], self.power_cap))
        turbine_tolerance = _tolerance(row['turbine'], self.turbine_cap)
        levels_start, levels_end = self._levels(row['storage_start']), self._levels(row['storage_end'])
        head = _vector(self.data.head, 'typical head')
        if self.physical['head_model'] == 'storage_sensitivity':
            scenario = self.data.head_sensitivity
            head = (scenario['reference_net_head_m']+scenario['head_slope_m_per_m']*
                    (.5*(levels_start+levels_end)-scenario['assumed_reference_forebay_level_m']))
        coefficient = (float(self.config['water_density_kg_m3'])*float(self.config['gravity_m_s2'])*
                       float(self.config['efficiency'])*head/1e6)
        expected_turbine = np.minimum(row['release'], np.minimum(self.turbine_cap, self.power_cap/coefficient))
        expected_power = expected_turbine*coefficient
        expected_energy = row['power']*self.dt/3600.
        rewards = self._rewards(row['power'], row['release'])
        differences = {name: np.asarray(row[name])-np.asarray(rewards[name])
                       for name in _SCALAR_REWARDS+_VECTOR_REWARDS}
        flags = dict(
            mass_balance=np.abs(residual) > tolerance,
            cascade_mass_balance=abs(cascade_residual) > cascade_tolerance,
            inflow_routing=np.abs(routing_residual) > routing_tolerance,
            storage_continuity=np.abs(continuity) > continuity_tolerance,
            storage_bounds=((row['storage_start'] < self.vmin-storage_tolerance) |
                            (row['storage_start'] > self.vmax+storage_tolerance) |
                            (row['storage_end'] < lower-storage_tolerance) |
                            (row['storage_end'] > upper+storage_tolerance)),
            release_bounds=((row['release'] < -flow_tolerance) |
                            (row['release'] > self.release_cap+flow_tolerance) |
                            (row['release'] < row['low']-flow_tolerance) |
                            (row['release'] > row['high']+flow_tolerance)),
            power_capacity=((row['power'] < -power_tolerance) |
                            (row['power'] > self.power_cap+power_tolerance)),
            turbine_capacity=((row['turbine'] < -turbine_tolerance) |
                              (row['turbine'] > self.turbine_cap+turbine_tolerance)),
            spill_partition=((row['spill'] < -flow_tolerance) |
                (np.abs(row['release']-row['turbine']-row['spill']) > flow_tolerance)),
            generation_calculation=((np.abs(row['power']-expected_power) > power_tolerance) |
                (np.abs(row['turbine']-expected_turbine) > turbine_tolerance) |
                (np.abs(row['energy']-expected_energy) > _tolerance(row['energy'], expected_energy))),
            level_lookup=((np.abs(row['levels_start']-levels_start) > 1e-9) |
                          (np.abs(row['levels_end']-levels_end) > 1e-9)),
            ecological=rewards['ecological_shortfall_m3s'] > 0.,
            forced=row['controls'] <= 0.)
        hard_names = ('mass_balance', 'inflow_routing', 'storage_continuity', 'storage_bounds',
                      'release_bounds', 'power_capacity', 'turbine_capacity', 'spill_partition')
        flags['hard_physical'] = np.logical_or.reduce([flags[name] for name in hard_names])
        reward_flags = {name: np.abs(differences[name]) >
            REWARD_ABSOLUTE_TOLERANCE+REWARD_RELATIVE_TOLERANCE*np.abs(rewards[name])
            for name in differences}
        flags['reward_mismatch'] = bool(any(np.any(value) for value in reward_flags.values()))
        checks = dict(mass_balance_residual_m3=residual, mass_balance_tolerance_m3=tolerance,
            cascade_mass_balance_residual_m3=cascade_residual, cascade_mass_balance_tolerance_m3=cascade_tolerance,
            inflow_routing_residual_m3s=routing_residual, storage_continuity_residual_m3=continuity,
            levels_start=levels_start, levels_end=levels_end, net_head_m=head,
            turbine=expected_turbine, power=expected_power, energy=expected_energy, **rewards)
        self.rows.append(row)
        self.checks.append(checks)
        self.differences.append(differences)
        self.flags.append(flags)

    def finish(self):
        if not self.started:
            raise RuntimeError('Begin a correctness episode before finishing it')
        if self.closed:
            return copy.deepcopy(self.result)
        stack = lambda rows: {key: [row[key] for row in rows] for key in rows[0]} if rows else {}
        flags = stack(self.flags)
        counts = {key: int(np.count_nonzero(value)) for key, value in flags.items()}
        active_days = len(self.rows)*int(np.count_nonzero(self.ecology > 0.))
        inventory = storage_change_summary(self.physical['initial_storage_m3'],
            self.rows[-1]['storage_end'] if self.rows else None)
        result = dict(schema_version=SCHEMA_VERSION, record_type='selected_episode_correctness',
            seed=self.seed, episode=self.episode, source_years=self.source_years,
            reservoir_ids=list(IDS), steps=len(self.rows), max_steps=self.max_steps,
            physical=self.physical, series=stack(self.rows), independent=stack(self.checks),
            reward_differences=stack(self.differences), violations=flags,
            summary=dict(violation_counts=counts,
                max_abs_mass_balance_residual_m3=(float(np.max(np.abs([row['mass_balance_residual_m3'] for row in self.checks]))) if self.checks else None),
                max_abs_cascade_mass_balance_residual_m3=(float(np.max(np.abs([row['cascade_mass_balance_residual_m3'] for row in self.checks]))) if self.checks else None),
                max_abs_reward_difference=(float(max(np.max(np.abs(value)) for row in self.differences for value in row.values())) if self.differences else None),
                active_ecological_reservoir_days=active_days,
                ecological_violation_rate=counts.get('ecological', 0)/active_days if active_days else None,
                hard_physical_reservoir_days=counts.get('hard_physical', 0),
                reward_mismatch_days=counts.get('reward_mismatch', 0)), storage=inventory,
            interpretation='Independent numerical checks of one executed trajectory; '
                'ecological deficits and endpoint inventory changes are distinct from hard physical failures. '
                'This does not establish hydraulic calibration, causal attribution, or optimal operation.')
        self.result = _json(result)
        self.closed = True
        return copy.deepcopy(self.result)
