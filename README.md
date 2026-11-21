# Five-reservoir HAPPO and MAPPO

Daily operation of the WDD, BHT, XLD, XJB and THR cascade using a specialized
feed-forward HARL implementation. The objective combines hydropower generation
with a soft ecological-flow penalty. Water balance, storage envelopes, outlet
capacities are hard physical constraints. Final storage is free within the
operating bounds; initial/final inventories and net depletion are reported
without an equality target or pass/fail test.

## Run

Install the dependencies declared in `setup.py` into a suitable Python environment:

```bash
python -m pip install -e .
```

Validate data, episode windows and observations without training:

```bash
python -B examples/train_cascade.py --check-data-only
```

Train HAPPO and MAPPO with paired seeds and three comparison figures:

```bash
python -B examples/train_cascade_compare.py \
  --episodes 300 --seeds 1 2 3 --gamma 0.995 --eval-interval 1 --diagnostics \
  --plot-output runs/ecological_comparison
```

The default comparison runs both algorithms; select `--algorithms happo` or
`--algorithms mappo` to run one. `examples/train_cascade.py` remains the HAPPO
entry point. The episode budget applies separately to each algorithm and seed.
`--no-plot` disables the legacy training/comparison figures and plot windows.
Automatic daily level/release diagnostics are still saved when `--episode-log`
provides a run directory. `--plot-python PATH` selects another interpreter with
Matplotlib. Existing output files are never overwritten.

By default results stay in memory. Add a fresh `--episode-log` path to retain
JSONL summaries and per-seed daily CSVs. For comparison runs that path is a stem:
separate `_happo.jsonl` and `_mappo.jsonl` logs are created. Add
`--save-checkpoints` to save final policies, or `--checkpoint-interval N` to also
save every N updates. Checkpoints support inference replay, not training resume.

## Data, observations and physical model

Inputs are in `data/cascade_reservoir_new/`. Actual historical daily external
inflows cover 1965–2024. BHT/XLD/XJB/THR columns are incremental interval inflows;
upstream executed release is added by the physical cascade. Releases propagate
upstream to downstream within one daily transition, with zero routing delay.

`five_reservoir_happo_input_bounds_1965_2024.csv` supplies the permitted next-state
storage and level for each action date. Its row for date t constrains the state
after action t; no extra date shift is applied. Previous executed bounds constrain
the current state, including across source-year joins.

Hard daily water-level ramps are configured under `daily_level_ramp_limits` in
`cascade_reservoir.yaml`. WDD/BHT/XLD/XJB use constant absolute limits of
1/2/3/4 m per day. THR uses 0.6 m/day in the dry months and 2.0 m/day in June--
September; its August--October refill rise limit is 3.0 m/day while downward
movement keeps the dry/flood seasonal limit. The current level plus/minus the
applicable rise/fall band is intersected with daily and operating level/storage
bounds before release dispatch. A disjoint intersection aborts the transition
with structured conflict evidence. Executed levels are checked again after mass
balance; storage and water levels are never clipped after execution.

Every decision, including the last day, uses the same physical release mapping.
There is no prescribed final water level, reserve-window override, terminal
projection, seasonal refill budget or terminal-deviation reward. Storage is
always calculated from executed flows. Episode boundaries still stop value
bootstrapping; this is independent of reservoir inventory requirements.

Actor observations use one fixed 22-feature layout for all five reservoirs. The
first 14 local features are normalized current water level; actual total inflow
(external or incremental inflow plus the previous day's executed upstream
release) divided by release capacity; the supplied issued next-seven-day mean
incremental inflow forecast, standardized using training-year actual-inflow
statistics; seasonal sine and cosine; next-day maximum and minimum permitted
water levels; the previous requested normalized action; current storage ratio;
remaining regulation-capacity ratio; previous actual total release, turbine
release and power, each divided by its capacity; and the previous realized action
after environmental correction. The next three features are the immediate
upstream reservoir's water level, actual release and issued forecast; WDD uses
`[0, 0, 0]`. The next two are the immediate downstream reservoir's water level
and remaining regulation-capacity ratio; THR uses `[0, 1]`. The final three are
total storage divided by combined normal storage, total realized power divided by
total installed capacity, and a flood-season indicator equal to 1 from June 1
through September 30 and 0 otherwise.

Forecasts are read from the supplied file, including at nonconsecutive held-out
year joins; they are not recomputed from future actual inflows. The centralized
critic receives the five observations concatenated in WDD/BHT/XLD/XJB/THR order,
for 110 input features. Checkpoints record the observation and critic layouts;
version 2 checkpoints use this contract, and earlier checkpoints cannot be loaded
because their observation dimensions and source hashes differ.

For categorical action k among K commands:

```text
u = k / (K - 1)
release = feasible_low + u² * (feasible_high - feasible_low)
```

Feasible release intervals enforce storage bounds, daily level-ramp limits,
outlet capacities and downstream compatibility. Physically forced intervals
mask actor learning and contribute a unit HAPPO correction factor. The default
100-command grid can be changed with `--action-bins`.

Generation uses turbine flow, installed capacity and net head. The default
`storage_sensitivity` head model depends on mean start/end forebay elevation and
supplied slope/offset assumptions; this is a sensitivity model. `--head-model
fixed` selects typical constant heads.

## Ecological flow and rewards

`reservoir_service_requirements.csv` supplies only `ecological_flow_m3s`, in
WDD/BHT/XLD/XJB/THR order. The requirement applies to total reservoir release,
including turbine flow and spill, on every active day. Zero thresholds are
inactive. `--ecological-flow-scale` scales reference CSV values and records the
effective thresholds and input hash. Ecological compliance is a soft objective;
it does not replace commands or impose a hard release floor.

```text
relative_flow_deficit_i = max(threshold_i - release_i, 0) / threshold_i
local_i = power_i / installed_capacity_i
          - ecological_flow_weight * relative_flow_deficit_i
global = sum(power_i) / sum(installed_capacity_i)
mixed_i = 0.5 * local_i + 0.5 * global
system = 0.5 * sum(local_i) + 0.5 * global
```

The deficit is zero for inactive thresholds and tolerance-filtered for numerical
roundoff. Local penalties use each reservoir's own relative deficit. The global
term measures generation only and enters system accounting once. The legacy
team metric is normalized total generation minus the mean ecological deficit
over the fixed active reservoir set. It remains separate from the learning
reward and from the sum of mixed rewards. `--no-reward-penalties` selects a
generation-only objective while physical ecological counts remain observable.

## Algorithms and training

The main settings are in `harl/configs/algos_cfgs/happo_cascade.yaml` and physical
settings in `harl/configs/envs_cfgs/cascade_reservoir.yaml`. Defaults are six
process-parallel episodes per update, 1,825 daily transitions per episode,
300 episodes per seed, gamma 0.995, GAE lambda 0.95 and five actor/critic epochs.
The default therefore performs 50 updates per seed. Leap days remain in the data;
the fixed horizon need not end on December 31.

HAPPO uses five independent actors and one centralized critic. The default
`team_gae` mode trains on `system_reward` with one value output; all actors share
one team GAE advantage, normalized once across rollout rows. The `mixed_gae` mode
trains on each reservoir's `mixed_rewards[i]` with five critic value outputs and
five independently computed GAE channels; each actor receives only its own
channel. Both modes update actors sequentially with accumulated likelihood
corrections for previously updated actors and PPO clipping of the current actor
ratio. Control masks zero forced rows, and losses retain a mean over the entire
rollout.

MAPPO uses one shared actor and optimizer. Five lightweight views append a
five-way one-hot reservoir identity, giving 27 actor network inputs.
Own mixed-return advantages are pooled in one joint training batch. One
centralized critic produces five mixed-return values plus a legacy accounting
value. MAPPO uses unit factors and no sequential likelihood correction. Its
sharing, reward targets and critic layout differ from HAPPO, so comparisons do
not isolate a single algorithmic difference.

Actor and critic MLPs use hidden widths `[64, 128, 256, 256]`, ReLU, hidden
LayerNorm, categorical action distributions, Adam and gradient clipping. GAE and
actor advantages use rollout ValueNorm statistics. The normalizer updates once
from full-rollout critic targets after actor training and stays fixed through
critic epochs. Training uses CPU execution.

`--happo-credit-assignment mixed_gae` selects the five-channel mixed-return mode.
`--happo-credit-assignment counterfactual` optionally replaces HAPPO actor GAE
with frozen one-step, model-assisted counterfactual credit. It enumerates commands
under the same full physical constraints and ecological reward, holding other
commands fixed. The scalar team critic retains its GAE targets and sequential
HAPPO corrections remain. Default credit assignment is `team_gae`.

## Sampling, evaluation and plots

Five source years are held out by default: 1998, 1999, 1970, 1997 and 2006.
Training windows contain training years only. Coverage-balanced sampling reduces
unequal year-day exposure; `--window-sampling uniform` restores uniform draws.
Storage carries across source-year boundaries within each episode.

Periodic evaluation runs on three fixed training-only scenarios before update
one, every configured interval and after the final update. It uses deterministic
policies and fresh environments without updating models, optimizers or RNG state.
Final held-out evaluation uses an ordered representative-year sequence and the
same episode horizon. Physical infeasibility is reported explicitly; failed
windows remain failures, with no completed aggregate substituted from survivors.

Three standard plot artifacts are available: stochastic training return/generation,
fixed-scenario evaluator return, and a single-panel ecological-flow violation
rate. Means and sample standard deviations stay separate by algorithm. Violation
rates count violating active reservoir-days, independent of reward weights and
gamma. Ecological deficit volumes and generation are undiscounted; reported
returns retain gamma. Failed or missing evaluations leave gaps.

Return plots follow the recorded learning objective: HAPPO uses shared-system
returns (including counterfactual-credit HAPPO), while MAPPO uses the mean of
its five individual mixed returns. Comparison legends identify these different
objectives. Saved-log plots reconstruct training returns from all completed
episode summaries in each update. Older logs without objective metadata retain
explicitly labeled legacy-team returns; declared learning objectives with missing
return data are rejected instead of substituting legacy accounting rewards.

```bash
python -B examples/plot_cascade_parallel.py --episode-log runs/EXPERIMENT.jsonl
python -B examples/plot_cascade_compare.py \
  --happo-log runs/EXPERIMENT_happo.jsonl --mappo-log runs/EXPERIMENT_mappo.jsonl
```

Plot validators check scenario identities, data hashes, physical settings and
training budgets. Forecast input hashes identify current actor dimensions while
older logs without that input retain their original dimensions.

## Automatic daily change diagnostics

Every run with a final held-out evaluation and either `--plot-output` or
`--episode-log` saves daily level and release diagnostics beside that output.
With the timestamped `--plot-output "$run_dir/curves"` pattern, files go directly
in `$run_dir`. Each seed gets one raw CSV, one level-statistics CSV, one ramp
conflict CSV, and four PNGs:
`daily_level_change.png`, `abs_level_change_heatmap.png`,
`level_change_statistics.png`, and `daily_release_change.png` (prefixed by the
algorithm and seed). The first daily difference is an empty CSV cell; all later
differences are current minus previous in episode order. Nonconsecutive held-out
year joins are retained on the actual-date axis, while the level difference across
each join remains part of the continuous evaluation sequence.

The figures show five reservoir panels per seed and the applicable daily rise/fall
envelope. Points outside the hard ramp limit are marked in red (blue crosses in
the heatmap); the conflict CSV records infeasible dates, reservoirs, limits,
bounds and reasons even when it contains only a header. The heatmap uses actual
calendar year and day-of-year, including leap days. The statistics panels and
summary CSV report mean, linearly interpolated P95, maximum `|ΔZ|`, the largest
applicable limit, and ramp-violation counts. These diagnostics read executed
end-of-day levels and daily releases only; they do not alter observations,
rewards, actions, constraints, or optimization.

## Correctness figures

`--correctness-output NEW_DIRECTORY` enables eight additional diagnostic figures
and saves their numerical evidence to `NEW_DIRECTORY/evidence.json`. It works
with `--no-plot`: these eight figures are saved using a headless Matplotlib backend,
while the standard interactive figures remain disabled. Select an interpreter
with NumPy and Matplotlib using `--plot-python` when training uses a different
Python environment.

```bash
python -B examples/train_cascade.py --seeds 1 2 3 \
  --correctness-output runs/happo_correctness_seeds123 --no-plot
```

The files are `mass_balance.png`, `water_levels.png`,
`release_feasibility.png`, `storage_change.png`,
`reward_reconciliation.png`, `happo_updates.png`, `operation_trajectories.png`,
and `training_reliability.png`.

Physical traces come from the final policy's deterministic median training-only
evaluation window, reusing the existing evaluation rather than collecting extra
training samples. One complete window per seed is retained. Balance residuals,
routing, physical limits and reward components are independently reconstructed
from the raw executed storage/flow/power values. Missing or partial trajectories,
ecological deficits, physically forced releases and inventory changes remain visible.

The additional `operation_trajectories.png` combines water levels (solid lines,
left axis) and release discharges (dashed lines, reversed right axis), with one
reservoir per row and one seed per column. Its horizontal axis covers exactly
five complete calendar years from the selected median training-only window,
including leap days. A separate deterministic final-policy replay covers the
complete calendar interval; the existing fixed-day diagnostics and training
episode lengths are unchanged. Filled markers and numeric labels show the
end-of-day water level on December 31 of each annual segment. Gray shading
identifies June through September, and vertical separators mark annual
boundaries. The replay is observational and does not update the policy.

Storage evidence retains the initial/final inventories of every collected
training episode, every periodic evaluation round, and the held-out sequence.
The storage-change figure displays the latest batch/round while reporting
all-history inventory counts. Positive values indicate accumulation and negative
values indicate depletion; neither is labeled a constraint violation. Completion
and physical feasibility are separate from soft ecological compliance.
`required_constraints_met` refers only to completed physical operation.
HAPPO evidence spans every update: the actor-entry factor is checked
against a separate float64 sum of categorical log-likelihood changes, with forced
commands contributing a unit ratio. Factor errors, final likelihood ratios, KL,
clipping and entropy are recorded as bounded summaries. These observers do not
sample actions, advance optimizers or change training targets.

`training_reliability.png` adds read-only Critic fit and convergence evidence.
It reports realized-return explained variance, error and bias normalized by the
corresponding return spread, one line per unique value channel. The same figure
shows fixed-scenario evaluator returns and ecological-flow violation rates for
every seed and the spread of each seed's final five-round mean. Shared-team GAE
has one Critic channel; mixed GAE has five reservoir-aligned channels. Legacy
evidence without these optional fields renders the panels as unavailable.

Correctness evidence uses schema `cascade_correctness_v3`. Critic fit histories
are read from each entry in `update_records`; the run-level `critic_quality` is
only an index of the latest update. The convergence panel follows the declared
learning objective: `system_evaluator_return` for `team_gae`, or the mean of
`mixed_evaluator_returns` for `mixed_gae`. It never substitutes the legacy
`evaluator_return` when the selected field is missing.

Each run with `--episode-log` or `--plot-output` writes an exclusive artifact
manifest after its log and figures are complete. Records carry the same `run_id`
as the run configuration, and the correctness renderer writes and verifies
`manifest.json` beside `evidence.json`. The log loader checks the run identity
and final log hash against `*_manifest.json`; changed or foreign artifacts are
rejected. Each algorithm in a paired comparison has its own log-derived
manifest, and preflight rejects a stale manifest before either algorithm trains.
Read-only v2 evidence remains renderable for historical inspection, but it
cannot supply learning-objective convergence without the selected return.

The feature is HAPPO-only, including counterfactual-credit HAPPO. For the
comparison entry point, select `--algorithms happo`. Existing output directories
are rejected before training. Figures illustrate numerical consistency of the
modeled problem; they do not establish hydraulic calibration or optimal operation.

Saved evidence can be replotted without training or loading models:

```bash
python -B examples/plot_cascade_correctness.py \
  --input runs/happo_correctness_seeds123/evidence.json \
  --output-dir runs/happo_correctness_replot --no-show
```

The plotter rejects existing figure targets and leaves its input evidence intact.

## Diagnostics and replay

The console shows concise per-seed progress by default. Add `--verbose-console`
to print full configuration and JSON diagnostics. This flag changes console
output only; `--episode-log`, trajectory files, checkpoints and correctness
evidence retain their complete contents.

`--diagnostics` records parameter changes, entropy/KL, surrogate changes and
Monte Carlo/GAE fit on collected trajectories. `--trajectory-diagnostics` requires
an episode log and records physical dispatch, ecological deficits, reward
components, frozen values, GAE and policy factors per reservoir/day/lane. Global,
system and legacy scalar rewards repeat across five CSV rows; count each once
per day/lane when reconstructing totals.

```bash
python -B examples/replay_cascade_checkpoint.py \
  --checkpoint runs/EXPERIMENT_seed1_checkpoints/update_000050.pt \
  --source-years 2007 2008 2009 2010 2011 --output runs/EXPERIMENT_replay.csv
```

`--override-action DAY RESERVOIR INDEX` changes one command; later commands react
through the saved policy. Replay preserves the saved horizon, physical and data
hashes, forecast layout and physical dispatch semantics. It reports generation, ecological
deficits, learning returns, spill and final storage without fitting models.

`analysis/dispatch_diagnostics/` reconstructs frozen trajectories and tests
same-state ecological release alternatives through the full cascade. Grid and
continuous witnesses retain physical bounds and the other sampled commands. These are separate same-day
comparisons, not an executable multi-day intervention or evidence of optimality.

## Source map

| Path | Responsibility |
| --- | --- |
| `examples/train_cascade.py` | CLI, configuration and split validation |
| `examples/train_cascade_compare.py` | Paired HAPPO/MAPPO orchestration |
| `examples/train_cascade_parallel.py` | Seed runs, logging and evaluation |
| `harl/runners/` | Full-episode collection and algorithm update order |
| `harl/algorithms/`, `harl/models/`, `harl/common/` | PPO, critics, networks, buffers and ValueNorm |
| `harl/envs/cascade_reservoir/data_loader.py` | Historical data, forecasts and ecological inputs |
| `harl/envs/cascade_reservoir/reservoir_physics.py` | Water balance, generation and hard feasibility |
| `harl/envs/cascade_reservoir/level_ramp.py` | Hard daily level-ramp bounds and conflict CSV evidence |
| `harl/envs/cascade_reservoir/reward.py` | Ecological rewards and episode accounting |
| `harl/envs/cascade_reservoir/evaluation.py`, `periodic_evaluation.py` | Evaluation and violation counts |
| `harl/envs/cascade_reservoir/checkpoint.py` | Provenance-checked inference replay |
| `analysis/dispatch_diagnostics/` | Ecological dispatch evidence and reports |

Run regression tests with `python -B -m unittest discover -s tests`.
