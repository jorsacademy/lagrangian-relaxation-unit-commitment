# Industrial Energy Management with Soft Actor-Critic

A compact industrial-engineering case study for controlling factory electricity demand with a battery energy storage system (BESS). The project formulates the problem as a continuous-action Markov decision process and compares reinforcement learning with interpretable baseline controllers.

## Industrial problem

Manufacturing sites increasingly face time-varying electricity tariffs, local renewable generation, battery storage, and demand charges. A controller must decide when to charge or discharge storage while balancing several competing objectives:

- reduce purchased electricity cost,
- shave costly demand peaks,
- avoid unnecessary battery cycling,
- respect battery state-of-charge and power constraints.

This is a natural sequential decision problem because an action taken now changes the battery state available for future high-price or high-load periods.

## MDP formulation

### State

The environment exposes a normalized five-dimensional observation:

1. battery state of charge,
2. electricity price,
3. factory load,
4. renewable generation,
5. time within the operating horizon.

### Action

The action is continuous in `[-1, 1]` and is mapped to battery power:

- `-1`: maximum charging,
- `0`: idle,
- `+1`: maximum discharging.

The environment clips requested power to feasible battery power based on state of charge, capacity, efficiency, and power limits.

### Reward

The agent minimizes total operating cost. Reward is the negative of:

`energy cost + peak-demand penalty + battery degradation cost`

The environment also tracks operational KPIs rather than only cumulative reward.

## Why SAC?

Soft Actor-Critic is well suited to this problem because battery dispatch is a continuous-control decision. SAC is off-policy, generally sample-efficient, and explicitly encourages exploration through entropy regularization. In an industrial setting, the trained policy would normally be developed in simulation or a digital twin before deployment.

## Baselines

Three deterministic controllers are included:

- `IdleBatteryPolicy`: battery is disabled; useful as the no-control reference.
- `RuleBasedTariffPolicy`: charges during low-price or renewable-rich periods and discharges during expensive/high-load periods.
- `PeakShavingPolicy`: an MPC-inspired feedback heuristic that attempts to keep grid demand near a target level.

These baselines make it possible to judge whether RL actually creates economic value instead of merely showing that an agent can learn.

## Key performance indicators

The evaluator reports:

- mean total operating cost,
- standard deviation of operating cost,
- maximum grid import,
- energy cost,
- battery degradation cost.

A useful extension is to add CO2 intensity and production-throughput constraints so that energy and manufacturing objectives are optimized jointly.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── pyproject.toml
├── src/industrial_energy_management/
│   ├── __init__.py
│   ├── environment.py
│   ├── baselines.py
│   ├── train.py
│   └── evaluate.py
└── tests/
    ├── test_environment.py
    └── test_baselines.py
```

## Installation

For the environment, baselines, and tests:

```bash
pip install -e '.[test]'
```

For SAC training:

```bash
pip install -e '.[rl,test]'
```

## Run baseline experiments

```bash
python -m industrial_energy_management.evaluate --episodes 20
```

The same random seeds are used across policies, giving each controller identical stochastic load and renewable scenarios.

## Train SAC

```bash
python -m industrial_energy_management.train \
  --timesteps 50000 \
  --output models/sac_industrial_energy \
  --seed 42
```

Then compare the trained policy with the baselines:

```bash
python -m industrial_energy_management.evaluate \
  --episodes 50 \
  --model models/sac_industrial_energy
```

## Run tests

```bash
pytest -q
```

GitHub Actions runs the tests and a baseline smoke test on Python 3.10, 3.11, and 3.12.

## Research extensions

This benchmark is intentionally compact. Strong academic or industrial extensions include:

1. **MPC vs SAC** — formulate a finite-horizon optimization model with perfect or forecast information and compare cost, peak demand, robustness, and inference time.
2. **Forecast uncertainty** — expose demand/renewable forecasts and forecast errors rather than the current state only.
3. **Demand response** — allow flexible production loads to shift between time periods.
4. **Multiple energy assets** — CHP, thermal storage, EV fleets, compressed-air storage, and multiple batteries.
5. **Carbon-aware control** — include time-dependent marginal grid emissions.
6. **Constrained RL** — enforce hard peak, temperature, or battery-health constraints.
7. **Offline RL** — learn from historical EMS/SCADA trajectories without exploratory interaction with the real plant.
8. **Digital twin integration** — replace synthetic profiles with a discrete-event or physics-based manufacturing model.

## Industrial interpretation

A production deployment should not allow an unconstrained RL policy to control critical plant infrastructure directly. A realistic architecture is:

`SCADA / meters -> forecasting & digital twin -> optimization/RL policy -> safety layer -> EMS/BESS controller`

The safety layer should enforce equipment ratings, SOC limits, contractual demand limits, fallback behavior, and operator overrides.

## Scope

The numerical parameters in this repository are synthetic and are intended for experimentation, education, and methodological comparison rather than direct plant deployment.
