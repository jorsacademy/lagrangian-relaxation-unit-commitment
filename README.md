# Lagrangian Relaxation for Unit Commitment

A reproducible implementation of **Lagrangian relaxation for thermal unit commitment** with unit-wise decomposition, Polyak-style subgradient updates, primal recovery, exact small-instance verification, tests, and CI.

The repository is intentionally transparent. It demonstrates the central Lagrangian-relaxation idea: a difficult mixed-integer scheduling model becomes separable after a small set of coupling constraints is dualized. For the unit-commitment model here, hourly demand balance is dualized and each generating unit can then be optimized independently over the full time horizon.

This is a teaching and research benchmark. It is not a production security-constrained unit-commitment solver.

## Unit-commitment model

For thermal unit `i` and hour `t`:

- `u_it` is the on/off commitment state;
- `p_it` is generation;
- `y_it` is a startup indicator.

The primal problem minimizes operating cost:

```text
minimize
    sum(i,t) marginal_cost_i * p_it
  + sum(i,t) no_load_cost_i * u_it
  + sum(i,t) startup_cost_i * y_it
```

subject to:

```text
sum_i p_it = demand_t                         for every hour t
p_min_i * u_it <= p_it <= p_max_i * u_it
minimum up-time restrictions
minimum down-time restrictions
u_it in {0,1}
```

Initial on/off status and the duration already spent in that status are part of the instance. Minimum up/down feasibility is enforced through admissible commitment-state transitions.

The benchmark deliberately omits ramping, reserve, network, fuel, emission, and security constraints so that the decomposition mechanics remain inspectable.

## Exact commitment-pattern formulation

Each thermal unit has a finite set of commitment patterns that respect:

- initial state;
- initial state duration;
- minimum up time;
- minimum down time.

The exact benchmark enumerates these patterns for each unit. The full UC model chooses exactly one pattern per unit and optimizes continuous hourly generation subject to demand balance and output limits.

This pattern formulation is exact for the declared model and is solved using `scipy.optimize.milp` / HiGHS. It provides a trusted small-instance primal optimum against which the Lagrangian bounds can be checked.

## Lagrangian relaxation

The hourly demand equations are the only constraints that couple different generating units:

```text
sum_i p_it = demand_t
```

Introduce unrestricted multipliers `lambda_t` and write the Lagrangian for the minimization problem as:

```text
L(u,p,y,lambda)
=
original_cost
+ sum_t lambda_t * (demand_t - sum_i p_it)
```

Rearranging gives:

```text
L
= sum_t lambda_t * demand_t
+ sum_i [
      commitment/startup cost of unit i
    + sum_t (marginal_cost_i - lambda_t) * p_it
  ]
```

The expression inside the brackets depends on only one unit. Therefore, for fixed multipliers, the relaxed problem decomposes into independent unit subproblems.

For a minimization problem, the Lagrangian dual value is a valid lower bound on the primal optimum for every multiplier vector. This is the classical lower-bound property emphasized in Fisher's survey of Lagrangian relaxation.

## Unit subproblems

Each unit subproblem is solved exactly over its feasible commitment patterns.

For an hour in which a unit is on, the relaxed generation coefficient is:

```text
marginal_cost_i - lambda_t
```

Because generation is continuous only between `p_min` and `p_max` and the cost is linear:

```text
if marginal_cost_i - lambda_t < 0:
    choose p_max
else:
    choose p_min
```

for that committed hour.

The resulting unit subproblem is therefore a finite commitment-pattern search plus analytical dispatch at the generation bounds. No centralized UC solve is needed to evaluate the Lagrangian dual function.

## Dual subgradient

Let the total relaxed generation at multiplier vector `lambda` be:

```text
P_t(lambda) = sum_i p_it(lambda)
```

A supergradient of the concave Lagrangian dual function is:

```text
g_t = demand_t - P_t(lambda)
```

The multiplier update is an ascent step:

```text
lambda_{k+1} = lambda_k + alpha_k * g_k
```

The implementation uses a Polyak-style step length:

```text
alpha_k
= theta * (UB - q(lambda_k)) / ||g_k||^2
```

where:

- `q(lambda_k)` is the current Lagrangian dual value;
- `UB` is the best known feasible primal cost;
- `theta` is a step-control parameter in `(0, 2]`.

When the best lower bound stalls, `theta` is reduced. Multipliers are not projected because they correspond to equality constraints and are unrestricted in sign.

The best dual bound is stored separately from the current dual value because raw subgradient iterates need not improve monotonically.

## Primal recovery

A Lagrangian solution generally violates demand balance because the units are optimized independently.

The repository converts the decomposed commitment decisions into a feasible primal schedule in two steps:

1. choose one feasible commitment pattern per unit while minimizing Hamming distance from the current decomposed patterns, subject to sufficient maximum capacity and compatible minimum generation in every hour;
2. economically dispatch the fixed commitments to meet demand exactly.

The first step is a small MILP over commitment patterns. The second step is exact linear dispatch because marginal generation costs are linear and there are no ramp constraints.

Every recovered schedule is therefore primal feasible and provides a valid upper bound. The recovery heuristic is not claimed to be optimal.

## Demo instance

The default benchmark contains four thermal units and eight hourly demand periods.

```text
Unit       Pmin   Pmax   marginal   no-load   startup   min-up   min-down
Base-1       20     80       18        120       350       3         2
Base-2       15     60       23         90       220       2         2
Mid-1        10     45       31         55       130       2         1
Peaker        0     35       52         15        40       1         1
```

Demand:

```text
[95, 110, 135, 155, 145, 120, 100, 85]
```

## Reproducible development result

Run:

```bash
python -m lagrangian_uc --iterations 120
```

The exact commitment-pattern MILP gives:

```text
exact UC objective            20705.000
```

The Lagrangian-relaxation run gives:

```text
best dual lower bound         20270.853
best recovered upper bound    20705.000
absolute LR duality gap         434.147
relative LR duality gap          2.097%
```

The recovered primal heuristic finds the exact optimum on this particular benchmark. That is an observed result for this instance, not a guarantee of the recovery procedure.

Best-bound progression:

```text
iteration    best lower bound    best upper bound
1                 19515.000          20915.000
5                 20025.530          20705.000
10                20068.196          20705.000
20                20092.818          20705.000
40                20120.476          20705.000
80                20252.100          20705.000
120               20270.853          20705.000
```

The residual gap is meaningful. Lagrangian relaxation of an integer program can have a nonzero duality gap; the method does not imply that subgradient optimization must recover the exact primal optimum or close the dual gap to zero.

The best multiplier vector from this seeded deterministic run is approximately:

```text
[22.845, 23.010, 23.987, 36.331, 30.858, 24.507, 24.623, 24.379]
```

These multipliers can be interpreted as shadow-price-like coordination signals for hourly energy balance, but they are Lagrange multipliers of a nonconvex mixed-integer model and should not automatically be interpreted as market-clearing prices.

## Exact optimal schedule

The exact benchmark commits both base units throughout the horizon, commits `Mid-1` during the two-hour peak block, and leaves the peaker off:

```text
Base-1   1 1 1 1 1 1 1 1
Base-2   1 1 1 1 1 1 1 1
Mid-1    0 0 0 1 1 0 0 0
Peaker   0 0 0 0 0 0 0 0
```

Generation is:

```text
Base-1   80 80 80 80 80 80 80 70
Base-2   15 30 55 60 55 40 20 15
Mid-1     0  0  0 15 10  0  0  0
Peaker    0  0  0  0  0  0  0  0
```

## Verification strategy

The regression suite checks:

- minimum up/down behavior under initial conditions;
- startup-marker consistency;
- exact demand balance and generation bounds;
- exact-objective regression for the demo instance;
- primal-recovery feasibility;
- Lagrangian lower-bound validity at several multiplier levels;
- the dual supergradient inequality for a local perturbation;
- expected high-multiplier generation behavior;
- monotonicity of the stored best lower bound;
- valid primal/dual bound ordering throughout the algorithm;
- CLI JSON output.

The dual supergradient test is important because a sign error in the demand residual would make the multiplier update descend rather than ascend the concave dual function.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.11+ is required.

## Run

```bash
python -m lagrangian_uc
```

Shorter run:

```bash
python -m lagrangian_uc --iterations 40 --theta 1.4
```

A compact example is also available:

```bash
python examples/run_demo.py
```

## Tests

```bash
pytest -q
```

GitHub Actions runs package installation, byte-code compilation, Ruff, the full test suite, and an end-to-end Lagrangian-relaxation smoke experiment on Python 3.11 and 3.12.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_demo.py
├── src/lagrangian_uc/
│   ├── __init__.py
│   ├── __main__.py
│   ├── data.py
│   ├── experiment.py
│   ├── lagrangian.py
│   ├── master.py
│   └── patterns.py
├── tests/
│   ├── test_cli.py
│   ├── test_dual.py
│   ├── test_exact.py
│   ├── test_lagrangian.py
│   ├── test_more.py
│   └── test_patterns.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Methodological boundaries

This repository intentionally does **not** claim to implement a production unit-commitment engine. In particular, it omits:

- ramp-rate constraints;
- spinning and operating reserve;
- transmission/network constraints;
- N-1 security constraints;
- piecewise-linear or quadratic heat-rate curves;
- shutdown costs;
- hot/warm/cold startup trajectories;
- must-run or outage constraints;
- hydro, storage, or renewable uncertainty;
- stochastic or robust UC;
- augmented Lagrangian, bundle, volume, or surrogate-subgradient methods;
- parallel unit-subproblem execution.

The exact pattern formulation and the recovery MILP are deliberately small-instance verification tools. On a large operational UC model, commitment-pattern enumeration would not be appropriate.

A stronger extension would add ramping and reserve, compare ordinary subgradient optimization with bundle or volume methods, parallelize unit subproblems, and benchmark against a monolithic MILP on a family of larger instances.

## Research grounding

- M. L. Fisher, **The Lagrangian Relaxation Method for Solving Integer Programming Problems**, *Management Science* 27(1), 1-18 (1981). DOI: `10.1287/mnsc.27.1.1`.
- J. F. Bard, **Short-Term Scheduling of Thermal-Electric Generators Using Lagrangian Relaxation**, *Operations Research* 36(5), 756-766 (1988). DOI: `10.1287/opre.36.5.756`.
- SciPy documentation for `scipy.optimize.milp` and HiGHS-backed linear/integer optimization.

Fisher's survey gives the general lower-bound and decomposition interpretation of Lagrangian relaxation. Bard applies the idea directly to thermal unit commitment by disaggregating the model by generator and using dual information to construct feasible schedules.

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
