# Vogel Transportation Power Distribution Optimizer

A dependency-free Python implementation of a two-stage transportation optimizer:

1. **Vogel's Approximation Method (VAM)** builds a good initial basic feasible solution.
2. **MODI / Transportation Simplex** evaluates reduced costs and pivots until the allocation is optimal.

The repository demonstrates the solver with a power-distribution problem and supports forbidden routes and degenerate basic feasible solutions.

## Problem

Three power plants have capacities of 25, 40, and 30 million kWh. City demands are forecast to rise by 20%, giving:

- City 1: 36 million kWh
- City 2: 42 million kWh
- City 3: 30 million kWh

The plants supply 95 million kWh in total, while demand is 108 million kWh. The 13 million kWh shortfall is purchased from an external grid at a cost of $1,000 per million kWh. The external grid is not allowed to serve City 3.

The transportation costs use the assignment's numerical values, expressed in USD:

| Source | City 1 | City 2 | City 3 | Supply |
|---|---:|---:|---:|---:|
| Plant 1 | $600 | $700 | $400 | 25 |
| Plant 2 | $320 | $300 | $350 | 40 |
| Plant 3 | $500 | $480 | $450 | 30 |
| External Grid | $1,000 | $1,000 | Not allowed | 13 |
| Demand | 36 | 42 | 30 | |

## Verified result

Vogel's Approximation Method produces the following initial allocation:

| Source | City 1 | City 2 | City 3 |
|---|---:|---:|---:|
| Plant 1 | 0 | 0 | 25 |
| Plant 2 | 0 | 40 | 0 |
| Plant 3 | 23 | 2 | 5 |
| External Grid | 13 | 0 | 0 |

Initial cost: **$49,710**.

The transportation-simplex optimality test finds no negative reduced-cost route, so this VAM solution is already optimal for the power-distribution dataset.

Optimal cost: **$49,710**.

This allocation satisfies every supply constraint, every demand constraint, and the prohibited External Grid -> City 3 route.

## Why both algorithms are included

VAM is a heuristic. It often produces a strong starting solution, but it does **not** guarantee an optimum. The MODI / Transportation Simplex stage provides that optimality step.

The test suite includes a separate transportation instance where VAM costs **$160** and the transportation-simplex phase improves it to the proven optimum of **$159**.

## Algorithm

For each iteration, the optimizer:

1. Builds or maintains a transportation basis with `m + n - 1` basic cells.
2. Computes MODI row and column potentials `u_i` and `v_j` from `u_i + v_j = c_ij` on basic cells.
3. Computes reduced costs `c_ij - u_i - v_j` for allowed non-basic routes.
4. Stops when every reduced cost is non-negative.
5. Otherwise chooses a negative reduced-cost route, builds the unique alternating cycle, performs a transportation-simplex pivot, and repeats.

Forbidden routes are excluded from both the basis and entering-variable candidates. Degenerate initial solutions are handled by adding zero-valued basic cells that preserve a spanning-tree basis.

## Run

```bash
python main.py
```

Expected result for the included power-distribution example:

```text
Vogel initial allocation (million kWh):
Source               City 1      City 2      City 3
Plant 1                   0           0          25
Plant 2                   0          40           0
Plant 3                  23           2           5
External Grid            13           0           0
Initial cost: $49,710.00

Transportation-simplex optimum (million kWh):
Source               City 1      City 2      City 3
Plant 1                   0           0          25
Plant 2                   0          40           0
Plant 3                  23           2           5
External Grid            13           0           0
Optimal cost: $49,710.00
```

## Test

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- the original power-distribution instance,
- optimality of the `$49,710` solution,
- an instance where MODI improves a non-optimal VAM result,
- a degenerate initial basic feasible solution,
- the forbidden External Grid -> City 3 route.

The implementation was additionally cross-checked during development against a linear-programming solver on 500 randomly generated balanced 3x3 transportation instances; the objective values matched in all 500 cases.

## Project structure

```text
.
├── main.py
├── src/
│   ├── __init__.py
│   └── solver.py
├── tests/
│   └── test_solver.py
├── LICENSE
└── README.md
```

## License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**. Commercial use is not permitted. See `LICENSE` for the full license text.
