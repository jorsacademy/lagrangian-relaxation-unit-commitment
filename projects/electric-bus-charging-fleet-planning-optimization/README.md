# Electric Bus Charging & Fleet Planning Optimization

Exact mixed-integer optimization models for charging-station location and daily electric-bus fleet planning in Python.

This repository replaces a previous greedy/random planning demonstration with two optimization models solved by SciPy/HiGHS:

1. a binary fixed-charge p-median-style facility-location MILP for charging-station selection and route assignment;
2. a time-expanded integer network-flow MILP for trip service, battery state of charge (SOC), charging, fleet capacity, and route-level service guarantees.

## Reference instance

The reproducible synthetic instance contains:

- 8 routes;
- 12 electric buses;
- 10 candidate charging-station sites;
- exactly 4 stations to build;
- 250 kWh battery capacity;
- 1.2 kWh/km traction consumption;
- 150 kW fast charging;
- 4 simultaneous fast chargers;
- 24 one-hour planning periods.

The random seed is fixed at `42` so the reference instance and regression results are reproducible.

## Optimization models

### 1. Charging-station location MILP

Binary variables determine which candidate sites are opened and which open station serves each route. The model enforces:

- exactly four stations are opened;
- unavailable land cannot be selected;
- every route is assigned to exactly one station;
- a route can only be assigned to an open station.

The objective combines normalized station CAPEX and passenger-demand-weighted access distance. The weights are configurable through `CAPEX_WEIGHT` and `ACCESS_WEIGHT`.

### 2. Fleet-operation MILP

The fleet model is an integer flow problem on a time-expanded `(hour, SOC)` network. Each unit of integer flow represents one bus. Feasible arcs represent idling, charging, or serving a route trip.

The formulation enforces:

- a fixed 12-bus fleet;
- route-hour demand upper bounds, so requested trips cannot be served more than once;
- battery SOC feasibility;
- trip duration occupancy;
- charging-station concurrency limits;
- minimum daily service coverage for every route;
- full-charge start and end states for the fleet.

The integer flow is explicitly decomposed into 12 individual bus duty paths after optimization.

## Battery modeling

SOC is discretized in 10 kWh increments to keep the MILP compact. Trip energy use is rounded **up** to the next SOC increment, making the battery feasibility approximation conservative rather than optimistic.

Charging uses 95% efficiency. Allowed one-hour net battery gains are 50, 100, or 140 kWh; 140 kWh is below the 142.5 kWh net energy implied by a 150 kW charger at 95% efficiency.

## Service policy

`MIN_ROUTE_COVERAGE = 0.15` is a hard policy constraint. It requires every route to receive at least 15% of its requested daily trips. This prevents a throughput-maximizing solution from abandoning lower-productivity routes.

It is a modeling policy, not a claimed industry target. Sensitivity testing showed the reference instance becomes infeasible at a 20% minimum route-coverage requirement with the current fleet, charger count, SOC assumptions, and full-charge end-of-day requirement.

## Reproducible reference results

Both MILPs terminate with HiGHS status `Optimal` for the reference instance.

### Charging infrastructure

| Metric | Result |
|---|---:|
| Selected stations | 2, 3, 7, 10 |
| Installation CAPEX | $1,003,328.81 |
| Passenger-weighted route-station distance | 26.43 km |

### Fleet operation

| Metric | Result |
|---|---:|
| Requested trips | 640 |
| Served unique trips | 143 |
| Trip coverage | 22.34% |
| Passenger-weighted coverage | 28.34% |
| Fleet utilization | 68.06% |
| Daily operated distance | 3,896.61 km |
| Traction energy | 4,675.94 kWh |
| Grid charging energy | 5,852.63 kWh |
| Maximum simultaneous charging buses | 4 |
| Daily operating cost | $6,357.97 |

Route-level service in the reference optimum:

| Route | Requested | Served | Coverage |
|---:|---:|---:|---:|
| 1 | 66 | 10 | 15.2% |
| 2 | 118 | 18 | 15.3% |
| 3 | 118 | 18 | 15.3% |
| 4 | 44 | 7 | 15.9% |
| 5 | 44 | 7 | 15.9% |
| 6 | 118 | 63 | 53.4% |
| 7 | 66 | 10 | 15.2% |
| 8 | 66 | 10 | 15.2% |

Route 6 receives more service because its combination of passenger demand, distance, and one-hour modeled duration makes it attractive after all route minimum-service constraints are satisfied.

## Sensitivity checks

The model responds consistently to charging-capacity changes:

| Fast chargers | Optimal served trips | Coverage |
|---:|---:|---:|
| 2 | 127 | 19.84% |
| 3 | 138 | 21.56% |
| 4 | 143 | 22.34% |
| 6 | 148 | 23.13% |

With one fast charger, the reference instance is infeasible under the 15% route-minimum policy.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python electric_bus_charging_fleet_optimization.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The regression suite checks station availability and assignment, demand non-duplication, route coverage, bus-duty decomposition, charging concurrency, and the reproducible reference optimum.

## Important modeling limitations

This is an operations-research reference implementation, not a deployment-ready transit scheduling system. In particular:

- route geometry and station coordinates are synthetic;
- travel times use a constant average speed;
- deadheading between route terminals and charging stations is not explicitly modeled in the fleet-flow network;
- charger queues within an hour are represented through a concurrency cap rather than continuous-time scheduling;
- SOC is discretized;
- passenger capacity, driver labor rules, depot positioning, maintenance windows, traffic uncertainty, battery degradation, and route-specific terminal compatibility are not included.

For operational use, these assumptions should be replaced with validated agency data and the corresponding constraints.

## Solver

The model uses `scipy.optimize.milp`, which calls the HiGHS mixed-integer solver distributed with SciPy.

## License

No license has been selected yet. Add a license before distributing the project under explicit open-source terms.
