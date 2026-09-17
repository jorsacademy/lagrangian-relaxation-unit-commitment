# Energy Optimization Research Series

This file maps energy-related optimization repositories across generation scheduling, storage, electrification, industrial energy control, and renewable-layout decisions. It is an index only: the projects remain separate because their decision variables, time scales, uncertainty structures, and algorithms differ.

## Power-system scheduling and decomposition

- `lagrangian-relaxation-unit-commitment` — thermal unit commitment solved through Lagrangian relaxation and unit-wise decomposition.
- `sddp-multistage-energy-storage` — multistage stochastic optimization for storage decisions.
- `mpi-sppy-multistage-stochastic-planning` — distributed stochastic-planning infrastructure that is relevant to larger energy planning models as well as other domains.

## Industrial energy and control

- `industrial-energy-management-sac` — continuous control/energy-management decisions using reinforcement learning.
- `production-control-with-mpc-vs-rl` — comparison of model-predictive control and reinforcement learning in a production-control setting with energy/control relevance.
- `safe-rl-constrained-production-control` — constrained/safe RL for operational control where feasibility and safety matter.

## Electrification and fleet planning

- `electric-bus-charging-fleet-planning-optimization` — integrated charging/fleet planning.
- `fleet-decarbonization-optimizer` — longer-horizon fleet-transition/decarbonization decisions.

## Renewable layout and design

- `wind-farm-layout-optimization` — discrete candidate-site MILP for wind-farm layout.
- `wind-farm-layout-optimizer` — heuristic/randomized-greedy wind-layout optimizer with a different computational approach.

These two wind-farm repositories should remain separate because one is an exact mathematical-programming formulation and the other is a heuristic layout generator.

## Energy-aware production

- `energy-aware-production-scheduling-ga-java` — production scheduling with energy-related objectives/constraints using a genetic algorithm.

## Portfolio rule

The word `energy` spans several distinct OR problem classes: commitment, storage, control, infrastructure planning, fleet transition, and spatial layout. Those distinctions are more important than the shared domain label, so this series is for navigation and comparison rather than consolidation.
