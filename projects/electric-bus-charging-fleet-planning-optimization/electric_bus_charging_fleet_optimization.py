"""
Electric Bus Charging & Fleet Planning — Exact MILP Models

This script replaces the earlier greedy/random demonstration with:
1) an exact fixed-charge p-median-style charging-station location MILP;
2) an exact time-expanded integer network-flow MILP for fleet operations.

The operational MILP uses a conservative 10-kWh battery-state discretization.
Every integer unit of flow represents one bus and can be decomposed into
individual bus duties, so duplicate trip assignment is impossible by design.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix


SEED = 42
N_ROUTES = 8
N_BUSES = 12
N_STATIONS = 10
N_STATIONS_TO_BUILD = 4
HOURS = 24

BATTERY_KWH = 250
ENERGY_KWH_PER_KM = 1.2
FAST_CHARGE_KW = 150
CHARGING_EFFICIENCY = 0.95
N_FAST_CHARGERS = 4
MIN_SOC_KWH = 20
SOC_STEP_KWH = 10

AVG_SPEED_KMH = 30.0
MAINTENANCE_COST_PER_KM = 0.15
DRIVER_COST_PER_HOUR = 25.0

# Policy constraint used to prevent the optimizer from starving low-demand routes.
MIN_ROUTE_COVERAGE = 0.15

# Facility-location multi-objective weights.
CAPEX_WEIGHT = 0.35
ACCESS_WEIGHT = 0.65


def build_data():
    np.random.seed(SEED)

    routes = pd.DataFrame({
        "route_id": range(1, N_ROUTES + 1),
        "distance_km": np.random.uniform(15, 45, N_ROUTES),
        "avg_passengers": np.random.randint(30, 120, N_ROUTES),
        "frequency_per_hour": np.random.randint(2, 6, N_ROUTES),
    })

    # Preserve the original data-generation sequence.
    np.random.seed(SEED)
    stations = pd.DataFrame({
        "station_id": range(1, N_STATIONS + 1),
        "x_coord": np.random.uniform(0, 100, N_STATIONS),
        "y_coord": np.random.uniform(0, 100, N_STATIONS),
        "installation_cost": np.random.uniform(200000, 500000, N_STATIONS),
        "land_availability": np.random.choice(
            [True, False], N_STATIONS, p=[0.8, 0.2]
        ),
    })

    endpoints = pd.DataFrame({
        "route_id": range(1, N_ROUTES + 1),
        "start_x": np.random.uniform(10, 90, N_ROUTES),
        "start_y": np.random.uniform(10, 90, N_ROUTES),
        "end_x": np.random.uniform(10, 90, N_ROUTES),
        "end_y": np.random.uniform(10, 90, N_ROUTES),
    })

    multiplier = np.ones(HOURS)
    peak_hours = [7, 8, 9, 16, 17, 18]
    multiplier[peak_hours] = 2.0
    multiplier[[22, 23, 0, 1, 2, 3, 4, 5]] = 0.3

    demand = np.zeros((HOURS, N_ROUTES), dtype=int)
    for h in range(HOURS):
        for r in range(N_ROUTES):
            demand[h, r] = int(routes.loc[r, "frequency_per_hour"] * multiplier[h])

    tariffs = np.full(HOURS, 0.18)
    tariffs[[22, 23, 0, 1, 2, 3, 4, 5]] = 0.08
    tariffs[peak_hours] = 0.28

    return routes, stations, endpoints, demand, tariffs


def solve_station_location(routes, stations, endpoints, demand):
    centers = np.column_stack((
        (endpoints["start_x"] + endpoints["end_x"]) / 2,
        (endpoints["start_y"] + endpoints["end_y"]) / 2,
    ))
    station_xy = stations[["x_coord", "y_coord"]].to_numpy()

    distance = np.linalg.norm(
        centers[:, None, :] - station_xy[None, :, :], axis=2
    )

    daily_route_passenger_demand = (
        demand.sum(axis=0) * routes["avg_passengers"].to_numpy()
    )

    # y_s: station open
    # z_rs: route r assigned to station s
    n_y = N_STATIONS
    n_z = N_ROUTES * N_STATIONS
    n_vars = n_y + n_z

    capex = stations["installation_cost"].to_numpy()
    weighted_distance = distance * daily_route_passenger_demand[:, None]

    objective = np.zeros(n_vars)
    objective[:n_y] = CAPEX_WEIGHT * capex / capex.sum()
    objective[n_y:] = (
        ACCESS_WEIGHT * weighted_distance / weighted_distance.sum()
    ).ravel()

    integrality = np.ones(n_vars)
    lower = np.zeros(n_vars)
    upper = np.ones(n_vars)

    for s, available in enumerate(stations["land_availability"]):
        if not available:
            upper[s] = 0

    rows, lbs, ubs = [], [], []

    row = np.zeros(n_vars)
    row[:n_y] = 1
    rows.append(row)
    lbs.append(N_STATIONS_TO_BUILD)
    ubs.append(N_STATIONS_TO_BUILD)

    for r in range(N_ROUTES):
        row = np.zeros(n_vars)
        row[n_y + r * N_STATIONS:n_y + (r + 1) * N_STATIONS] = 1
        rows.append(row)
        lbs.append(1)
        ubs.append(1)

    for r in range(N_ROUTES):
        for s in range(N_STATIONS):
            row = np.zeros(n_vars)
            row[n_y + r * N_STATIONS + s] = 1
            row[s] = -1
            rows.append(row)
            lbs.append(-np.inf)
            ubs.append(0)

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            csr_matrix(np.vstack(rows)),
            np.array(lbs),
            np.array(ubs),
        ),
        options={"mip_rel_gap": 0.0},
    )

    if not result.success:
        raise RuntimeError(f"Station MILP failed: {result.message}")

    open_mask = np.rint(result.x[:n_y]).astype(int)
    selected = np.where(open_mask == 1)[0]
    assignment = result.x[n_y:].reshape(N_ROUTES, N_STATIONS).argmax(axis=1)

    selected_table = stations.loc[
        selected,
        ["station_id", "x_coord", "y_coord", "installation_cost"],
    ].copy()

    route_assignment = pd.DataFrame({
        "route_id": routes["route_id"],
        "assigned_station_id": assignment + 1,
        "distance_to_station_km": distance[np.arange(N_ROUTES), assignment],
    })

    metrics = {
        "selected_indices": selected,
        "total_installation_cost": capex[selected].sum(),
        "average_route_station_distance_km": (
            distance[np.arange(N_ROUTES), assignment].mean()
        ),
        "weighted_average_route_station_distance_km": np.average(
            distance[np.arange(N_ROUTES), assignment],
            weights=daily_route_passenger_demand,
        ),
        "solver_message": result.message,
    }

    return selected_table, route_assignment, metrics


@dataclass
class Arc:
    u: tuple[int, int]
    v: tuple[int, int]
    kind: str
    hour: int
    route: int | None = None
    grid_kwh: float = 0.0
    monetary_cost: float = 0.0
    passengers: float = 0.0


def solve_fleet_network(routes, demand, tariffs):
    durations = np.ceil(routes["distance_km"].to_numpy() / AVG_SPEED_KMH).astype(int)
    actual_energy = routes["distance_km"].to_numpy() * ENERGY_KWH_PER_KM

    # Conservative rounding: a route consumes the next-highest SOC state.
    discrete_energy = (
        np.ceil(actual_energy / SOC_STEP_KWH) * SOC_STEP_KWH
    ).astype(int)

    soc_levels = np.arange(MIN_SOC_KWH, BATTERY_KWH + 1, SOC_STEP_KWH)
    soc_set = set(soc_levels.tolist())

    # Net battery gains allowed in one charging hour. 140 kWh is below
    # 150 kW * 0.95 = 142.5 kWh net.
    charge_gains = [50, 100, 140]

    arcs: list[Arc] = []
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    trip_groups = defaultdict(list)
    charge_groups = defaultdict(list)

    def add_arc(arc: Arc):
        idx = len(arcs)
        arcs.append(arc)
        outgoing[arc.u].append(idx)
        incoming[arc.v].append(idx)
        if arc.kind == "trip":
            trip_groups[(arc.hour, arc.route)].append(idx)
        elif arc.kind == "charge":
            charge_groups[arc.hour].append(idx)

    for h in range(HOURS):
        for soc in soc_levels:
            soc = int(soc)

            add_arc(Arc((h, soc), (h + 1, soc), "idle", h))

            for gain in charge_gains:
                next_soc = min(BATTERY_KWH, soc + gain)
                if next_soc > soc and next_soc in soc_set:
                    net_gain = next_soc - soc
                    grid_kwh = net_gain / CHARGING_EFFICIENCY
                    add_arc(Arc(
                        (h, soc),
                        (h + 1, int(next_soc)),
                        "charge",
                        h,
                        grid_kwh=grid_kwh,
                        monetary_cost=grid_kwh * tariffs[h],
                    ))

            for r in range(N_ROUTES):
                if h + durations[r] > HOURS:
                    continue
                next_soc = soc - discrete_energy[r]
                if next_soc < MIN_SOC_KWH or next_soc not in soc_set:
                    continue

                trip_cost = (
                    routes.loc[r, "distance_km"] * MAINTENANCE_COST_PER_KM
                    + durations[r] * DRIVER_COST_PER_HOUR
                )
                add_arc(Arc(
                    (h, soc),
                    (h + int(durations[r]), int(next_soc)),
                    "trip",
                    h,
                    route=r,
                    monetary_cost=float(trip_cost),
                    passengers=float(routes.loc[r, "avg_passengers"]),
                ))

    n_vars = len(arcs)
    integrality = np.ones(n_vars)
    lower = np.zeros(n_vars)
    upper = np.full(n_vars, N_BUSES, dtype=float)

    for (h, r), indices in trip_groups.items():
        for i in indices:
            upper[i] = min(N_BUSES, demand[h, r])

    # Lexicographic-equivalent scalarization:
    # 1) maximize number of served trips,
    # 2) among those, prefer higher passenger throughput,
    # 3) among those, minimize cost.
    objective = np.zeros(n_vars)
    for i, arc in enumerate(arcs):
        if arc.kind == "trip":
            objective[i] = (
                -1000000.0
                - arc.passengers
                + 1e-3 * arc.monetary_cost
            )
        elif arc.kind == "charge":
            objective[i] = 1e-3 * arc.monetary_cost

    nodes = [
        (h, int(soc))
        for h in range(HOURS + 1)
        for soc in soc_levels
    ]
    source = (0, BATTERY_KWH)
    sink = (HOURS, BATTERY_KWH)

    n_constraints = (
        len(nodes)
        + HOURS * N_ROUTES
        + HOURS
        + N_ROUTES
    )

    A = lil_matrix((n_constraints, n_vars))
    lbs = np.full(n_constraints, -np.inf)
    ubs = np.full(n_constraints, np.inf)
    row = 0

    # Integer bus flow conservation.
    for node in nodes:
        for i in outgoing.get(node, []):
            A[row, i] += 1
        for i in incoming.get(node, []):
            A[row, i] -= 1

        rhs = N_BUSES if node == source else (-N_BUSES if node == sink else 0)
        lbs[row] = rhs
        ubs[row] = rhs
        row += 1

    # Each route-hour demand request can be served at most once.
    for h in range(HOURS):
        for r in range(N_ROUTES):
            for i in trip_groups.get((h, r), []):
                A[row, i] = 1
            ubs[row] = demand[h, r]
            row += 1

    # At most four buses can occupy fast chargers in any hour.
    for h in range(HOURS):
        for i in charge_groups.get(h, []):
            A[row, i] = 1
        ubs[row] = N_FAST_CHARGERS
        row += 1

    # Network-service policy: every route must receive at least 15% of
    # its daily requested trips. This is a hard constraint, not a KPI claim.
    total_route_requests = demand.sum(axis=0)
    for r in range(N_ROUTES):
        for h in range(HOURS):
            for i in trip_groups.get((h, r), []):
                A[row, i] = -1
        ubs[row] = -math.ceil(MIN_ROUTE_COVERAGE * total_route_requests[r])
        row += 1

    assert row == n_constraints

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A.tocsr(), lbs, ubs),
        options={"mip_rel_gap": 0.0},
    )

    if not result.success:
        raise RuntimeError(f"Fleet MILP failed: {result.message}")

    flow = np.rint(result.x).astype(int)

    served = np.zeros((HOURS, N_ROUTES), dtype=int)
    charge_events = np.zeros(HOURS, dtype=int)
    grid_energy = np.zeros(HOURS)

    for i, value in enumerate(flow):
        if value <= 0:
            continue
        arc = arcs[i]
        if arc.kind == "trip":
            served[arc.hour, arc.route] += value
        elif arc.kind == "charge":
            charge_events[arc.hour] += value
            grid_energy[arc.hour] += value * arc.grid_kwh

    # Exact path decomposition: each path is one bus duty.
    residual = flow.copy()
    paths = []

    for bus in range(N_BUSES):
        node = source
        path = []
        while node != sink:
            candidates = [i for i in outgoing[node] if residual[i] > 0]
            if not candidates:
                raise RuntimeError(f"Path decomposition failed at {node}")

            # Deterministic order only; does not change the optimal flow.
            priority = {"trip": 0, "charge": 1, "idle": 2}
            i = min(candidates, key=lambda k: priority[arcs[k].kind])
            residual[i] -= 1
            path.append(i)
            node = arcs[i].v

        paths.append(path)

    if residual.sum() != 0:
        raise RuntimeError("Residual flow remained after duty decomposition.")

    bus_rows = []
    for bus_id, path in enumerate(paths, start=1):
        trip_arcs = [arcs[i] for i in path if arcs[i].kind == "trip"]
        charge_arcs = [arcs[i] for i in path if arcs[i].kind == "charge"]

        distance = sum(
            routes.loc[a.route, "distance_km"] for a in trip_arcs
        )
        occupied_hours = sum(durations[a.route] for a in trip_arcs)
        grid_kwh = sum(a.grid_kwh for a in charge_arcs)

        bus_rows.append({
            "bus_id": bus_id,
            "trips": len(trip_arcs),
            "distance_km": distance,
            "occupied_hours": occupied_hours,
            "utilization_pct": 100 * occupied_hours / HOURS,
            "grid_charge_kwh": grid_kwh,
        })

    bus_stats = pd.DataFrame(bus_rows)

    served_by_route = served.sum(axis=0)
    route_results = pd.DataFrame({
        "route_id": routes["route_id"],
        "requested_trips": total_route_requests,
        "served_trips": served_by_route,
        "coverage_pct": 100 * served_by_route / total_route_requests,
    })

    total_distance = bus_stats["distance_km"].sum()
    occupied_hours = bus_stats["occupied_hours"].sum()

    electricity_cost = 0.0
    for i, value in enumerate(flow):
        if value > 0 and arcs[i].kind == "charge":
            electricity_cost += value * arcs[i].monetary_cost

    maintenance_cost = total_distance * MAINTENANCE_COST_PER_KM
    driver_cost = occupied_hours * DRIVER_COST_PER_HOUR
    total_operating_cost = electricity_cost + maintenance_cost + driver_cost

    metrics = {
        "requested_trips": int(demand.sum()),
        "served_trips": int(served.sum()),
        "trip_coverage_pct": 100 * served.sum() / demand.sum(),
        "passenger_weighted_coverage_pct": (
            100
            * (served * routes["avg_passengers"].to_numpy()).sum()
            / (demand * routes["avg_passengers"].to_numpy()).sum()
        ),
        "fleet_utilization_pct": 100 * occupied_hours / (N_BUSES * HOURS),
        "total_distance_km": total_distance,
        "actual_traction_energy_kwh": total_distance * ENERGY_KWH_PER_KM,
        "grid_charge_energy_kwh": grid_energy.sum(),
        "electricity_cost": electricity_cost,
        "maintenance_cost": maintenance_cost,
        "driver_cost": driver_cost,
        "total_operating_cost": total_operating_cost,
        "max_simultaneous_charging_buses": int(charge_events.max()),
        "solver_message": result.message,
    }

    return served, route_results, bus_stats, metrics


def main():
    routes, stations, endpoints, demand, tariffs = build_data()

    selected, assignments, station_metrics = solve_station_location(
        routes, stations, endpoints, demand
    )
    served, route_results, bus_stats, fleet_metrics = solve_fleet_network(
        routes, demand, tariffs
    )

    print("=" * 80)
    print("EXACT ELECTRIC BUS OPTIMIZATION")
    print("=" * 80)

    print("\nSELECTED CHARGING STATIONS")
    print(selected.to_string(index=False))
    print(
        f"\nTotal installation cost: "
        f"${station_metrics['total_installation_cost']:,.2f}"
    )
    print(
        f"Weighted average route-station distance: "
        f"{station_metrics['weighted_average_route_station_distance_km']:.2f} km"
    )
    print(f"Station solver: {station_metrics['solver_message']}")

    print("\nROUTE SERVICE RESULTS")
    print(route_results.to_string(index=False, formatters={
        "coverage_pct": lambda x: f"{x:.1f}%"
    }))

    print("\nFLEET RESULTS")
    for key, value in fleet_metrics.items():
        if key == "solver_message":
            continue
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value}")
    print(f"Fleet solver: {fleet_metrics['solver_message']}")

    print("\nBUS DUTY SUMMARY")
    print(bus_stats.to_string(index=False, formatters={
        "distance_km": lambda x: f"{x:.1f}",
        "utilization_pct": lambda x: f"{x:.1f}%",
        "grid_charge_kwh": lambda x: f"{x:.1f}",
    }))


if __name__ == "__main__":
    main()
