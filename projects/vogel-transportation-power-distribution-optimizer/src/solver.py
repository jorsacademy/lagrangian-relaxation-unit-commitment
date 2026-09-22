from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

EPSILON = 1e-9


@dataclass(frozen=True)
class TransportationProblem:
    costs: tuple[tuple[float, ...], ...]
    supply: tuple[float, ...]
    demand: tuple[float, ...]
    forbidden: frozenset[tuple[int, int]] = frozenset()

    def validate(self) -> None:
        if not self.costs or not self.costs[0]:
            raise ValueError("Cost matrix must be non-empty.")

        rows = len(self.costs)
        cols = len(self.costs[0])

        if len(self.supply) != rows or len(self.demand) != cols:
            raise ValueError("Supply/demand dimensions do not match the cost matrix.")
        if any(len(row) != cols for row in self.costs):
            raise ValueError("Cost matrix must be rectangular.")
        if any(value < 0 for value in self.supply + self.demand):
            raise ValueError("Supply and demand must be non-negative.")
        if abs(sum(self.supply) - sum(self.demand)) > EPSILON:
            raise ValueError("Problem must be balanced.")

        for i, j in self.forbidden:
            if not (0 <= i < rows and 0 <= j < cols):
                raise ValueError("Forbidden route is outside the cost matrix.")


def _penalty(values: list[float]) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    return values[1] - values[0]


def vogel_approximation(problem: TransportationProblem) -> list[list[float]]:
    """Return a basic feasible solution using Vogel's Approximation Method."""
    problem.validate()
    m, n = len(problem.supply), len(problem.demand)
    remaining_supply = list(problem.supply)
    remaining_demand = list(problem.demand)
    allocation = [[0.0] * n for _ in range(m)]

    while any(value > EPSILON for value in remaining_supply):
        candidates: list[tuple[float, float, str, int]] = []

        for i in range(m):
            if remaining_supply[i] <= EPSILON:
                continue
            feasible_costs = [
                problem.costs[i][j]
                for j in range(n)
                if remaining_demand[j] > EPSILON and (i, j) not in problem.forbidden
            ]
            if feasible_costs:
                candidates.append((_penalty(feasible_costs), -min(feasible_costs), "row", i))

        for j in range(n):
            if remaining_demand[j] <= EPSILON:
                continue
            feasible_costs = [
                problem.costs[i][j]
                for i in range(m)
                if remaining_supply[i] > EPSILON and (i, j) not in problem.forbidden
            ]
            if feasible_costs:
                candidates.append((_penalty(feasible_costs), -min(feasible_costs), "col", j))

        if not candidates:
            raise ValueError("No feasible allocation can satisfy the remaining supply and demand.")

        _, _, kind, index = max(candidates)

        if kind == "row":
            i = index
            feasible_columns = [
                j
                for j in range(n)
                if remaining_demand[j] > EPSILON and (i, j) not in problem.forbidden
            ]
            j = min(feasible_columns, key=lambda column: (problem.costs[i][column], column))
        else:
            j = index
            feasible_rows = [
                i
                for i in range(m)
                if remaining_supply[i] > EPSILON and (i, j) not in problem.forbidden
            ]
            i = min(feasible_rows, key=lambda row: (problem.costs[row][j], row))

        amount = min(remaining_supply[i], remaining_demand[j])
        allocation[i][j] += amount
        remaining_supply[i] -= amount
        remaining_demand[j] -= amount

    if any(value > EPSILON for value in remaining_demand):
        raise ValueError("Demand remains unsatisfied.")

    return allocation


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        self.parent[right_root] = left_root
        return True


def _build_basis(
    problem: TransportationProblem,
    allocation: Sequence[Sequence[float]],
) -> set[tuple[int, int]]:
    """Build a spanning-tree basis, adding zero basic cells for degeneracy."""
    m, n = len(problem.supply), len(problem.demand)
    basis: set[tuple[int, int]] = set()
    components = _DisjointSet(m + n)

    positive_cells = [
        (i, j)
        for i in range(m)
        for j in range(n)
        if allocation[i][j] > EPSILON
    ]

    for i, j in positive_cells:
        if (i, j) in problem.forbidden:
            raise ValueError("Initial allocation uses a forbidden route.")
        if not components.union(i, m + j):
            raise ValueError("Positive allocations contain a cycle and are not a basic solution.")
        basis.add((i, j))

    for i in range(m):
        for j in range(n):
            if len(basis) == m + n - 1:
                break
            if (i, j) in basis or (i, j) in problem.forbidden:
                continue
            if components.union(i, m + j):
                basis.add((i, j))
        if len(basis) == m + n - 1:
            break

    if len(basis) != m + n - 1:
        raise ValueError("Allowed routes cannot form a connected transportation basis.")

    return basis


def _potentials(
    problem: TransportationProblem,
    basis: set[tuple[int, int]],
) -> tuple[list[float], list[float]]:
    """Compute MODI row and column potentials for the current basis."""
    m, n = len(problem.supply), len(problem.demand)
    row_potentials: list[float | None] = [None] * m
    column_potentials: list[float | None] = [None] * n
    row_potentials[0] = 0.0

    changed = True
    while changed:
        changed = False
        for i, j in basis:
            cost = problem.costs[i][j]
            if row_potentials[i] is not None and column_potentials[j] is None:
                column_potentials[j] = cost - row_potentials[i]
                changed = True
            elif column_potentials[j] is not None and row_potentials[i] is None:
                row_potentials[i] = cost - column_potentials[j]
                changed = True

    if any(value is None for value in row_potentials + column_potentials):
        raise ValueError("Transportation basis is disconnected.")

    return (
        [float(value) for value in row_potentials],
        [float(value) for value in column_potentials],
    )


def _tree_path_edges(
    rows: int,
    columns: int,
    basis: set[tuple[int, int]],
    start_node: int,
    end_node: int,
) -> list[tuple[int, int]]:
    """Return the basis edges on the unique path between two bipartite nodes."""
    adjacency: list[list[tuple[int, tuple[int, int]]]] = [
        [] for _ in range(rows + columns)
    ]

    for i, j in basis:
        row_node = i
        column_node = rows + j
        adjacency[row_node].append((column_node, (i, j)))
        adjacency[column_node].append((row_node, (i, j)))

    previous: dict[int, tuple[int | None, tuple[int, int] | None]] = {
        start_node: (None, None)
    }
    queue = [start_node]

    for node in queue:
        if node == end_node:
            break
        for next_node, edge in adjacency[node]:
            if next_node not in previous:
                previous[next_node] = (node, edge)
                queue.append(next_node)

    if end_node not in previous:
        raise ValueError("No cycle path exists in the current basis.")

    path: list[tuple[int, int]] = []
    current = end_node
    while previous[current][0] is not None:
        parent, edge = previous[current]
        assert parent is not None and edge is not None
        path.append(edge)
        current = parent

    path.reverse()
    return path


def transportation_simplex(
    problem: TransportationProblem,
    initial_allocation: Sequence[Sequence[float]] | None = None,
    max_iterations: int = 10_000,
) -> list[list[float]]:
    """Optimize a balanced transportation problem with the MODI/transportation-simplex method.

    If no initial allocation is supplied, Vogel's Approximation Method is used.
    Forbidden routes are never considered as basic or entering cells.
    """
    problem.validate()
    allocation = [
        list(row)
        for row in (
            initial_allocation
            if initial_allocation is not None
            else vogel_approximation(problem)
        )
    ]
    assert_feasible(problem, allocation)

    m, n = len(problem.supply), len(problem.demand)
    basis = _build_basis(problem, allocation)

    for _ in range(max_iterations):
        row_potentials, column_potentials = _potentials(problem, basis)

        reduced_costs = [
            (
                problem.costs[i][j] - row_potentials[i] - column_potentials[j],
                i,
                j,
            )
            for i in range(m)
            for j in range(n)
            if (i, j) not in basis and (i, j) not in problem.forbidden
        ]

        if not reduced_costs:
            return allocation

        reduced_cost, entering_i, entering_j = min(
            reduced_costs,
            key=lambda item: (item[0], item[1], item[2]),
        )

        if reduced_cost >= -EPSILON:
            return allocation

        path = _tree_path_edges(
            m,
            n,
            basis,
            entering_i,
            m + entering_j,
        )

        minus_cells: list[tuple[int, int]] = []
        plus_cells: list[tuple[int, int]] = []
        for index, cell in enumerate(reversed(path)):
            if index % 2 == 0:
                minus_cells.append(cell)
            else:
                plus_cells.append(cell)

        theta = min(allocation[i][j] for i, j in minus_cells)
        allocation[entering_i][entering_j] += theta

        for i, j in plus_cells:
            allocation[i][j] += theta
        for i, j in minus_cells:
            allocation[i][j] -= theta

        leaving_cell = min(
            (
                (i, j)
                for i, j in minus_cells
                if allocation[i][j] <= EPSILON
            ),
            key=lambda cell: (cell[0], cell[1]),
        )

        for i, j in minus_cells + plus_cells + [(entering_i, entering_j)]:
            if abs(allocation[i][j]) <= EPSILON:
                allocation[i][j] = 0.0

        basis.add((entering_i, entering_j))
        basis.remove(leaving_cell)

    raise RuntimeError("Transportation simplex did not converge within max_iterations.")


def transportation_cost(
    problem: TransportationProblem,
    allocation: Sequence[Sequence[float]],
) -> float:
    return sum(
        allocation[i][j] * problem.costs[i][j]
        for i in range(len(problem.supply))
        for j in range(len(problem.demand))
    )


def assert_feasible(
    problem: TransportationProblem,
    allocation: Sequence[Sequence[float]],
) -> None:
    problem.validate()
    m, n = len(problem.supply), len(problem.demand)

    if len(allocation) != m or any(len(row) != n for row in allocation):
        raise AssertionError("Allocation dimensions are invalid.")

    for i in range(m):
        if any(value < -EPSILON for value in allocation[i]):
            raise AssertionError(f"Negative allocation found in row {i}.")
        if abs(sum(allocation[i]) - problem.supply[i]) > EPSILON:
            raise AssertionError(f"Supply constraint violated for row {i}.")

    for j in range(n):
        if abs(sum(allocation[i][j] for i in range(m)) - problem.demand[j]) > EPSILON:
            raise AssertionError(f"Demand constraint violated for column {j}.")

    for i, j in problem.forbidden:
        if abs(allocation[i][j]) > EPSILON:
            raise AssertionError(f"Forbidden route ({i}, {j}) was used.")


def example_problem() -> TransportationProblem:
    """Power-distribution example from the assignment, with costs expressed in USD."""
    return TransportationProblem(
        costs=(
            (600.0, 700.0, 400.0),
            (320.0, 300.0, 350.0),
            (500.0, 480.0, 450.0),
            (1000.0, 1000.0, 0.0),
        ),
        supply=(25.0, 40.0, 30.0, 13.0),
        demand=(36.0, 42.0, 30.0),
        forbidden=frozenset({(3, 2)}),
    )
