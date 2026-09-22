from src.solver import (
    assert_feasible,
    example_problem,
    transportation_cost,
    transportation_simplex,
    vogel_approximation,
)


def print_allocation(title: str, allocation: list[list[float]]) -> None:
    labels = ["Plant 1", "Plant 2", "Plant 3", "External Grid"]
    cities = ["City 1", "City 2", "City 3"]

    print(title)
    print(f"{'Source':<15}" + "".join(f"{city:>12}" for city in cities))
    for label, row in zip(labels, allocation):
        print(f"{label:<15}" + "".join(f"{value:>12.0f}" for value in row))


def main() -> None:
    problem = example_problem()

    initial = vogel_approximation(problem)
    optimized = transportation_simplex(problem, initial)

    assert_feasible(problem, initial)
    assert_feasible(problem, optimized)

    print_allocation("Vogel initial allocation (million kWh):", initial)
    print(f"Initial cost: ${transportation_cost(problem, initial):,.2f}\n")

    print_allocation("Transportation-simplex optimum (million kWh):", optimized)
    print(f"Optimal cost: ${transportation_cost(problem, optimized):,.2f}")


if __name__ == "__main__":
    main()
