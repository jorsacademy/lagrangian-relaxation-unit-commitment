import unittest

from src.solver import (
    TransportationProblem,
    assert_feasible,
    example_problem,
    transportation_cost,
    transportation_simplex,
    vogel_approximation,
)


class TransportationSolverTests(unittest.TestCase):
    def test_assignment_example_vogel_solution_is_optimal(self):
        problem = example_problem()
        initial = vogel_approximation(problem)
        optimized = transportation_simplex(problem, initial)

        assert_feasible(problem, initial)
        assert_feasible(problem, optimized)

        expected = [
            [0.0, 0.0, 25.0],
            [0.0, 40.0, 0.0],
            [23.0, 2.0, 5.0],
            [13.0, 0.0, 0.0],
        ]
        self.assertEqual(initial, expected)
        self.assertEqual(optimized, expected)
        self.assertAlmostEqual(transportation_cost(problem, optimized), 49710.0)

    def test_simplex_improves_nonoptimal_vogel_solution(self):
        problem = TransportationProblem(
            costs=((28.0, 7.0, 1.0), (28.0, 18.0, 2.0), (28.0, 6.0, 4.0)),
            supply=(1.0, 17.0, 12.0),
            demand=(2.0, 12.0, 16.0),
        )

        initial = vogel_approximation(problem)
        optimized = transportation_simplex(problem, initial)

        assert_feasible(problem, optimized)
        self.assertAlmostEqual(transportation_cost(problem, initial), 160.0)
        self.assertAlmostEqual(transportation_cost(problem, optimized), 159.0)
        self.assertEqual(
            optimized,
            [
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 15.0],
                [0.0, 12.0, 0.0],
            ],
        )

    def test_degenerate_initial_solution_is_supported(self):
        problem = TransportationProblem(
            costs=((1.0, 3.0), (2.0, 1.0)),
            supply=(5.0, 5.0),
            demand=(5.0, 5.0),
        )
        optimized = transportation_simplex(problem)
        assert_feasible(problem, optimized)
        self.assertAlmostEqual(transportation_cost(problem, optimized), 10.0)

    def test_forbidden_external_grid_to_city_3(self):
        problem = example_problem()
        optimized = transportation_simplex(problem)
        self.assertEqual(optimized[3][2], 0.0)


if __name__ == "__main__":
    unittest.main()
