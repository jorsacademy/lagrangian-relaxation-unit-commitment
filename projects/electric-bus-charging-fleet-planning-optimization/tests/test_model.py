import math
import unittest

import numpy as np

import electric_bus_charging_fleet_optimization as model


class ElectricBusOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routes, cls.stations, cls.endpoints, cls.demand, cls.tariffs = model.build_data()
        cls.selected, cls.assignments, cls.station_metrics = model.solve_station_location(
            cls.routes, cls.stations, cls.endpoints, cls.demand
        )
        cls.served, cls.route_results, cls.bus_stats, cls.fleet_metrics = model.solve_fleet_network(
            cls.routes, cls.demand, cls.tariffs
        )

    def test_station_solution_is_exactly_four_available_sites(self):
        self.assertEqual(len(self.selected), model.N_STATIONS_TO_BUILD)
        chosen = self.selected["station_id"].astype(int).tolist()
        for station_id in chosen:
            self.assertTrue(bool(self.stations.loc[station_id - 1, "land_availability"]))

    def test_route_assignments_use_open_stations(self):
        chosen = set(self.selected["station_id"].astype(int))
        assigned = set(self.assignments["assigned_station_id"].astype(int))
        self.assertTrue(assigned.issubset(chosen))
        self.assertEqual(len(self.assignments), model.N_ROUTES)

    def test_demand_is_never_overserved(self):
        self.assertTrue(np.all(self.served >= 0))
        self.assertTrue(np.all(self.served <= self.demand))
        self.assertEqual(int(self.served.sum()), self.fleet_metrics["served_trips"])

    def test_minimum_route_coverage_policy(self):
        requested = self.demand.sum(axis=0)
        served = self.served.sum(axis=0)
        required = np.ceil(model.MIN_ROUTE_COVERAGE * requested).astype(int)
        self.assertTrue(np.all(served >= required))

    def test_bus_duty_decomposition(self):
        self.assertEqual(len(self.bus_stats), model.N_BUSES)
        self.assertTrue((self.bus_stats["utilization_pct"] <= 100.0 + 1e-9).all())
        self.assertEqual(int(self.bus_stats["trips"].sum()), self.fleet_metrics["served_trips"])

    def test_charging_capacity(self):
        self.assertLessEqual(
            self.fleet_metrics["max_simultaneous_charging_buses"],
            model.N_FAST_CHARGERS,
        )

    def test_reference_optimum_is_reproducible(self):
        self.assertEqual(self.selected["station_id"].astype(int).tolist(), [2, 3, 7, 10])
        self.assertEqual(self.fleet_metrics["served_trips"], 143)
        self.assertTrue(math.isclose(
            self.station_metrics["total_installation_cost"],
            1003328.8112196852,
            rel_tol=0,
            abs_tol=1e-6,
        ))


if __name__ == "__main__":
    unittest.main()
