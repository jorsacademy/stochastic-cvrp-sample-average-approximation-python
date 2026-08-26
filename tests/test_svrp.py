import math
import unittest

import numpy as np

from stochastic_cvrp_saa import (
    Customer,
    StochasticCVRPSAA,
    create_random_instance,
)


class StochasticCVRPTests(unittest.TestCase):
    def test_recourse_cost_does_not_double_count_depot_return(self):
        customers = [
            Customer(1, 10.0, 0.0, 8.0, 0.0),
            Customer(2, 20.0, 0.0, 8.0, 0.0),
        ]
        solver = StochasticCVRPSAA(
            customers,
            num_vehicles=1,
            vehicle_capacity=16.0,
            depot_location=(0.0, 0.0),
            num_scenarios=1,
            seed=1,
            failure_penalty=100.0,
        )
        scenario = np.array([0.0, 8.0, 8.0])

        cost, failures = solver.route_scenario_cost([0, 1, 2, 0], scenario)
        self.assertTrue(math.isclose(cost, 40.0, abs_tol=1e-12))
        self.assertEqual(failures, 0)

        solver.Q = 10.0
        cost, failures = solver.route_scenario_cost([0, 1, 2, 0], scenario)
        self.assertTrue(math.isclose(cost, 160.0, abs_tol=1e-12))
        self.assertEqual(failures, 1)

    def test_reference_solution_is_feasible(self):
        solver = create_random_instance(
            num_customers=15,
            num_vehicles=4,
            vehicle_capacity=60.0,
            seed=42,
        )
        result = solver.solve(
            max_iterations=25,
            max_neighbors=40,
            out_of_sample_scenarios=500,
        )
        self.assertTrue(solver.validate_routes(result.routes))
        self.assertLessEqual(len(result.routes), 4)
        self.assertTrue(math.isfinite(result.saa_cost))
        self.assertTrue(math.isfinite(result.out_of_sample_cost))
        self.assertGreaterEqual(result.failure_probability, 0.0)
        self.assertLessEqual(result.failure_probability, 1.0)

    def test_reproducibility(self):
        a_solver = create_random_instance(10, 3, 50.0, 7)
        b_solver = create_random_instance(10, 3, 50.0, 7)

        a = a_solver.solve(
            max_iterations=10,
            max_neighbors=20,
            out_of_sample_scenarios=300,
        )
        b = b_solver.solve(
            max_iterations=10,
            max_neighbors=20,
            out_of_sample_scenarios=300,
        )

        self.assertEqual(a.routes, b.routes)
        self.assertTrue(math.isclose(a.saa_cost, b.saa_cost, abs_tol=1e-12))
        self.assertTrue(
            math.isclose(
                a.out_of_sample_cost,
                b.out_of_sample_cost,
                abs_tol=1e-12,
            )
        )

    def test_total_mean_capacity_infeasibility_rejected(self):
        customers = [
            Customer(1, 10.0, 0.0, 8.0, 1.0),
            Customer(2, 20.0, 0.0, 8.0, 1.0),
            Customer(3, 30.0, 0.0, 8.0, 1.0),
        ]
        with self.assertRaises(ValueError):
            StochasticCVRPSAA(
                customers,
                num_vehicles=2,
                vehicle_capacity=10.0,
                num_scenarios=20,
            )

    def test_route_validation_requires_exact_customer_coverage(self):
        solver = create_random_instance(6, 2, 50.0, 3)
        self.assertFalse(solver.validate_routes([[0, 1, 2, 0], [0, 3, 4, 0]]))

    def test_out_of_sample_scenarios_are_independent_from_training_sample(self):
        solver = create_random_instance(8, 3, 50.0, 9)
        validation = solver._generate_scenarios(
            solver.seed + 1_000_003,
            100,
        )
        self.assertFalse(
            np.array_equal(
                solver.training_scenarios[: min(100, len(solver.training_scenarios))],
                validation[: min(100, len(validation))],
            )
        )


if __name__ == "__main__":
    unittest.main()
