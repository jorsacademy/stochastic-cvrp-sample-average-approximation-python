from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    mean_demand: float
    std_demand: float


@dataclass
class SVRPResult:
    routes: List[List[int]]
    saa_cost: float
    out_of_sample_cost: float
    failure_probability: float
    expected_failures: float
    elapsed_time: float
    iterations: int
    status: str


class StochasticCVRPSAA:
    """
    Educational stochastic-demand CVRP with depot-restocking recourse.

    First-stage decision:
      - assign every customer exactly once to at most K planned routes
      - each planned route has expected (mean) demand <= vehicle capacity

    Second-stage recourse:
      - if realized cumulative demand would exceed capacity before serving the
        next customer, the vehicle returns to the depot, reloads, then travels
        from the depot to that customer and continues the planned sequence.

    Optimization:
      - fixed SAA scenarios (common random numbers)
      - permutation representation
      - dynamic-programming split decoder for route assignment
      - local search over swap / insertion / reversal neighborhoods

    This is a heuristic SAA implementation, not an exact stochastic-programming
    certificate.
    """

    def __init__(
        self,
        customers: Sequence[Customer],
        *,
        num_vehicles: int,
        vehicle_capacity: float,
        depot_location: Tuple[float, float] = (0.0, 0.0),
        num_scenarios: int = 200,
        seed: int = 42,
        failure_penalty: float = 100.0,
    ):
        if not customers:
            raise ValueError("at least one customer is required")
        if num_vehicles <= 0 or vehicle_capacity <= 0:
            raise ValueError("num_vehicles and vehicle_capacity must be positive")
        if num_scenarios <= 0:
            raise ValueError("num_scenarios must be positive")

        ids = [c.id for c in customers]
        if sorted(ids) != list(range(1, len(customers) + 1)):
            raise ValueError("customer ids must be consecutive 1..n")
        if any(c.mean_demand < 0 or c.std_demand < 0 for c in customers):
            raise ValueError("demand parameters must be nonnegative")
        if any(c.mean_demand > vehicle_capacity + 1e-12 for c in customers):
            raise ValueError("a customer's mean demand exceeds vehicle capacity")

        self.customers = list(customers)
        self.n = len(customers)
        self.k = int(num_vehicles)
        self.Q = float(vehicle_capacity)
        self.num_scenarios = int(num_scenarios)
        self.failure_penalty = float(failure_penalty)
        self.seed = int(seed)
        self.rng = random.Random(seed)

        if sum(c.mean_demand for c in customers) > self.k * self.Q + 1e-12:
            raise ValueError("total mean demand exceeds planned fleet capacity")

        coords = np.asarray(
            [depot_location] + [(c.x, c.y) for c in customers],
            dtype=float,
        )
        self.distance_matrix = np.linalg.norm(
            coords[:, None, :] - coords[None, :, :],
            axis=2,
        )

        self.training_scenarios = self._generate_scenarios(seed, self.num_scenarios)
        self.best_routes: Optional[List[List[int]]] = None

    def _generate_scenarios(self, seed: int, count: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        scenarios = np.zeros((count, self.n + 1), dtype=float)
        for customer in self.customers:
            draws = rng.normal(
                customer.mean_demand,
                customer.std_demand,
                size=count,
            )
            scenarios[:, customer.id] = np.maximum(0.0, draws)
        return scenarios

    def route_scenario_cost(
        self,
        route: List[int],
        scenario: np.ndarray,
    ) -> Tuple[float, int]:
        """
        Evaluate one planned route under one realized-demand scenario.

        On failure before customer j:
          current -> depot -> j
        replaces the direct current -> j arc.
        """
        if route[0] != 0 or route[-1] != 0:
            raise ValueError("route must start and end at depot")

        cost = 0.0
        failures = 0
        load = 0.0
        current = 0

        for customer_id in route[1:-1]:
            demand = float(scenario[customer_id])

            if load + demand > self.Q + 1e-12:
                cost += self.distance_matrix[current, 0]
                cost += self.distance_matrix[0, customer_id]
                cost += self.failure_penalty
                failures += 1
                load = demand
                current = customer_id
            else:
                cost += self.distance_matrix[current, customer_id]
                load += demand
                current = customer_id

        cost += self.distance_matrix[current, 0]
        return float(cost), failures

    def evaluate_routes(
        self,
        routes: List[List[int]],
        scenarios: np.ndarray,
    ) -> Tuple[float, float, float]:
        total_cost = 0.0
        total_failures = 0
        failed_scenarios = 0

        for scenario in scenarios:
            scenario_cost = 0.0
            scenario_failures = 0
            for route in routes:
                c, f = self.route_scenario_cost(route, scenario)
                scenario_cost += c
                scenario_failures += f
            total_cost += scenario_cost
            total_failures += scenario_failures
            if scenario_failures:
                failed_scenarios += 1

        m = len(scenarios)
        return (
            total_cost / m,
            total_failures / m,
            failed_scenarios / m,
        )

    def validate_routes(self, routes: List[List[int]]) -> bool:
        if not (1 <= len(routes) <= self.k):
            return False
        seen = []
        for route in routes:
            if len(route) < 3 or route[0] != 0 or route[-1] != 0:
                return False
            if any(c == 0 for c in route[1:-1]):
                return False
            mean_load = sum(
                self.customers[c - 1].mean_demand
                for c in route[1:-1]
            )
            if mean_load > self.Q + 1e-12:
                return False
            seen.extend(route[1:-1])
        return sorted(seen) == list(range(1, self.n + 1))

    def _split_permutation(
        self,
        permutation: Tuple[int, ...],
        scenarios: np.ndarray,
    ) -> Tuple[List[List[int]], float]:
        if sorted(permutation) != list(range(1, self.n + 1)):
            raise ValueError("invalid customer permutation")

        dp = [[math.inf] * (self.n + 1) for _ in range(self.k + 1)]
        pred = [[-1] * (self.n + 1) for _ in range(self.k + 1)]
        dp[0][0] = 0.0

        for r in range(1, self.k + 1):
            for i in range(self.n):
                if not math.isfinite(dp[r - 1][i]):
                    continue

                mean_load = 0.0
                segment = []
                for j in range(i, self.n):
                    c = permutation[j]
                    mean_load += self.customers[c - 1].mean_demand
                    if mean_load > self.Q + 1e-12:
                        break

                    segment.append(c)
                    route = [0, *segment, 0]
                    route_cost, _, _ = self.evaluate_routes([route], scenarios)
                    candidate = dp[r - 1][i] + route_cost

                    if candidate < dp[r][j + 1] - 1e-12:
                        dp[r][j + 1] = candidate
                        pred[r][j + 1] = i

        route_count = min(
            (
                r for r in range(1, self.k + 1)
                if math.isfinite(dp[r][self.n])
            ),
            key=lambda r: dp[r][self.n],
            default=None,
        )
        if route_count is None:
            raise ValueError("permutation cannot be split within fleet capacity")

        segments = []
        r = route_count
        end = self.n
        while r:
            start = pred[r][end]
            if start < 0:
                raise RuntimeError("invalid split predecessor")
            segments.append(permutation[start:end])
            end = start
            r -= 1
        segments.reverse()

        routes = [[0, *segment, 0] for segment in segments]
        if not self.validate_routes(routes):
            raise RuntimeError("split decoder produced infeasible routes")
        return routes, float(dp[route_count][self.n])

    def _nearest_neighbor_permutation(self) -> Tuple[int, ...]:
        remaining = set(range(1, self.n + 1))
        current = 0
        order = []
        while remaining:
            nxt = min(
                remaining,
                key=lambda j: (self.distance_matrix[current, j], j),
            )
            order.append(nxt)
            remaining.remove(nxt)
            current = nxt
        return tuple(order)

    def _neighbors(self, permutation: Tuple[int, ...], max_neighbors: int) -> List[Tuple[int, ...]]:
        neighbors = set()
        n = len(permutation)

        for _ in range(min(max_neighbors // 3 + 1, n * (n - 1) // 2)):
            i, j = self.rng.sample(range(n), 2)
            seq = list(permutation)
            seq[i], seq[j] = seq[j], seq[i]
            neighbors.add(tuple(seq))

        for _ in range(max_neighbors // 3 + 1):
            i, j = self.rng.sample(range(n), 2)
            seq = list(permutation)
            value = seq.pop(i)
            seq.insert(j, value)
            neighbors.add(tuple(seq))

        for _ in range(max_neighbors // 3 + 1):
            i, j = sorted(self.rng.sample(range(n), 2))
            seq = list(permutation)
            seq[i:j + 1] = reversed(seq[i:j + 1])
            neighbors.add(tuple(seq))

        return list(neighbors)[:max_neighbors]

    def solve(
        self,
        *,
        max_iterations: int = 100,
        max_neighbors: int = 80,
        time_limit: Optional[float] = None,
        out_of_sample_scenarios: int = 2000,
    ) -> SVRPResult:
        start = time.time()

        current_perm = self._nearest_neighbor_permutation()
        try:
            current_routes, current_cost = self._split_permutation(
                current_perm,
                self.training_scenarios,
            )
        except ValueError:
            current_routes = None
            for _ in range(500):
                seq = list(range(1, self.n + 1))
                self.rng.shuffle(seq)
                try:
                    current_routes, current_cost = self._split_permutation(
                        tuple(seq),
                        self.training_scenarios,
                    )
                    current_perm = tuple(seq)
                    break
                except ValueError:
                    continue
            if current_routes is None:
                raise ValueError("could not construct a mean-capacity-feasible plan")

        best_perm = current_perm
        best_routes = current_routes
        best_cost = current_cost

        completed = 0
        timed_out = False

        for iteration in range(max_iterations):
            if time_limit is not None and time.time() - start >= time_limit:
                timed_out = True
                break

            improved = False
            candidates = []

            for neighbor in self._neighbors(best_perm, max_neighbors):
                try:
                    routes, cost = self._split_permutation(
                        neighbor,
                        self.training_scenarios,
                    )
                    candidates.append((cost, neighbor, routes))
                except ValueError:
                    continue

            if candidates:
                cost, perm, routes = min(candidates, key=lambda x: x[0])
                if cost < best_cost - 1e-10:
                    best_cost = cost
                    best_perm = perm
                    best_routes = routes
                    improved = True

            completed = iteration + 1
            if not improved:
                break

        self.best_routes = best_routes

        validation = self._generate_scenarios(
            self.seed + 1_000_003,
            out_of_sample_scenarios,
        )
        oos_cost, expected_failures, failure_probability = self.evaluate_routes(
            best_routes,
            validation,
        )

        return SVRPResult(
            routes=best_routes,
            saa_cost=best_cost,
            out_of_sample_cost=oos_cost,
            failure_probability=failure_probability,
            expected_failures=expected_failures,
            elapsed_time=time.time() - start,
            iterations=completed,
            status="TIME_LIMIT" if timed_out else "LOCAL_OPTIMUM",
        )


def create_random_instance(
    num_customers: int = 15,
    num_vehicles: int = 4,
    vehicle_capacity: float = 60.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    customers = []
    for i in range(1, num_customers + 1):
        mean = float(rng.uniform(5, 15))
        customers.append(
            Customer(
                id=i,
                x=float(rng.uniform(0, 100)),
                y=float(rng.uniform(0, 100)),
                mean_demand=mean,
                std_demand=0.20 * mean,
            )
        )
    return StochasticCVRPSAA(
        customers,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        depot_location=(50.0, 50.0),
        num_scenarios=200,
        seed=seed,
        failure_penalty=100.0,
    )


if __name__ == "__main__":
    solver = create_random_instance()
    result = solver.solve(max_iterations=50, max_neighbors=60)
    print(result)
