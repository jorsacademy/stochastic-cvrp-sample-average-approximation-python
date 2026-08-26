# Stochastic CVRP with Sample Average Approximation in Python

Educational stochastic-demand Capacitated Vehicle Routing Problem (CVRP) implementation using Sample Average Approximation (SAA), depot-restocking recourse, and local search.

## Model

The first-stage plan assigns every customer exactly once to at most `K` planned routes. Planned routes must satisfy capacity using **mean demand**.

Demand is stochastic. In each sampled scenario, if the next customer's realized demand would exceed the remaining vehicle capacity, the vehicle performs depot-restocking recourse:

```text
current location -> depot -> next customer
```

A configurable failure/replenishment penalty is added for each such recourse event.

## Objective

The SAA objective minimizes average scenario cost over a fixed training scenario set. Route evaluation includes:

- travel distance;
- depot-restocking detours;
- failure/replenishment penalties.

The same scenarios are used for every candidate during optimization, providing common random numbers for more stable comparisons.

## Optimization

The implementation uses:

- a customer-permutation representation;
- dynamic-programming split decoding into at most `K` mean-capacity-feasible routes;
- stochastic SAA segment costs in the decoder;
- swap, insertion, and reversal neighborhoods;
- local search over customer order and induced route assignments.

This is a **heuristic SAA implementation**. It does not provide a global optimality certificate for the stochastic program.

## Out-of-sample validation

After optimization, the selected plan is evaluated on an independent Monte Carlo sample. The result reports:

- in-sample `saa_cost`;
- independent `out_of_sample_cost`;
- probability of at least one replenishment/failure in a scenario;
- expected number of replenishments/failures per scenario.

Keeping optimization and validation samples separate avoids reporting only an in-sample SAA fit.

## Reference behavior

For the bundled 15-customer, 4-vehicle, capacity-60 example with seed 42, the validated implementation produced approximately:

- SAA cost: `560.29`
- out-of-sample cost: `561.12`
- failure probability: about `0.5%`
- expected failures: about `0.005` per scenario

Monte Carlo statistics depend on scenario count and the exact configured seed. They should not be interpreted as an exact stochastic optimum.

## Validation

The test suite checks:

- correct depot-restocking cost accounting;
- route coverage and mean-capacity feasibility;
- fleet-count enforcement;
- total mean-capacity infeasibility rejection;
- seeded reproducibility;
- separation of training and out-of-sample scenario sets.

Run:

```bash
python -m unittest discover -s tests -v
```

## Usage

```python
from stochastic_cvrp_saa import create_random_instance

solver = create_random_instance(
    num_customers=15,
    num_vehicles=4,
    vehicle_capacity=60.0,
    seed=42,
)

result = solver.solve(
    max_iterations=50,
    max_neighbors=60,
    out_of_sample_scenarios=2000,
)

print(result.status)
print(result.saa_cost)
print(result.out_of_sample_cost)
print(result.failure_probability)
print(result.routes)
```

## Scope and limitations

This repository demonstrates one specific stochastic-demand recourse policy. It does not model time windows, split deliveries, heterogeneous fleets, correlated demands, chance constraints, multi-stage route resequencing, or an exact stochastic-programming master/subproblem decomposition.

The planned capacity constraint uses mean demand; realized-demand violations are handled by depot-restocking recourse rather than forbidden outright.
