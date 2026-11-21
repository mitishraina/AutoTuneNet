# Benchmark extension points

The classes you use to add tasks, scenarios, methods and runtimes. See [Extending the benchmark](../benchmarks/extending.md) for worked examples.

## Tasks

::: autotunenet.bench.tasks.base.TaskSpec

## Scenarios

::: autotunenet.bench.scenarios.base.Scenario

## Methods

::: autotunenet.bench.core.method.Method

::: autotunenet.bench.core.method.SingleRunMethod

::: autotunenet.bench.core.method.Controller

::: autotunenet.bench.core.method.RunContext

::: autotunenet.bench.core.method.train_single_run

::: autotunenet.bench.core.method.train_epochs

::: autotunenet.bench.core.ledger.ComputeLedger

## Registries

`METHODS`, `TASKS` and `SCENARIOS` in `autotunenet.bench.core.registry` are instances of `Registry`.

::: autotunenet.bench.core.registry.Registry

## Results

::: autotunenet.bench.core.result.RunOutcome

::: autotunenet.bench.core.result.RunResult

## Runtimes

::: autotunenet.bench.runtimes.base.Runtime
