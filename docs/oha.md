# This refers to AutoTuneNet_bench

## Formal Definition of Online Hyperparameter Adaptation (OHA)

### 1. Problem Setting

Consider a supervised learning problem where a model with parameters

$$
\theta_t \in \Theta
$$

is trained over discrete time steps

$$
t = 1, 2, \ldots, T.
$$

At each time step, training is governed by a set of hyperparameters

$$
\lambda_t \in \Lambda
$$

(e.g., learning rate, momentum, weight decay), which influence the optimization
dynamics but are not directly optimized by gradient descent.

Let:

- $D_t$ denote the data distribution observed at time $t$,
- $\ell(\theta, x)$ denote the training loss,
- $f_t(\theta, \lambda)$ denote an evaluation metric (e.g., validation loss or accuracy).

The training dynamics are defined by the update operator:

$$
\theta_{t+1} = U(\theta_t, \lambda_t, D_t).
$$


---

### 2. Non-Stationarity

Unlike classical hyperparameter optimization, we do **not assume stationarity** of the evaluation objective. Instead, the objective may evolve over time:

$$
f_t \neq f_{t+1}
$$

This non-stationarity can arise from several sources:

- **Changes in the data distribution**  
  $$
  \mathcal{D}_t
  $$
- **Changes in optimization dynamics**, such as gradient noise or numerical instability
- **Changes in training phase**, e.g., early vs. late training behavior

As a result, hyperparameters that are optimal at time $t$ may become suboptimal at time $t + k$.


---

### 3. Online Hyperparameter Adaptation (OHA)

**Online Hyperparameter Adaptation (OHA)** is the problem of selecting hyperparameters sequentially during a single training run, without restarting training.

At each time step $t$, a controller selects hyperparameters:

$$
\lambda_t = \pi\left(H_{1:t-1}\right)
$$

where:

- $\pi$ is an adaptation policy (controller)
- $H_{1:t-1}$ is the history of observed metrics, hyperparameters, and optionally internal signals up to time $t-1$

After applying $\lambda_t$, the system observes feedback:

$$
y_t = f_t\left(\theta_t, \lambda_t\right)
$$

This feedback is used to update the controller’s internal state before the next decision.


---

### 4. Constraints and Practical Considerations

An OHA system operates under the following constraints:

1. **Single-Run Constraint**  
   Training proceeds continuously; restarting from scratch is not allowed.

2. **Limited Feedback**  
   The controller observes only scalar metrics (e.g., validation loss), not gradients of  
   $$
   f_t \quad \text{with respect to} \quad \lambda_t
   $$

3. **Safety Constraints**  
   Hyperparameter updates must avoid destabilizing training (e.g., divergence or exploding loss).

4. **Compute Efficiency**  
   The controller must introduce minimal overhead relative to standard training.


---

### 5. Objectives

The goal of OHA is **not** to find a globally optimal hyperparameter configuration, but to optimize training dynamically under non-stationarity.

Typical objectives include:

- **Minimizing cumulative validation loss**:
  $$
  \sum_{t=1}^{T} f_t\left(\theta_t, \lambda_t\right)
  $$

- **Minimizing instability under distribution or optimization shifts**
- **Reducing sensitivity to manual hyperparameter tuning**
- **Achieving robust performance across training phases**


---

### 6. Relation to Classical Hyperparameter Optimization

| Aspect          | Classical HPO      | OHA                           |
|-----------------|--------------------|-------------------------------|
| Stationarity    | Assumed            | Not assumed                   |
| Training runs   | Multiple restarts  | Single run                    |
| Decision timing | Offline / batch    | Online / sequential           |
| Objective       | Final performance  | Trajectory-level performance  |
| Safety          | Implicit           | Explicitly required           |


---

### 7. Scope of AutoTuneNet-Bench

AutoTuneNet-Bench evaluates OHA methods by:

- Injecting controlled non-stationarity during training
- Measuring stability, recovery, and degradation
- Comparing online controllers against static and scheduled baselines

The benchmark is controller-agnostic and does not assume any specific adaptation algorithm.
