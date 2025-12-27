from src.core.parameters import ParameterSpace
from src.core.bayesian_optimizer import BayesianOptimizer

def test_suggest_then_observe_does_not_crash():
    space = ParameterSpace({"lr": (0.001, 0.1)})
    opt = BayesianOptimizer(space, seed=42)
    
    params = opt.suggest()
    score = -(params["lr"] - 0.01) ** 2  # Dummy objective: peak at lr=0.01
    
    # must not raise
    opt.observe(params, score)
    
def test_mutliple_test_steps_do_not_corrupt_state():
    space = ParameterSpace({"lr": (0.001, 0.1)})
    opt = BayesianOptimizer(space, seed=42)
    
    for _ in range(10):
        params = opt.suggest()
        score = -(params["lr"] - 0.01) ** 2
        opt.observe(params, score)
    
    assert opt.best_score() is not None
    