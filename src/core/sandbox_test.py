from src.core.parameters import ParameterSpace
# from src.core.optimizer_test import DummyOptimizer
from src.core.bayesian_optimizer import BayesianOptimizer
import logging

def test_dummy_optimizer(params):
    lr = params["lr"]
    return -(lr - 0.01) ** 2

def main():
    space = ParameterSpace({
        "lr": (0.001, 0.1)
    })
    
    optimizer = BayesianOptimizer(space, seed=42)
    
    for step in range(50):
        params = optimizer.suggest()
        score = test_dummy_optimizer(params)
        optimizer.observe(params, score)
        
        best_lr = optimizer.best_params()["lr"]
        best_score = optimizer.best_score()
        
        print(
            f"step: {step:02d} | "  
            f"lr: {params['lr']:.5f} | "  
            f"best_lr: {best_lr:.5f} | "  
            f"best_score: {best_score:.5f} | "  
        )
        
if __name__ == "__main__":
    main()
        