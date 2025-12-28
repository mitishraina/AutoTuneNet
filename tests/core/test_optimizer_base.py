from autotunenet.optimizer import Optimizer
from autotunenet.parameters import ParameterSpace

class DummyOptimizer(Optimizer):
    def suggest(self):
        return self.param_space.sample()
    
    def test_observe_and_best_score():
        space = ParameterSpace({
            "x": (0,1)
        })
        opt = DummyOptimizer(space)
        
        opt.observe({"x": 0.1}, 1.0)
        opt.observe({"x": 0.2}, 2.0)
        
        assert opt.best_score() == 2.0
        assert opt.best_params() == {"x": 0.2}
        
    def test_best_score_without_history_raises():
        space = ParameterSpace({"x": (0,1)})
        opt = DummyOptimizer(space)
        
        try:
            opt.best_score()
            assert False, "Expected exception"
        except RuntimeError:
            assert True