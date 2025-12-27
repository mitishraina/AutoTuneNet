from src.core.parameters import ParameterSpace

def test_sample_returns_all_params():
    space = ParameterSpace({
        "lr": (0.001, 0.1),
        "batch_size": [32, 64, 128]
    })
    
    params = space.sample()
    
    assert "lr" in params
    assert "batch_size" in params
    
def test_continous_param_in_range():
    space = ParameterSpace({
        "lr": (0.001, 0.1)
    })
    
    for _ in range(100):
        lr = space.sample()["lr"]
        assert 0.001 <= lr <= 0.1
        
def test_discrete_param_valid():
    choices = [32, 64, 128]
    space = ParameterSpace({
        "batch_size": choices
    })
    
    for _ in range(50):
        assert space.sample()["batch_size"] in choices