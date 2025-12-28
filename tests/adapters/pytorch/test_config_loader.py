from src.config.loader import load_config_from_dict

def test_valid_config_loads():
    config = load_config_from_dict({
        "parameter_space": {
            "lr": [0.001, 0.1]
        }
    })
    
    assert "lr" in config.parameter_space
    
def test_invalid_parameter_space_raises():
    try:
        load_config_from_dict({
            "parameter_space": "invalid"
        })
        
        assert False
    except ValueError:
        assert True