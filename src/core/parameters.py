import random
from typing import Dict, Any, Tuple, List
import logging

class ParameterSpace:
    def __init__(self, space: Dict[str, Any]):
        self.space = space
        self._validate_space()
        
    def _validate_space(self) -> None:
        for name, values in self.space.items():
            if isinstance(values, tuple):
                if len(values) != 2:
                    logging.info(f"Tuple for {name} must be (min, max)")
                    raise ValueError(f"Tuple for {name} must be (min, max)")
            elif isinstance(values, list):
                if len(values) == 0:
                    logging.info(f"List for {name} cannot be empty")
                    raise ValueError(f"List for {name} cannot be empty")
            else:
                logging.info(f"Parameter {name} must be defineed by a tuple or list")
                raise TypeError(
                    f"Parameter {name} must be defined by a tuple or a list"
                )
                    
        
    def sample(self) -> Dict[str, Any]:
        params = {}
        for name, values in self.space.items():
            if isinstance(values, tuple):
                low, high = values
                params[name] = random.uniform(low, high)
            elif isinstance(values, list):
                params[name] = random.choice(values)
                
        return params
            