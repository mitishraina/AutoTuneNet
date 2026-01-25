import random
from typing import Dict, Any, Tuple, List

class ParameterSpace:
    def __init__(self, space: Dict[str, Any]):
        self.space = space
        self._validate_space()
    
    def _validate_space(self) -> None:
        for name, values in self.space.items():
            if isinstance(values, tuple):
                if len(values) != 2:
                    raise ValueError(f"{name} tuple must be (min, max)")
                self.space[name] = tuple(self._coerce_numeric(v, name) for v in values)

            elif isinstance(values, list):
                if len(values) == 0:
                    raise ValueError(f"{name} list cannot be empty")
                self.space[name] = [self._coerce_numeric(v, name) for v in values]

            else:
                raise TypeError(
                    f"Parameter {name} must be defined by a tuple or a list"
                )
                
    def _coerce_numeric(self, value, name):
        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(
                    f"Parameter '{name}' value '{value}' is not numeric"
                )

        raise TypeError(
            f"Parameter '{name}' has invalid type: {type(value)}"
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
            