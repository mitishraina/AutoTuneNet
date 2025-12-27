# For restoring last known good params/config

from typing import Dict, Any, Optional

class Rollback:
    def __init__(self):
        self._last_good_params: Optional[Dict[str, Any]] = None
        
    def update(self, params: Dict[str, Any]) -> None:
        self._last_good_params = params.copy()
        
    def rollback(self) -> Dict[str, Any]:
        if self._last_good_params is None:
            raise RuntimeError("No safe parameters available for rollback")
        
        return self._last_good_params.copy()