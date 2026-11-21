"""Stores the last known-good hyperparameters so they can be restored."""

from typing import Dict, Any, Optional

class Rollback:
    """
    Holds a copy of the last configuration accepted as good.

    Only hyperparameters are stored: model weights and optimizer state are
    not, so a rollback cannot undo divergence that has already happened.
    """

    def __init__(self):
        self._last_good_params: Optional[Dict[str, Any]] = None
        
    def update(self, params: Dict[str, Any]) -> None:
        """Remember `params` (a copy) as the last good configuration."""
        self._last_good_params = params.copy()
        
    def rollback(self) -> Dict[str, Any]:
        """
        Return a copy of the last good configuration.

        Raises:
            RuntimeError: No configuration has been stored yet.
        """
        if self._last_good_params is None:
            raise RuntimeError("No safe parameters available for rollback")
        
        return self._last_good_params.copy()