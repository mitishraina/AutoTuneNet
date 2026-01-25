from abc import ABC, abstractmethod

class NonStationarityRegime(ABC):
    """
    Base class for non stationarity regime
    """
    
    def __init__(self, start_epoch: int):
        self.start_epoch = start_epoch
        self._active = False
        
    def maybe_apply(self, epoch, **context):
        """
        Called every epoch by the runner
        """
        if epoch >= self.start_epoch and not self._active:
            self.apply(**context)
            self._active = True
            
    @abstractmethod
    def apply(self, **context):
        """
        Apply non-stationary change.
        """
        pass