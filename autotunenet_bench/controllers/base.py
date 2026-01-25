from abc import ABC, abstractmethod
from typing import Dict

class BaseController(ABC):
    """
    Abstract interface for Online Hyperparameter Controllers
    """
    
    def __init__(self):
        self.last_instability = False
        self.last_rollback = False
    
    @abstractmethod
    def on_epoch_end(self, metric: float):
        pass # for observing training feedback at the end of epoch
    
    @abstractmethod
    def apply(self, optimizer) -> None:
        pass # apply controller decisions to the optimizer
    
    # @abstractmethod
    # def on_regime_start(self, epoch: int):
    #     pass
    
    @abstractmethod
    def name(self) -> str:
        pass # human-readable controller name