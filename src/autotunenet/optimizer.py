from autotunenet.logging.logger import logging
from autotunenet.exceptions.exception import CustomException
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class Optimizer(ABC):
    '''
    Abstract base class for all optimizers.
    '''
    def __init__(self, param_space):
        '''
        Args:
            param_space: Object responsible for sampling and vaidating hyperparameters config
        '''
        self.param_space = param_space
        self.history: List[Tuple[Dict[str, Any], float]] = []
        self._step = 0

    @abstractmethod
    def suggest(self) -> Dict[str, Any]:
        '''
        Suggest a new set of hyperparameters.
        Returns:
            A dictionary of hyperparameters.
        '''
        pass

    def observe(self, params: Dict[str, Any], score: float) -> None:
        '''
        Observe the result of a hyperparameter configuration.
        
        Args:
            params: A dictionary of hyperparameters.
            score: The score (e.g., validation loss) obtained with these hyperparameters. Overall Performance metrics
        '''
        if not isinstance(score, (int, float)):
            logging.info("Score must be a numeric value")
            raise ValueError("Score must be a numeric value")
        
        self._step += 1
        self.history.append((params, score))
        
    def best_result(self) -> Tuple[Dict[str, Any], float]:
        if not self.history:
            logging.info("No observations available yet")
            raise RuntimeError("No observations available yet")
        return max(self.history, key=lambda item: item[1])

    def best_params(self) -> Dict[str, Any]:
        return self.best_result()[0]
    
    def best_score(self) -> float:
        """
        Convenience method to return only the best score.
        """
        return self.best_result()[1]

    def num_observations(self) -> int:
        """
        Number of observations collected so far.
        """
        return len(self.history)