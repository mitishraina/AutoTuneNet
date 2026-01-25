from .base import BaseController

class SchedulerController(BaseController):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        
    def on_epoch_end(self, metric: float):
        self.scheduler.step()
     
    def apply(self, optimizer):
        pass #scheduler already updated optimizer
    
    def name(self) -> str:
        return "Scheduler"