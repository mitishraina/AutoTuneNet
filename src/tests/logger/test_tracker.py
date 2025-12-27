from src.logging.tracker import Tracker

def test_tracker_does_not_crash_on_none_best():
    tracker = Tracker()
    
    tracker.log_step(
        step=0,
        params={"lr": 0.1},
        score=0.01,
        best_score=None
    )
    
def test_tracker_rollback_logs():
    tracker = Tracker()
    tracker.log_rollback_start()
    tracker.log_rollback_end()