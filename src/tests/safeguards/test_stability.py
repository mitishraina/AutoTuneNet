from src.safeguards.stability import StabilityMonitor

def test_single_patience_does_not_rollback():
    guard = StabilityMonitor(patience=2)
    
    best = -0.01
    score = -0.02
    
    assert guard.update(score, best) is True
    
def test_consecutive_bad_steps_trigger_rollback():
    guard = StabilityMonitor(patience=2)
    
    best = -0.01
    
    assert guard.update(-0.02, best) is True
    assert guard.update(-0.03, best) is False # triggers rollback
    
def test_cooldown_allows_steps():
    guard = StabilityMonitor(patience=1, cooldown=2)
    
    best = -0.01
    
    assert guard.update(-0.02, best) is False # triggers rollback
    
    # During cooldown, all steps are accepted
    assert guard.update(-0.5, best) is True
    assert guard.update(-0.6, best) is True
    
def test_reset_clears_state():
    guard = StabilityMonitor(patience=1)
    
    guard.update(-0.02, -0.01)
    guard.reset()
    
    assert guard.in_rollback is False