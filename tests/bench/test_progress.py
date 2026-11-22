import io

from autotunenet.bench import progress
from autotunenet.bench.progress import Board, NotebookProgress


def _event(kind, gpu, i, n, **kw):
    return {"event": kind, "gpu": gpu, "i": i, "n": n, "method": "pbt", "task": "mnist",
            "scenario": "stationary", "seed": 42 + i, "time": 0.0, **kw}


def test_events_round_trip_through_a_log_line(monkeypatch, capsys):
    monkeypatch.setenv(progress.EVENTS_ENV, "1")
    progress.emit("done", i=1, n=2, accuracy=0.99)
    line = capsys.readouterr().out
    event = progress.parse("GPU 0 noise " + line)
    assert event["event"] == "done" and event["accuracy"] == 0.99
    assert progress.parse("an ordinary log line") is None


def test_events_are_silent_outside_notebooks(monkeypatch, capsys):
    monkeypatch.delenv(progress.EVENTS_ENV, raising=False)
    progress.emit("start", i=1, n=1)
    assert capsys.readouterr().out == ""


def test_replayed_events_are_counted_once():
    nb = NotebookProgress("nb", cells=2)
    events = [_event("start", 0, 1, 2), _event("done", 0, 1, 2, accuracy=0.98, seconds=60),
              _event("start", 0, 2, 2), _event("failed", 0, 2, 2, error="RuntimeError: boom")]
    for e in events + events:                      # Kaggle replays the log on reconnect
        nb.apply(e)
    assert nb.finished == 2 and nb.failures == 1
    assert nb.last.startswith("✗") and "boom" in nb.last


def test_board_shows_running_finished_and_failed_runs():
    nb = NotebookProgress("nb", cells=3, state="running")
    nb.apply(_event("start", 0, 1, 2))
    nb.apply(_event("done", 1, 1, 1, accuracy=0.9, seconds=5))
    board = Board("exp · Kaggle", [nb], "logs", stream=io.StringIO())
    text = "\n".join(board.lines())
    assert "1/3 done" in text and "▶ pbt · mnist · stationary · seed 43" in text
    assert "GPU 1  ■ finished its 1 runs" in text
    assert "✓ pbt · mnist · stationary · seed 43  90.00%" in text


def test_board_prints_plain_lines_when_not_a_terminal():
    out = io.StringIO()
    board = Board("exp", [NotebookProgress("nb", cells=1)], "logs", stream=out)
    assert not board.live
    board.note("✓ something")
    board.draw()                                   # no redraw codes in a log file
    assert "✓ something" in out.getvalue() and "\x1b[" not in out.getvalue()
