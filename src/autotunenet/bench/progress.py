"""
Progress of benchmark runs: what each run reports, and how the terminal shows it.

`autotunenet run` prints one readable line when a run starts, finishes or fails.
With AUTOTUNENET_EVENTS=1 (set inside hosted notebooks) it also prints the same
event as one machine-readable line, `@@autotunenet {...}`, which the notebook
forwards to its live output. The local terminal follows that output and draws a
`Board`: per notebook and GPU, what is running, what finished and what failed.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MARKER = "@@autotunenet "
EVENTS_ENV = "AUTOTUNENET_EVENTS"


def describe(method: str, task: str, scenario: str, seed) -> str:
    return f"{method} · {task} · {scenario} · seed {seed}"


def duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def emit(event: str, **fields) -> None:
    """Print a machine-readable event if this process runs inside a hosted notebook."""
    if os.environ.get(EVENTS_ENV) == "1":
        print(MARKER + json.dumps({"event": event, "time": time.time(), **fields}), flush=True)


def parse(line: str) -> Optional[dict]:
    """The event in a log line, or None."""
    i = line.find(MARKER)
    if i < 0:
        return None
    try:
        return json.loads(line[i + len(MARKER):])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- local lines

def run_header(experiment: str, todo: int, total: int, device) -> None:
    done = total - todo
    tail = f" ({done} already done)" if done else ""
    print(f"{experiment}: {todo} of {total} cells to run on {device}{tail}", flush=True)


def run_start(i: int, n: int, cell) -> None:
    print(f"[{i}/{n}] ▶ {describe(cell.method, cell.task, cell.scenario, cell.seed)}", flush=True)


def run_done(i: int, n: int, cell, accuracy: float, epochs: int, seconds: float) -> None:
    print(f"[{i}/{n}] ✓ {describe(cell.method, cell.task, cell.scenario, cell.seed)}   "
          f"{accuracy * 100:.2f}%  {epochs} epochs  {duration(seconds)}", flush=True)


def run_failed(i: int, n: int, cell, error: str) -> None:
    print(f"[{i}/{n}] ✗ {describe(cell.method, cell.task, cell.scenario, cell.seed)}   failed: {error}",
          flush=True)


# ---------------------------------------------------------------- live board

@dataclass
class Slot:
    """One GPU process inside a notebook."""
    run: Optional[str] = None
    since: float = 0.0
    done: int = 0
    total: int = 0
    failed: List[str] = field(default_factory=list)


@dataclass
class NotebookProgress:
    title: str
    cells: int
    state: str = "prepared"
    slots: Dict[int, Slot] = field(default_factory=dict)
    last: Optional[str] = None
    last_time: float = 0.0
    applied: set = field(default_factory=set)

    def apply(self, event: dict) -> Optional[str]:
        """
        Update from one event; return a one-line description of it, or None if
        it was already applied (Kaggle replays the whole log on every reconnect).
        """
        key = (event.get("gpu", 0), event.get("i"), event.get("event"))
        if key in self.applied:
            return None
        self.applied.add(key)
        slot = self.slots.setdefault(int(event.get("gpu", 0)), Slot())
        slot.total = event.get("n", slot.total)
        name = describe(event.get("method"), event.get("task"), event.get("scenario"), event.get("seed"))
        kind = event.get("event")
        if kind == "start":
            slot.run, slot.since = name, event.get("time", time.time())
            return f"▶ {name}"
        if kind == "done":
            slot.done += 1
            slot.run = None
            self.last = (f"✓ {name}  {event.get('accuracy', 0) * 100:.2f}%  "
                         f"{duration(event.get('seconds', 0))}")
            self.last_time = event.get("time", time.time())
            return self.last
        if kind == "failed":
            slot.done += 1
            slot.run = None
            slot.failed.append(name)
            self.last = f"✗ {name}  failed: {event.get('error', '')}"
            self.last_time = event.get("time", time.time())
            return self.last
        return None

    @property
    def finished(self) -> int:
        return sum(s.done for s in self.slots.values())

    @property
    def failures(self) -> int:
        return sum(len(s.failed) for s in self.slots.values())


def _bar(done: int, total: int, width: int = 10) -> str:
    filled = round(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


class Board:
    """Draws the progress of a hosted run; redraws in place on a terminal."""

    def __init__(self, title: str, notebooks: List[NotebookProgress], log_dir, stream=None):
        self.title, self.notebooks, self.log_dir = title, notebooks, log_dir
        self.stream = stream or sys.stdout
        self.started = time.time()
        self.downloaded = 0
        self.total = sum(nb.cells for nb in notebooks)
        self._height = 0
        self.live = self.stream.isatty() and os.environ.get("TERM") != "dumb"

    def lines(self) -> List[str]:
        now = time.time()
        gpus = sum(max(len(nb.slots), 2) for nb in self.notebooks)
        out = [f"{self.title} · {self.total} runs on {len(self.notebooks)} notebook(s) ({gpus} GPUs)"
               f"   elapsed {duration(now - self.started)}", ""]
        for k, nb in enumerate(self.notebooks, 1):
            head = f"notebook {k}/{len(self.notebooks)}  {nb.state:<10} "
            if nb.state == "downloaded" and not nb.slots:
                out.append(head + f"results downloaded ({nb.cells} runs)")
                continue
            out.append(head + f"{_bar(nb.finished, nb.cells)}  {nb.finished}/{nb.cells} done"
                       + (f", {nb.failures} failed" if nb.failures else ""))
            for gpu in sorted(nb.slots):
                slot = nb.slots[gpu]
                if slot.run:
                    out.append(f"  GPU {gpu}  ▶ {slot.run:<48} running {duration(now - slot.since)}")
                elif slot.total and slot.done >= slot.total:
                    out.append(f"  GPU {gpu}  ■ finished its {slot.total} runs")
            if not nb.slots and nb.state == "running":
                out.append("  starting (installing, loading data)…")
        finished = sum(nb.finished for nb in self.notebooks)
        failed = sum(nb.failures for nb in self.notebooks)
        waiting = any(nb.state not in ("downloaded",) for nb in self.notebooks)
        out += ["", f"finished {finished}/{self.total}" + (f" · failed {failed}" if failed else "")
                + f" · downloaded {self.downloaded}/{self.total}"
                + (" (results arrive when each notebook ends)" if waiting else "")]
        last = max(self.notebooks, key=lambda nb: nb.last_time, default=None)
        if last and last.last:
            out.append(f"last  {last.last}")
        log_dir = os.path.abspath(self.log_dir)
        if log_dir.startswith(os.getcwd() + os.sep):
            log_dir = os.path.relpath(log_dir)
        out.append(f"logs  {log_dir}  (live.log per notebook; tail -f to follow)")
        return out

    def draw(self) -> None:
        """Redraw the board in place (terminal only)."""
        if not self.live:
            return
        lines = self.lines()
        if self._height:
            self.stream.write(f"\x1b[{self._height}F\x1b[J")
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self._height = len(lines)

    def note(self, text: str) -> None:
        """An event as a plain line (when output is not a terminal)."""
        if not self.live:
            print(f"[{duration(time.time() - self.started)}] {text}", file=self.stream, flush=True)
