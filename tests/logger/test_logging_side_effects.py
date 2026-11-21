import logging
import os
import subprocess
import sys

from autotunenet.logging.logger import get_logger
from autotunenet.safeguards.stability import StabilityMonitor


def _run_in(tmp_path, code, env=None):
    subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, check=True,
        env={**os.environ, **(env or {})}, capture_output=True,
    )
    return sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))


def test_importing_the_library_writes_nothing_to_disk(tmp_path):
    code = "import autotunenet.bayesian_optimizer, autotunenet.adapters.pytorch.adapter"
    assert _run_in(tmp_path, code) == []


def test_file_logging_is_opt_in_via_environment(tmp_path):
    code = "from autotunenet.logging.logger import get_logger; get_logger().info('hello')"
    files = _run_in(tmp_path, code, env={"AUTOTUNENET_LOG_FILE": "run/autotune.log"})
    assert "run/autotune.log" in files
    assert "hello" in (tmp_path / "run" / "autotune.log").read_text()


def test_messages_are_not_duplicated_through_the_root_logger():
    log = get_logger()
    assert log.propagate is False
    # pytest attaches its own capture handlers; count only the library's console handler
    consoles = [h for h in log.handlers if type(h) is logging.StreamHandler]
    assert len(consoles) == 1
    get_logger()
    assert len([h for h in log.handlers if type(h) is logging.StreamHandler]) == 1


def test_stability_monitor_does_not_print(capsys):
    guard = StabilityMonitor()
    guard.update(-0.1, -0.1)
    guard.update(-5.0, -0.1)
    assert capsys.readouterr().out == ""
