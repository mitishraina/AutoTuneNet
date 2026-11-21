"""The `autotunenet` command: help text, plugins, resumable runs."""
import pytest

from autotunenet import cli
from autotunenet.bench.core.result import RunResult

PLUGIN = '''
from dataclasses import replace
from autotunenet.bench.core.registry import TASKS
TASKS.add("my_blobs", replace(TASKS.get("toy"), name="my_blobs"))
'''

EXPERIMENT = '''
name: mine
epochs: 4
seeds: [1]
tasks: [my_blobs]
scenarios:
  stationary: {}
methods:
  fixed_lr: {}
  pbt: {population: 2, perturbation_interval: 2}
'''


def test_top_level_help_lists_every_command_on_one_line(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for name, (summary, _, _) in cli.COMMANDS.items():
        assert name in out
    assert "autotunenet <command> --help" in out


@pytest.mark.parametrize("command", list(cli.COMMANDS))
def test_each_command_explains_itself_with_examples(command, capsys):
    with pytest.raises(SystemExit):
        cli.main([command, "--help"])
    out = capsys.readouterr().out
    assert "examples:" in out and f"autotunenet {command}" in out


def test_unknown_experiment_names_the_built_in_ones(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["status", "--experiment", "no_such_experiment"])
    assert "comparison" in str(e.value) + capsys.readouterr().out


def test_a_plugin_task_runs_everywhere_including_ray_workers(tmp_path, capsys):
    plugin, exp = tmp_path / "my_tasks.py", tmp_path / "mine.yaml"
    plugin.write_text(PLUGIN)
    exp.write_text(EXPERIMENT)
    args = ["--experiment", str(exp), "--plugins", str(plugin), "--results-dir", str(tmp_path / "out")]
    assert cli.main(["run", *args, "--device", "cpu", "--no-report"]) == 0
    r = RunResult.load(tmp_path / "out" / "mine" / "pbt" / "my_blobs" / "stationary" / "seed_1.json")
    assert r.task == "my_blobs" and len(r.history) == 4

    capsys.readouterr()
    assert cli.main(["status", *args]) == 0
    assert "mine: 2/2 cells done" in capsys.readouterr().out
