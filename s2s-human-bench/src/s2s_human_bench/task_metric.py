"""The Cekura custom-code metric that scores the main task on the platform, built from ``tasks.py``.

Runs scored on the platform get Task Outcome from this metric instead of the offline task file:
it reads the scenario's main task from the test profile field ``eval_target_task`` and decides
from the call's tool calls with the same checker the scorer uses for ``--tasks``.
``metric_source`` copies that checker into one self-contained file for the metric's code field,
dropping the type annotations, the task-file loader and the imports the metric does not need.
Never edit the copy: change ``tasks.py``, rebuild with ``s2s-human-bench task-metric``, and update
the platform metric.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROFILE_FIELD = "eval_target_task"

HEADER = """\
# Human Bench: main task completion (Cekura custom-code metric, binary, no LLM).
#
# Built from s2s-human-bench/src/s2s_human_bench/tasks.py by `s2s-human-bench task-metric`; change
# tasks.py and rebuild, never this copy. Reads the scenario's main task from the test profile field
# eval_target_task (JSON: requirements, must_not_run, then_undo, undo_tools) and decides from the
# call's tool calls whether the agent completed it.
# TRUE: done. FALSE: not done, with the reason. None: no task in the profile, or no verdict.
"""

WRAPPER = """

def _task_spec(data):
    profile = data.get("test_profile") or {}
    nested = (profile.get("information") or {}).get("main_agent_variables") or profile.get("main_agent_variables") or {}
    raw = profile.get("eval_target_task") or nested.get("eval_target_task")
    if not raw:
        return None
    return json.loads(raw, strict=False) if isinstance(raw, str) else raw


def _transcript(data):
    raw = data.get("cekura_transcript_json")
    if raw is None:
        raw = data.get("transcript_json")
    if isinstance(raw, str):
        raw = json.loads(raw, strict=False)
    if not isinstance(raw, list):
        raise ValueError("transcript unavailable")
    return raw


try:
    _spec = _task_spec(data)
    if _spec is None:
        _result, _explanation = None, "No main task in this test profile (not reached)."
    else:
        _done, _reason = task_verdict(_spec, _transcript(data))
        _result = _done
        _explanation = "Main task done: every required tool call was made with the caller's values." if _done else f"Main task not done: {_reason}."
except Exception as error:
    _result = None
    _explanation = "Evaluation error (no verdict): " + type(error).__name__ + ": " + str(error)
"""

_DROP_FUNCTIONS = {"load_tasks"}
_DROP_CLASSES = {"TaskFileError"}
_DROP_IMPORTS = {"__future__", "pathlib", "typing"}


class _StripAnnotations(ast.NodeTransformer):
    """The platform's Python version is not pinned, so the copy carries no annotations."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        a = node.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, *(x for x in (a.vararg, a.kwarg) if x)]:
            arg.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign | None:
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def _kept(index: int, node: ast.stmt) -> bool:
    if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return False  # the module docstring describes the scorer flag, not the metric
    if isinstance(node, ast.ImportFrom) and node.module in _DROP_IMPORTS:
        return False
    if isinstance(node, ast.FunctionDef) and node.name in _DROP_FUNCTIONS:
        return False
    return not (isinstance(node, ast.ClassDef) and node.name in _DROP_CLASSES)


def metric_source(tasks_path: Path | None = None) -> str:
    """The metric's code: the checker from ``tasks.py`` plus the platform wrapper."""
    tree = ast.parse((tasks_path or Path(__file__).with_name("tasks.py")).read_text(encoding="utf-8"))
    tree.body = [node for i, node in enumerate(tree.body) if _kept(i, node)]
    tree = ast.fix_missing_locations(_StripAnnotations().visit(tree))
    return HEADER + "\n" + ast.unparse(tree) + "\n" + WRAPPER
