"""ClearML integration helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def init_task(project_name: str, task_name: str, params: dict[str, Any] | None = None):
    """Initialize a ClearML task. Returns Task or None if clearml is unavailable."""
    try:
        from clearml import Task
    except ImportError:
        logger.warning("clearml not installed, skipping task init")
        return None

    logger.info("initializing ClearML task %s/%s", project_name, task_name)
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        reuse_last_task_id=False,
    )
    if params is not None:
        task.connect(params)
    logger.info("ClearML task id: %s", task.id)
    return task


def set_description(task, text: str) -> None:
    """Attach a human-readable description to the task."""
    if task is None:
        return
    task.set_description(text)


def log_metrics(task, metrics: dict[str, float], title: str = "metrics") -> None:
    if task is None:
        return
    logger.debug("logging %d metrics to ClearML under '%s'", len(metrics), title)
    task_logger = task.get_logger()
    for name, value in metrics.items():
        task_logger.report_scalar(title, name, value=value, iteration=0)


def log_scalar(task, title: str, name: str, value: float, iteration: int = 0) -> None:
    if task is None:
        return
    task.get_logger().report_scalar(title, name, value=value, iteration=iteration)


def upload_artifact(task, name: str, path: Path) -> None:
    """Upload a single local file as a ClearML artifact."""
    if task is None or not path.exists():
        return
    logger.info("uploading artifact %s <- %s", name, path)
    task.upload_artifact(name=name, artifact_object=str(path))