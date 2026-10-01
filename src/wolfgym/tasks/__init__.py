from wolfgym.tasks.specs import (
    TASK_SPECS,
    TaskId,
    TaskModality,
    TaskSpec,
    eligible_tasks,
)
from wolfgym.tasks.adapters import TaskExample, adapt_record, adapt_trace

__all__ = [
    "TASK_SPECS", "TaskExample", "TaskId", "TaskModality", "TaskSpec",
    "adapt_record", "adapt_trace", "eligible_tasks",
]
