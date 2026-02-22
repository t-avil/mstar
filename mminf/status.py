from enum import Enum


class Status(Enum):
    WAITING = "waiting"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"