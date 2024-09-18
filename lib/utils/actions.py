from enum import Enum


class Actions(Enum):
    init = -1
    in_queue = 1
    start = 5
    done = 10
    failed = 20
    duplicated = 25
