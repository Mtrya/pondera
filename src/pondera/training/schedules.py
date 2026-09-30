"""Map supervised core exits to teacher depths for recurrent-depth training.

An exit is a core repetition boundary. The schedule fixes which teacher depth
supervises each exit; None leaves an exit unsupervised. Progressive spreads the
teacher depths linearly over the exits (first exit gets depth 1, last gets the
full teacher depth) with round-half-up quantization.
"""

import math

SCHEDULES = ("final_only", "uniform", "progressive")


def exit_teacher_depths(num_exits, schedule, teacher_depth=16):
    if num_exits < 1 or teacher_depth < 1:
        raise ValueError("Exit count and teacher depth must be positive")
    if schedule == "final_only":
        return [None] * (num_exits - 1) + [teacher_depth]
    if schedule == "uniform":
        return [teacher_depth] * num_exits
    if schedule == "progressive":
        if num_exits == 1:
            return [teacher_depth]
        return [
            int(math.floor(1 + (teacher_depth - 1) * k / (num_exits - 1) + 0.5))
            for k in range(num_exits)
        ]
    raise ValueError(f"Unknown supervision schedule: {schedule}")
