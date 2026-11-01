"""On-disk format constants for sequence trace datasets and releases."""

SEQUENCE_FORMAT_VERSION = 5
TRACE_ARRAYS = ("board_tokens", "trace_tokens", "trace_len", "trace_roles")
STATE_ARRAYS = (
    "state_no_tool",
    "state_tool",
    "state_best",
    "state_basis",
    "state_value",
)
SEQUENCE_ARRAYS = (*TRACE_ARRAYS, *STATE_ARRAYS, "state_count")
