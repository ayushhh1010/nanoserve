from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SamplingParams(_message.Message):
    __slots__ = ("max_tokens", "temperature", "top_k", "top_p", "ignore_eos")
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    IGNORE_EOS_FIELD_NUMBER: _ClassVar[int]
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    ignore_eos: bool
    def __init__(self, max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., top_k: _Optional[int] = ..., top_p: _Optional[float] = ..., ignore_eos: _Optional[bool] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("prompt", "params", "request_id", "deadline_unix", "resume_tokens")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_UNIX_FIELD_NUMBER: _ClassVar[int]
    RESUME_TOKENS_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    params: SamplingParams
    request_id: str
    deadline_unix: float
    resume_tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, prompt: _Optional[str] = ..., params: _Optional[_Union[SamplingParams, _Mapping]] = ..., request_id: _Optional[str] = ..., deadline_unix: _Optional[float] = ..., resume_tokens: _Optional[_Iterable[int]] = ...) -> None: ...

class GenerateChunk(_message.Message):
    __slots__ = ("token", "finish")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    FINISH_FIELD_NUMBER: _ClassVar[int]
    token: Token
    finish: Finish
    def __init__(self, token: _Optional[_Union[Token, _Mapping]] = ..., finish: _Optional[_Union[Finish, _Mapping]] = ...) -> None: ...

class Token(_message.Message):
    __slots__ = ("text", "token_id", "index")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    text: str
    token_id: int
    index: int
    def __init__(self, text: _Optional[str] = ..., token_id: _Optional[int] = ..., index: _Optional[int] = ...) -> None: ...

class Finish(_message.Message):
    __slots__ = ("reason", "output_tokens", "message")
    class Reason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        REASON_UNSPECIFIED: _ClassVar[Finish.Reason]
        EOS: _ClassVar[Finish.Reason]
        LENGTH: _ClassVar[Finish.Reason]
        CANCELLED: _ClassVar[Finish.Reason]
        REJECTED: _ClassVar[Finish.Reason]
        PREEMPTED: _ClassVar[Finish.Reason]
        ERROR: _ClassVar[Finish.Reason]
    REASON_UNSPECIFIED: Finish.Reason
    EOS: Finish.Reason
    LENGTH: Finish.Reason
    CANCELLED: Finish.Reason
    REJECTED: Finish.Reason
    PREEMPTED: Finish.Reason
    ERROR: Finish.Reason
    REASON_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    reason: Finish.Reason
    output_tokens: int
    message: str
    def __init__(self, reason: _Optional[_Union[Finish.Reason, str]] = ..., output_tokens: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("ready", "draining", "num_running", "num_waiting", "kv_utilization", "prefix_hit_rate", "replica_id", "uptime_seconds")
    READY_FIELD_NUMBER: _ClassVar[int]
    DRAINING_FIELD_NUMBER: _ClassVar[int]
    NUM_RUNNING_FIELD_NUMBER: _ClassVar[int]
    NUM_WAITING_FIELD_NUMBER: _ClassVar[int]
    KV_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    PREFIX_HIT_RATE_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    draining: bool
    num_running: int
    num_waiting: int
    kv_utilization: float
    prefix_hit_rate: float
    replica_id: str
    uptime_seconds: int
    def __init__(self, ready: _Optional[bool] = ..., draining: _Optional[bool] = ..., num_running: _Optional[int] = ..., num_waiting: _Optional[int] = ..., kv_utilization: _Optional[float] = ..., prefix_hit_rate: _Optional[float] = ..., replica_id: _Optional[str] = ..., uptime_seconds: _Optional[int] = ...) -> None: ...

class DrainRequest(_message.Message):
    __slots__ = ("graceful",)
    GRACEFUL_FIELD_NUMBER: _ClassVar[int]
    graceful: bool
    def __init__(self, graceful: _Optional[bool] = ...) -> None: ...

class DrainResponse(_message.Message):
    __slots__ = ("in_flight",)
    IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    in_flight: int
    def __init__(self, in_flight: _Optional[int] = ...) -> None: ...
