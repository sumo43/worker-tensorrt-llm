import time
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams

import torch

DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_CONCURRENCY = 300
DEFAULT_BATCH_SIZE_GROWTH_FACTOR = 3
DEFAULT_MIN_BATCH_SIZE = 1

SAMPLING_PARAM_TYPES = {
    "n": int,
    "best_of": int,
    "presence_penalty": float,
    "frequency_penalty": float,
    "repetition_penalty": float,
    "temperature": Union[float, int],
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "use_beam_search": bool,
    "length_penalty": float,
    "early_stopping": Union[bool, str],
    "stop": Union[str, list],
    "stop_token_ids": list,
    "ignore_eos": bool,
    "max_tokens": int,
    "logprobs": int,
    "prompt_logprobs": int,
    "skip_special_tokens": bool,
    "spaces_between_special_tokens": bool,
    "include_stop_str_in_output": bool
}





class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
