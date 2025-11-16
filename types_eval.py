from dataclasses import dataclass, field
from typing import Any, Literal, overload
import threading
from decimal import Decimal

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


@dataclass
class SamplerSpeedStats:
    """
    Aggregated speed statistics for a sampler instance.
    """
    total_chars: int = 0
    total_duration_ms: Decimal = Decimal('0')
    num_requests: int = 0
    sum_tps: Decimal = Decimal('0')  # sum of per-request tokens/sec
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    
    def record_request(self, duration_ms: float, output_char_count: int) -> None:
        """
        Record timing and output metrics for a single request.
        
        Args:
            duration_ms: Wall-clock duration in milliseconds
            output_char_count: Number of characters in the response
        """
        # Calculate tokens and TPS for this request using Decimal for precision
        # 1 token ≈ 4 characters
        duration_decimal = Decimal(str(duration_ms))
        tokens = Decimal(str(output_char_count)) / Decimal('4')
        tps = (tokens / duration_decimal) * Decimal('1000') if duration_ms > 0 else Decimal('0')
        
        with self._lock:
            self.total_chars += output_char_count
            self.total_duration_ms += duration_decimal
            self.num_requests += 1
            self.sum_tps += tps
    
    def get_average_tps(self) -> float:
        """Calculate average tokens per second across all requests."""
        with self._lock:
            if self.num_requests == 0:
                return 0.0
            return float(self.sum_tps / Decimal(str(self.num_requests)))



@dataclass
class SamplerResponse:
    """
    Response from a sampler.
    """
    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]

class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """
    
    def __init__(self):
        self.speed_stats = SamplerSpeedStats()

    def __call__(
        self, 
        message_list: MessageList,
    ) -> SamplerResponse:
        raise NotImplementedError


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    htmls: list[str]  # strings of valid HTML
    convos: list[MessageList]  # sampled conversations
    metadata: dict[str, Any] | None  # Extra data such as rubric scores or sollen


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None  # sampled conversation
    example_level_metadata: dict[str, Any] | None = (
        None  # Extra data such as rubric scores or sollen
    )


class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError

