import time
from typing import Any
import re
import ollama
from dotenv import load_dotenv
from ..types_eval import MessageList, SamplerBase, SamplerResponse

OLLAMA_SYSTEM_MESSAGE_DEFAULT = "You are a helpful assistant."


class OllamaSampler(SamplerBase):
    """
    Sample from Ollama's chat completion API
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        super().__init__()
        load_dotenv()
        self.model = model
        self.system_message = system_message or OLLAMA_SYSTEM_MESSAGE_DEFAULT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ollama.Client()

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def _clean_think_tags(self, text: str) -> str:
        # Remove everything between <think> and </think>, including the tags themselves
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        MAX_RETRIES = 10
        while trial < MAX_RETRIES:
            try:
                start_time = time.perf_counter()
                response = self.client.chat(
                    model=self.model,
                    messages=message_list,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                )
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                content = response["message"]["content"]
                content = self._clean_think_tags(content)

                if not content:
                    raise ValueError("Ollama API returned empty response; retrying")
                
                # Record speed stats
                self.speed_stats.record_request(duration_ms, len(content))
                
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                print(
                    f"Exception, retrying",
                    e,
                )
                trial += 1
            # unknown error shall throw exception
        raise RuntimeError(f"Ollama failed after {MAX_RETRIES} retries")
