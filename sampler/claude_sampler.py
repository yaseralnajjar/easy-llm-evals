import time
import os

import anthropic
from dotenv import load_dotenv

from ..types_eval import MessageList, SamplerBase, SamplerResponse
from .. import common

CLAUDE_SYSTEM_MESSAGE_LMSYS = (
    "The assistant is Claude, created by Anthropic. The current date is "
    "{currentDateTime}. Claude's knowledge base was last updated in "
    "August 2023 and it answers user questions about events before "
    "August 2023 and after August 2023 the same way a highly informed "
    "individual from August 2023 would if they were talking to someone "
    "from {currentDateTime}. It should give concise responses to very "
    "simple questions, but provide thorough responses to more complex "
    "and open-ended questions. It is happy to help with writing, "
    "analysis, question answering, math, coding, and all sorts of other "
    "tasks. It uses markdown for coding. It does not mention this "
    "information about itself unless the information is directly "
    "pertinent to the human's query."
).format(currentDateTime="2024-04-01")
# reference: https://github.com/lm-sys/FastChat/blob/7899355ebe32117fdae83985cf8ee476d2f4243f/fastchat/conversation.py#L894


class ClaudeCompletionSampler(SamplerBase):

    def __init__(
        self,
        model: str,
        system_message: str | None = None,
        temperature: float = 0.0,  # default in Anthropic example
        max_tokens: int = 4096,
        thinking_budget: int | None = None,
    ):
        super().__init__()
        load_dotenv()  # Load .env file
        self.client = anthropic.Anthropic()
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.image_format = "base64"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image",
            "source": {
                "type": encoding,
                "media_type": f"image/{format}",
                "data": image,
            },
        }
        return new_image

    def _handle_text(self, text):
        return {"type": "text", "text": text}

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        while True:
            try:
                if not common.has_only_user_assistant_messages(message_list):
                    raise ValueError(f"Claude sampler only supports user and assistant messages, got {message_list}")
                
                start_time = time.perf_counter()
                
                # Calculate max_tokens: must be greater than thinking budget
                max_tokens = self.max_tokens
                if self.thinking_budget is not None and max_tokens <= self.thinking_budget:
                    # Ensure max_tokens is at least thinking_budget + 1024 for response
                    max_tokens = self.thinking_budget + 1024
                
                # Build API call parameters
                api_params = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": message_list,
                }
                
                # Add thinking configuration if budget is specified
                if self.thinking_budget is not None:
                    api_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget,
                    }
                    # When thinking is enabled, temperature must be 1
                    api_params["temperature"] = 1.0
                else:
                    # Use configured temperature when thinking is disabled
                    api_params["temperature"] = self.temperature
                
                # Add system message if provided
                if self.system_message:
                    api_params["system"] = self.system_message
                    claude_input_messages: MessageList = [{"role": "system", "content": self.system_message}] + message_list
                else:
                    claude_input_messages = message_list
                
                response_message = self.client.messages.create(**api_params)
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                
                # Extract text from response content blocks
                # Claude returns array of content blocks: [{"type": "thinking", ...}, {"type": "text", ...}]
                response_text = ""
                for content_block in response_message.content:
                    if hasattr(content_block, 'type') and content_block.type == "text":
                        response_text = content_block.text
                        break
                
                # Fallback to old behavior if no text block found
                if not response_text and len(response_message.content) > 0:
                    response_text = response_message.content[0].text
                
                # Record speed stats
                self.speed_stats.record_request(duration_ms, len(response_text))
                
                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={"content_blocks": response_message.content},
                    actual_queried_message_list=claude_input_messages,
                )
            except anthropic.RateLimitError as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
