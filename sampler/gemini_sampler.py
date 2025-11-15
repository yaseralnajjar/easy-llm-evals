import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

from ..types_eval import MessageList, SamplerBase, SamplerResponse


class GeminiSampler(SamplerBase):
    """
    Sample from Google's Gemini API using the google-genai library
    Reference: https://ai.google.dev/gemini-api/docs/quickstart
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 8192,
    ):
        load_dotenv()  # Load .env file
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content: Any) -> dict:
        """Convert message to Gemini format"""
        if isinstance(content, str):
            return {"role": role, "parts": [{"text": content}]}
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append({"text": item["text"]})
                    elif item.get("type") == "image_url":
                        # Gemini handles images differently
                        parts.append({"text": "[Image content]"})
                else:
                    parts.append({"text": str(item)})
            return {"role": role, "parts": parts}
        return {"role": role, "parts": [{"text": str(content)}]}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        max_trials = 10
        
        # Convert message list to Gemini format
        contents = []
        for msg in message_list:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map OpenAI roles to Gemini roles
            if role == "system":
                # Gemini doesn't have a system role, prepend to first user message
                continue
            elif role == "assistant":
                gemini_role = "model"
            else:
                gemini_role = "user"
            
            contents.append(self._pack_message(gemini_role, content))
        
        # Prepend system message if provided
        if self.system_message and contents:
            if contents[0]["role"] == "user":
                # Add system message to first user message
                system_part = {"text": f"System: {self.system_message}\n\n"}
                contents[0]["parts"].insert(0, system_part)
        
        while trial < max_trials:
            try:
                config = GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                
                # Extract text from response
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text = candidate.content.parts[0].text
                        
                        # Ensure message_list has proper format for reporting
                        formatted_message_list = []
                        for msg in message_list:
                            if "content" not in msg and "parts" in msg:
                                # Convert Gemini format back to OpenAI format
                                formatted_message_list.append({
                                    "role": msg.get("role", "user"),
                                    "content": " ".join([p.get("text", "") for p in msg["parts"]])
                                })
                            else:
                                formatted_message_list.append(msg)
                        
                        return SamplerResponse(
                            response_text=text,
                            response_metadata={"usage": getattr(response, "usage_metadata", None)},
                            actual_queried_message_list=formatted_message_list,
                        )
                
                raise ValueError("Gemini API returned empty response")
                
            except Exception as e:
                exception_backoff = 2**trial
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
        
        # If all trials fail, return empty response
        return SamplerResponse(
            response_text="",
            response_metadata={"usage": None},
            actual_queried_message_list=message_list,
        )

