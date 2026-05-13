from __future__ import annotations
import base64
import io
import os
from typing import List, Union

from matplotlib.figure import Figure
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from .base_session import BaseVLMSession
from .utils import extract_json, extract_code, extract_program

DEFAULT_MODEL: str = "gpt-4o-2024-08-06"
REASONING_MODELS: list[str] = ["o3-mini", "o4-mini", "gpt-5"]
RETRY_COUNT: int = 10
MAX_IMAGE_SIZE: int = 2048


def _get_env_value(*keys: str) -> str | None:
    """Return the first non-empty environment variable value."""
    for key in keys:
        value = os.getenv(key)
        if value:
            return value.strip()
    return None


def get_default_model_name() -> str:
    """Resolve the default model name from environment or fallback."""
    return _get_env_value("OPENAI_MODEL_NAME", "OPENAI_MODEL") or DEFAULT_MODEL


def get_default_base_url() -> str | None:
    """Resolve the OpenAI-compatible base URL from environment if provided."""
    return _get_env_value("OPENAI_BASE_URL")


class Session(BaseVLMSession):
    """GPT-based VLM session using OpenAI API."""

    def __init__(self, prompts_path, 
                 model: str | None = None,
                 base_url: str | None = None,
                 temperature: float = 0.7, 
                 output_dir: str = "", 
                 prompt_info: dict[str, str] | None = None) -> None:
        """
        Initialize a GPT Session.
        """
        load_dotenv()
        resolved_base_url = base_url or get_default_base_url()
        self.client = OpenAI(base_url=resolved_base_url or None)
        self.model = model or get_default_model_name()
        super().__init__(prompts_path, self.model, temperature, output_dir, prompt_info)
    
    def send(self, task: str, prompt_info: dict[str, str] | None = None,
             info_validate: bool = True, is_json: bool = False, verbose: bool = False,
             images: str | Figure | List[str | Figure] | None = None,
             image_detail: str = "high", append_text: str = "") -> str:
        """
        Send a message of a specific task to the VLM model and return the response.

        Args:
            task: string, the task of the message
            prompt_info: dictionary, the extra information for making the prompt for the task
            info_validate: boolean, whether to validate the input info
            is_json: boolean, whether the response should be in JSON format
            verbose: boolean, whether to print the prompt
            images: string, Figure, or list of them, the image(s) to be sent to the model
            image_detail: string, the detail level of the image
            append_text: string, additional text to append to the prompt

        Returns:
            response: string, the response from the model
        """

        self.logger.debug(f"Sending task: {task}")
        self.past_tasks.append(task)
        prompt = self._make_prompt(task, prompt_info, info_validate)
        if append_text:
            prompt = append_text + "\n\n" + prompt

        if images is not None:
            num_images = len(images) if isinstance(images, list) else 1
        else:
            num_images = 0
        self.logger.debug(f"Past messages: {len(self.past_messages)} Prompt length: {len(prompt)} with {num_images} images")
        if verbose:
            self.logger.debug(f"Prompt:\n{prompt}")
        self._send(prompt, is_json, images, image_detail)
        response = self.past_responses[-1]
        if verbose:
            self.logger.debug(f"Response:\n{response}")

        return response

    def _encode_image(self, image_or_path, detail="auto"):
        """Encode image for VLM models"""
        if isinstance(image_or_path, str):
            img = Image.open(image_or_path)
        elif isinstance(image_or_path, Figure):
            buf = io.BytesIO()
            image_or_path.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
        else:
            self.logger.debug(f"Unsupported image type: {type(image_or_path)}, value: {image_or_path}")
            raise ValueError(f"Warning: Unsupported image type: {type(image_or_path)}. Please provide a file path or a matplotlib Figure.")

        # Optimize image size based on detail level
        if detail == "low":
            target_size = (512, 512)
        elif detail == "high":
            target_size = (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)
        else:
            width, height = img.size
            if width * height <= 512 * 512:
                target_size = (512, 512)
            else:
                target_size = (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)

        # Preserve aspect ratio while resizing
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # removes alpha channel (Ref: https://www.oranlooney.com/post/gpt-cnn/)
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _send(self, new_message: str, json: bool = False, 
              images: Union[str, Figure, List[Union[str, Figure]], None] = None,
              image_detail="high") -> None:
        """Send message to VLM models with image."""
        message_content = []
        
        # Add text content first
        if new_message.strip():
            message_content.append({
                "type": "text",
                "text": new_message
            })
        
        # Handle multiple images
        if images is not None:
            # Convert single image to list for uniform processing and filter out None values
            image_list_raw = images if isinstance(images, list) else [images]
            image_list = [img for img in image_list_raw if img is not None]
            
            for image in image_list:
                try:
                    image_base64 = self._encode_image(image, detail=image_detail)
                except ValueError as exc:
                    self.logger.warning(
                        "Skipping unsupported image in _send (type=%s): %s", type(image), exc
                    )
                    continue
                if image_base64 is None:
                    continue

                message_content.append({
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": image_detail
                }
            })

        self.past_messages.append({"role": "user", "content": message_content})
        
        retries = 0
        max_retries = 3
        while retries < max_retries:
            params = {
                "model": self.model,
                "messages": self.past_messages,
                "temperature": self.temperature if self.model not in REASONING_MODELS else 1.0
            }
            if json:
                params["response_format"] = {"type": "json_object"}

            if self.model in REASONING_MODELS:
                params["reasoning_effort"] = "high"
            
            completion = self.client.chat.completions.create(**params)
            response = completion.choices[0].message.content
            
            if completion.usage:
                self.total_prompt_tokens += completion.usage.prompt_tokens
                self.total_completion_tokens += completion.usage.completion_tokens
                self.total_tokens_this_session += completion.usage.total_tokens
            
            if response is not None:
                self.past_messages.append({"role": "assistant", "content": response})
                self.past_responses.append(response)
                return

            self.logger.info(f"Received None response, retrying... (Attempt {retries + 1}/{max_retries})")
            retries += 1
            
        raise RuntimeError(f"Failed to get a valid response after {max_retries} attempts")
