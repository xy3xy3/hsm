"""
Base VLM Session Class

This module provides a base class for VLM sessions with shared functionality
across different model implementations (GPT, Qwen, etc.).
"""

from __future__ import annotations
import datetime
import hashlib
import json
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import yaml

class BaseVLMSession(ABC):
    """Base class for VLM sessions with shared functionality across different models."""

    # Class-level default output directory
    SESSION_OUTPUT_DIR = ""

    @classmethod
    def set_global_output_dir(cls, dir_path: str) -> None:
        """Set a global output directory for all Session instances"""
        cls.SESSION_OUTPUT_DIR = dir_path
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    def __init__(self, prompts_path: str, model: str, temperature: float = 0.7,
                 output_dir: str = "", prompt_info: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize a VLM Session.

        Args:
            prompts_path: Path to the YAML file containing prompts
            model: Model name to use
            temperature: Sampling temperature for the model
            output_dir: Directory to save session logs
            prompt_info: Optional dictionary to replace placeholders in the system prompt
        """
        if not output_dir and self.SESSION_OUTPUT_DIR:
            output_dir = self.SESSION_OUTPUT_DIR

        from hsm_core.utils import get_logger
        self.logger = get_logger('vlm.session')

        self.prompts_dir = prompts_path
        self.model = model
        self.predefined_prompts = self._load_prompts(prompts_path)

        # Replace placeholders in the system prompt with the provided prompt_info
        if prompt_info and "system" in self.predefined_prompts.keys():
            system_prompt = self.predefined_prompts["system"]
            for key, value in prompt_info.items():
                placeholder = f"<{key.upper()}>"
                if placeholder in system_prompt:
                    system_prompt = system_prompt.replace(placeholder, str(value))
                    self.logger.debug(f"Initialized system prompt with {key}: {value}")
                else:
                    self.logger.warning(f"Placeholder {placeholder} not found in system prompt")
            self.predefined_prompts["system"] = system_prompt

        self.past_tasks: List[str] = []
        self.past_messages: List[Dict[str, str]] = []
        self.past_responses: List[str] = []
        self.clear_past_messages()
        self.temperature = temperature
        self.output_dir = output_dir

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens_this_session = 0

        prompts_path_obj = pathlib.Path(self.prompts_dir)
        self._session_filename = f"{self.output_dir}/session_{prompts_path_obj.stem}.txt"

    def _load_prompts(self, prompts_file: str) -> Dict[str, str]:
        """
        Load predefined prompts from a YAML file.

        Args:
            prompts_file: str, path to the YAML file

        Returns:
            dict[str, str]: Dictionary of predefined prompts
        """
        with open(prompts_file) as file:
            return yaml.safe_load(file)

    def clear_past_messages(self) -> None:
        """
        Reset the conversation context by clearing past messages and responses.
        """
        self.past_messages = [{"role": "system", "content": self.predefined_prompts["system"]}]

    def add_feedback(self, feedback: str) -> None:
        """
        Add a feedback to the past responses.
        """
        self.past_messages.append({"role": "user", "content": feedback + " Please try again."})
    
    @abstractmethod
    def send(self, task: str, prompt_info: Optional[Dict[str, str]] = None,
             info_validate: bool = True, is_json: bool = False, verbose: bool = False,
             images: Union[str, Any, List[Union[str, Any]], None] = None,
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
        pass

    def send_with_validation(self, task: str,
                             prompt_info: Optional[Dict[str, Any]] = None,
                             validation: Optional[Callable[[Any], tuple[bool, str, int]]] = None,
                             retry: int = 5,
                             images: Union[str, Any, List[Union[str, Any]], None] = None,
                             image_detail: str = "high",
                             is_json: bool = False,
                             verbose: bool = False) -> str:
        """
        Send a message of a specific task and return the response after validating it.

        Args:
            task: string, the task of the message
            prompt_info: dictionary, the extra information for making the prompt for the task
            validation: function, the validation function to validate the response for the task
            retry: integer, the number of retries for the task
            images: string, Figure, or list of them, the image(s) to be sent to the model
            image_detail: string, the detail level of the image
            is_json: boolean, whether the response should be in JSON format
            verbose: boolean, whether to print the prompt

        Returns:
            response: string, the response from the model
        """
        response = self.send(task, prompt_info, images=images, image_detail=image_detail,
                          is_json=is_json, verbose=verbose)

        count = 0
        while count <= retry:
            if validation is not None:
                valid, error_message, error_index = validation(response)

                if not valid:
                    self.logger.info(f"Validation failed for task {task} at try {count+1}")
                    # Skip logging WN synset key validation errors
                    if error_message and not error_message.startswith("The WordNet synset key"):
                        self.logger.info(f"Validation error: {error_message}")

                    if count < retry:
                        count += 1
                        self.logger.info(f"Retrying task {task} [try {count+1} / {retry}]")
                        # Get the specific retry prompt if available (by order of appearance in the prompts file)
                        retry_prompt_keys = [key for key in self.predefined_prompts.keys() if task in key and "feedback" in key]
                        if retry_prompt_keys:
                            retry_task_name = retry_prompt_keys[error_index]
                        else:
                            # If there is no specific retry prompt, use the generic one
                            retry_task_name = "invalid_response"

                        response = self.send(retry_task_name, {"feedback": error_message}, is_json=is_json)
                        # Continue to next iteration for validation
                    else:
                        # No more retries available
                        break
                else:
                    self.logger.info(f"Validation passed for task {task} at try {count+1}")
                    break
            else:
                break  # No validation function, assume valid

        if count >= retry:
            raise RuntimeError(f"$ --- Validation failed for task {task} after {retry} retries")

        return response

    def _execute_with_retry(self, task: str, initial_response: str,
                           validation: Optional[Callable[[Any], tuple[bool, str, int]]],
                           retry: int, is_json: bool, **kwargs) -> str:
        """
        Execute validation and retry logic for a response.

        Args:
            task: the task name
            initial_response: the initial response to validate
            validation: validation function
            retry: maximum number of retries
            is_json: whether response should be JSON
            **kwargs: additional arguments for send

        Returns:
            final response after validation/retry
        """
        response = initial_response
        count = 0

        while count <= retry:
            if validation is not None:
                valid, error_message, error_index = validation(response)

                if not valid:
                    self._log_validation_failure(task, count + 1, error_message)

                    if count < retry:
                        count += 1
                        retry_prompt = self._find_retry_prompt(task, error_index)
                        response = self._send_retry_request(
                            retry_prompt, error_message, is_json, **kwargs
                        )
                    else:
                        break
                else:
                    self.logger.info(f"Validation passed for task {task} at try {count+1}")
                    break
            else:
                break  # No validation function, assume valid

        if count >= retry:
            raise RuntimeError(f"$ --- Validation failed for task {task} after {retry} retries")

        return response

    def _log_validation_failure(self, task: str, attempt: int, error_message: str) -> None:
        """Log validation failure with appropriate filtering."""
        self.logger.info(f"Validation failed for task {task} at try {attempt}")
        if error_message and not error_message.startswith("The WordNet synset key"):
            self.logger.info(f"Validation error: {error_message}")

    def _find_retry_prompt(self, task: str, error_index: int) -> str:
        """Find the appropriate retry prompt for a task."""
        retry_prompt_keys = [key for key in self.predefined_prompts.keys()
                           if task in key and "feedback" in key]
        if retry_prompt_keys:
            return retry_prompt_keys[error_index]
        else:
            return "invalid_response"

    def _send_retry_request(self, retry_prompt: str, error_message: str,
                           is_json: bool, **kwargs) -> str:
        """Send a retry request with error feedback."""
        retry_kwargs = kwargs.copy()
        retry_kwargs.pop('use_cached', None)  # Remove use_cached for retries
        retry_kwargs['is_json'] = is_json

        return self.send(retry_prompt, {"feedback": error_message}, **retry_kwargs)

    def _make_prompt(self, task: str, prompt_info: Optional[Dict[str, str]], info_validate: bool = True) -> str:
        """
        Make a prompt for the VLM model.

        Args:
            task: string, the task of the prompt
            prompt_info: dictionary, the extra information for making the prompt for the task
            info_validate: boolean, whether to validate the input info

        Returns:
            prompt: string, the prompt for the VLM model
        """
        prompt = self.predefined_prompts[task]
        if info_validate:
            self._validate_prompt_info(task, prompt_info)

        # Replace the placeholders in the prompt with the information
        if prompt_info is not None:
            for key in prompt_info:
                prompt = prompt.replace(f"<{key.upper()}>", str(prompt_info[key]))

        return prompt

    def _validate_prompt_info(self, task: str, prompt_info: Optional[Dict[str, str]]) -> None:
        """
        Validate that required information is provided for the task.

        Args:
            task: string, the task of the prompt
            prompt_info: dictionary, the extra information for making the prompt

        Raises:
            ValueError: If required information is missing
        """
        valid = True
        match task:
            case "wnsynsetkey":
                valid = all(key in prompt_info for key in ["obj_label", "wnsynsetkeys"]) if prompt_info else False
            case "wnsynsetkeys":
                valid = all(key in prompt_info for key in ["wnsynsetkeys", "object_labels"]) if prompt_info else False

        if not valid:
            raise ValueError(f"Extra information is required for the task: {task}")

    def save_session(self, filename: Optional[str] = None) -> str:
        """
        Save the current session to a JSON file.

        Args:
            filename: Optional string, custom filename to save the session.
                     If None, uses the default session filename with .json extension.

        Returns:
            str: Path to the saved session file
        """
        filename = self._prepare_session_filename(filename)
        session_data = self._create_session_data()
        self._process_session_messages(session_data)
        self._save_session_to_file(session_data, filename)
        return filename

    def _prepare_session_filename(self, filename: Optional[str]) -> str:
        """Prepare the session filename and create necessary directories."""
        output_dir = self.output_dir if self.output_dir else self.SESSION_OUTPUT_DIR
        if output_dir:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        if filename is None:
            prompts_path = pathlib.Path(self.prompts_dir)
            filename = f"session_{prompts_path.stem}.json"
        
        if output_dir:
            filename = os.path.join(output_dir, filename)

        dir_path = os.path.dirname(os.path.abspath(filename))
        os.makedirs(dir_path, exist_ok=True)

        return filename

    def _create_session_data(self) -> Dict[str, Any]:
        """Create the basic session data structure."""
        return {
            "token_usage": self.get_session_token_usage(),
            "model": self.model,
            "temperature": self.temperature,
            "timestamp": datetime.datetime.now().isoformat(),
            "messages": [],
        }

    def _process_session_messages(self, session_data: Dict[str, Any]) -> None:
        """Process and deduplicate session messages."""
        processed_prompts = set()
        tasks_with_responses = list(zip(self.past_tasks, self.past_responses))

        for i, (task, response) in enumerate(tasks_with_responses):
            user_msg, assistant_msg = self._get_message_pair(i)
            if user_msg is None:
                continue

            prompt_hash = self._calculate_prompt_hash(user_msg)

            # Only save the latest response for each unique prompt
            if prompt_hash not in processed_prompts:
                processed_prompts.add(prompt_hash)
                exchange = self._create_message_exchange(task, prompt_hash, user_msg, assistant_msg, response)
                session_data["messages"].append(exchange)

    def _get_message_pair(self, index: int) -> tuple[Optional[Dict], Optional[Dict]]:
        """Get the user and assistant message pair for a given index."""
        if index * 2 + 1 >= len(self.past_messages):
            return None, None

        user_msg = self.past_messages[index * 2 + 1] if index > 0 else self.past_messages[1]
        assistant_msg = self.past_messages[index * 2 + 2] if index > 0 else self.past_messages[2]

        return user_msg, assistant_msg

    def _calculate_prompt_hash(self, user_msg: Dict) -> str:
        """Calculate a hash for the user message content."""
        return hashlib.md5(str(user_msg["content"]).encode()).hexdigest()[:8]

    def _create_message_exchange(self, task: str, prompt_hash: str,
                               user_msg: Dict, assistant_msg: Dict, response: str) -> Dict:
        """Create a message exchange dictionary."""
        return {
            "task": task,
            "prompt_hash": prompt_hash,
            "user_message": user_msg,
            "assistant_message": assistant_msg,
            "response": response
        }

    def _save_session_to_file(self, session_data: Dict[str, Any], filename: str) -> None:
        """Save session data to a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

    def get_session_token_usage(self) -> Dict[str, int]:
        """
        Get the accumulated token usage for the current session.

        Returns:
            dict: A dictionary containing total_prompt_tokens,
                  total_completion_tokens, and total_tokens_this_session.
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens_this_session": self.total_tokens_this_session
        }
