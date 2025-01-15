from dataclasses import dataclass
import time
from typing import Optional

import requests

from gllm import data_def


class GLLM:
    def __init__(self, server_address):
        while server_address[-1] == "/":
            server_address = server_address[:-1]
        self.server_address = server_address

        # TODO: To deal with the statefulness which continues beyond the life of this
        # object (i.e between success script runs) we need to fetch this from 
        # the server. This is a temporary solution. 
        self.last_loaded_model = None

    def get_completions(
        self,
        model: str,
        prompt: str,
        best_of: Optional[int] = None,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        logit_bias: dict = {},
        logprobs: Optional[int] = None,
        max_tokens: int = 16,
        n: int = 1,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        stop: Optional[list[str] | str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_mode: str = "openai",
    ):
        completion_request = data_def.CompletionRequest(
            model=model,
            prompt=prompt,
            best_of=best_of,
            echo=echo,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        response = requests.post(
            self.server_address + "/completions",
            json=completion_request.model_dump_json(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get completions. Status code: {response.status_code}\n"
                f"Response: {response.text}"
            )
        response = data_def.CompletionResponse(**response.json())
        if return_mode == "primitives":
            return [choice.text for choice in response.choices]
        elif return_mode == "openai":
            return response
        else:
            raise ValueError(
                f"Unknown return mode: {return_mode}, must be 'openai' or 'primitives'"
            )

    def get_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        return_mode: str = "openai",
    ):
        chat_request = data_def.ChatGenerationRequest(
            name_of_model=model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
        )

        response = requests.post(
            self.server_address + "/chat_completion",
            json=chat_request.model_dump_json(),
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get chat completion. Status code: {response.status_code}\n"
                f"Response: {response.text}"
            )
        if return_mode == "primitives":
            return [choice for choice in response.json()["choices"]]
        elif return_mode == "openai":
            response = Response(
                choices=[
                    MessageWrapper(Message(**choice))
                    for choice in response.json()["choices"]
                ]
            )
        else:
            raise ValueError(
                f"Unknown return mode: {return_mode}, must be 'openai' or 'primitives'"
            )

        return response

    def load_model(self, model_identifier: str):
        request = data_def.LoadModelRequest(model_path=model_identifier)
        response = requests.post(
            self.server_address + "/load_model", json=request.model_dump_json()
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to load model. Status code: {response.status_code}\n"
                f"Response: {response.text}"
            )
        
        self.last_loaded_model = model_identifier
        return response.text

    def is_healthy(self, timeout: int = 5):
        try:
            response = requests.get(self.server_address + "/health", timeout=timeout)
        except requests.RequestException as e:
            return False

        if response.status_code == 200:
            return True
        return False

    def wait_for_health(self, timeout: int = 300, check_interval: int = 5):
        for _ in range(timeout // check_interval):
            if self.is_healthy():
                return True
            time.sleep(check_interval)
            print("Waiting for server to be healthy ...")

        raise TimeoutError("Server did not become healthy in time")

    def wait_for_live(self, timeout: int = 300, check_interval: int = 5):
        """Wait until /health returns with any return code."""
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                response = requests.get(
                    self.server_address + "/health", timeout=check_interval
                )
                return
            except requests.RequestException:
                pass
            time.sleep(check_interval)
            print("Waiting for server to come live ...")

        raise TimeoutError("Server did not come live in time")


# The following classes allow us to define an object
# with a structure that mimics the openAI response.
# Quack Quack.
@dataclass
class Message:
    content: str
    role: str


@dataclass
class MessageWrapper:
    message: Message


@dataclass
class Response:
    choices: list[MessageWrapper]
