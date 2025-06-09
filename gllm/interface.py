from dataclasses import dataclass
import logging
import time
from typing import Optional

import backoff
import requests

from gllm import data_def
from gllm.consts import Endpoints


class RemoteError(Exception):
    """Raised when the server returns an error response."""


class GLLM:
    def __init__(self, server_address: str, api_key: Optional[str] = None):
        while server_address[-1] == "/":
            server_address = server_address[:-1]
        self.server_address = server_address
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        # TODO: To deal with the statefulness which continues beyond the life of this
        # object (i.e between success script runs) we need to fetch this from
        # the server. This is a temporary solution.
        # A few months latet this temporary solution is still around.
        self.last_loaded_model = None

    # TODO: tighten up. We need requests.exceptions.ConnectTimeout
    # and not sure what else.
    @backoff.on_exception(backoff.expo, Exception, max_time=1000)
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
            self.server_address + Endpoints.COMPLETIONS,
            json=completion_request.model_dump(),
            headers=self.headers,
        )
        if response.status_code != 200:
            raise RemoteError(
                f"Failed to get completions.\nStatus code: {response.status_code}\n"
                f"Response: {response.text}"
            )

        try:
            response = data_def.CompletionResponse(**response.json())
        except Exception as e:
            raise RemoteError(
                f"Failed to parse response.\nStatus code: {response.status_code}\n"
                f"This typically happens due to being rate limited.\n"
                f"Response: {response.text}\n"
                f"Error: {e}"
            )
        if return_mode == "primitives":
            return [choice.text for choice in response.choices]
        elif return_mode == "openai":
            return response
        elif return_mode == "raw":
            return response
        else:
            raise ValueError(
                f"Unknown return mode: {return_mode}, must be 'openai' or 'primitives'"
            )

    @backoff.on_exception(backoff.expo, RemoteError, max_time=60)
    def get_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        n: int = 1,
        stop: Optional[list[str] | str] = None,
        return_mode: str = "openai",
    ):
        chat_request = data_def.ChatGenerationRequest(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=stop,
        )

        response = requests.post(
            self.server_address + Endpoints.CHAT_COMPLETIONS,
            json=chat_request.model_dump(),
            headers=self.headers,
        )

        if response.status_code != 200:
            raise RemoteError(
                f"Failed to get chat completion. Status code: {response.status_code}\n"
                f"Response: {response.text}"
            )
        print(response.text)
        print(f"Chat response: {response.json()}")
        choices = response.json()["choices"]
        # OpenAI and Vllm have different response structures
        # This code tries to handle that but further research might be
        # needed to ensure that this is correct.
        if choices and "message" in choices[0]:
            choices = [choice["message"] for choice in choices]

        if return_mode == "primitives":
            return [choice for choice in choices]
        elif return_mode == "openai":
            return Response(
                choices=[MessageWrapper(Message(**choice)) for choice in choices]
            )

        raise ValueError(
            f"Unknown return mode: {return_mode}, must be 'openai' or 'primitives'"
        )

    def load_model(
        self, model_identifier: str, force_reload: bool = False, timeout: int = 10
    ):
        """Load a model onto a GLLM worker.

        Note: Function only supported for GLLM worker backend.
        Args:
            model_identifier (str): Identifier of the model to load. Can be a
                path to the model or a huggingface model identifier. Path
                must be accessible by the worker.
        """
        request = data_def.LoadModelRequest(
            model_path=model_identifier, force_reload=force_reload
        )
        response = requests.post(
            self.server_address + Endpoints.LOAD_MODEL,
            json=request.model_dump(),
            timeout=timeout,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to load model. Status code: {response.status_code}\n"
                f"Response: {response.text}"
            )

        self.last_loaded_model = model_identifier
        return response.text

    def is_healthy(self, timeout: int = 5):
        """Return True if the server is healthy, False otherwise.

        Not supported for all backends. TODO: If not supported, always return True."""
        try:
            response = requests.get(
                self.server_address + Endpoints.HEALTH,
                timeout=timeout,
                headers=self.headers,
            )
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
            logging.info("Waiting for server to be healthy ...")

        raise TimeoutError("Server did not become healthy in time")

    def wait_for_live(self, timeout: int = 300, check_interval: int = 5):
        """Wait until /health returns with any return code."""
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                requests.get(
                    self.server_address + Endpoints.HEALTH,
                    timeout=check_interval,
                    headers=self.headers,
                )
                return
            except requests.RequestException:
                pass
            time.sleep(check_interval)
            logging.info("Waiting for server to come live ...")

        raise TimeoutError("Server did not come live in time")

    def release_gpus(self, timeout: int = 20):
        response = requests.post(
            self.server_address + Endpoints.RELEASE_GPUS,
            headers=self.headers,
            timeout=timeout,
            # No retries for release_gpus
        )
        if response.status_code != 200:
            raise RemoteError(
                f"Failed to release GPUs. Status code: {response.status_code}\n"
                f"Response: {response.text}"
            )


# The following classes allow us to define an object
# with a structure that mimics the openAI response.
# Quack quack.
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
