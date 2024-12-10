from dataclasses import dataclass
import time

import requests

from gllm import data_def



class DistributionServerInterface:
    def __init__(self, server_address):
        while server_address[-1] == "/":
            server_address = server_address[:-1]
        self.server_address = server_address

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
# with a structure that mimics the openAI response. Quack Quack.
@dataclass
class Message:
    content: str
    role: str


@dataclass
class MessageWrapper:
    message: Message


@dataclass
class Response:
    choices: list[Message]
