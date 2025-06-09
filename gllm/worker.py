import dataclasses
import logging
import os
import signal
import subprocess
import sys
import threading

import openai
import psutil
import torch
import requests

from gllm import data_def
from gllm.consts import Endpoints
from flask import Flask, Response, request
from gllm import utils

vllm_process = None
model_path = ""
app = Flask(__name__)

start_and_kill_lock = threading.Lock()


def signal_handler(signum, frame):
    """Handle termination signals to clean up vLLM process."""
    print(f"\nReceived signal {signum}, shutting down worker...")
    logging.info(f"Worker received signal {signum}, terminating vLLM process")
    terminate_vllm()
    logging.info("Worker terminated gracefully")
    sys.exit(0)


def kill_process_and_children(pid: int):
    try:
        # Access the process
        parent = psutil.Process(pid)
        # Iterate over children and terminate each
        for child in parent.children(recursive=True):
            print(f"Terminating child process {child.pid}")
            child.terminate()
        parent.terminate()

        # Wait for termination, then use kill if any still exist
        gone, alive = psutil.wait_procs(
            [parent] + parent.children(recursive=True), timeout=5
        )
        for p in alive:
            p.kill()
    except psutil.NoSuchProcess:
        pass  # Process has already been terminated


def terminate_vllm():
    global vllm_process
    if vllm_process is None:
        return
    with start_and_kill_lock:
        kill_process_and_children(vllm_process.pid)
        vllm_process.terminate()
        vllm_process.wait()
        vllm_process = None


def spin_up_vllm(model_path: str, vllm_port: int):
    global vllm_process
    if vllm_process is not None:
        terminate_vllm()
    with start_and_kill_lock:
        print(f"Starting VLLM process on port {vllm_port}")
        n_gpus = torch.cuda.device_count()
        vllm_process = subprocess.Popen(
            [
                "vllm",
                "serve",
                model_path,
                "--enable-prefix-caching",
                "--max-model-len",
                "14000",
                "--port",
                str(vllm_port),
                "--pipeline-parallel-size",
                str(n_gpus),
                "--disable-log-stats",
                "--disable-log-requests",
                "--uvicorn-log-level=error",
                "--trust-remote-code",
            ]
        )

    logging.info(f"VLLM process started with PID: {vllm_process.pid}")


@dataclasses.dataclass
class Config:
    router_address: str
    self_hostname: str
    port: int
    vllm_port: int


def get_config():
    router_address = os.environ.get("ROUTER_ADDRESS")
    if router_address is None:
        raise ValueError("ROUTER_ADDRESS is not set.")

    logging.info(f"Router address: {router_address}")
    port = os.environ.get("PORT", 8976)
    port = int(port)
    logging.info(f"Port: {port}")

    self_hostname = os.environ.get("SELF_HOSTNAME", "localhost")
    logging.info(f"Self hostname: {self_hostname}")

    vllm_port = os.environ.get("VLLM_PORT", None)
    if vllm_port is None:
        vllm_port = 8087
    vllm_port = int(vllm_port)

    logging.info(f"VLLM port: {vllm_port}")

    return Config(router_address, self_hostname, port, vllm_port)


def register_worker(router_address: str, self_hostname: str, port):

    worker_add_rq = data_def.AddWorkerRequest(address=f"http://{self_hostname}:{port}")
    response = requests.post(
        f"{router_address}/add_worker", json=worker_add_rq.model_dump()
    )

    if response.status_code != 200:
        raise ValueError(
            f"Failed to register worker. Status code: {response.status_code}"
            f"Response: {response.text}"
        )

    logging.info("Worker registered successfully")
    model = response.text  # todo make this a json response {model_name: ....}
    if model != "":
        logging.info(f"Model to initialize with: {model}")
        spin_up_vllm(model, config.vllm_port)


@app.route(Endpoints.LOAD_MODEL, methods=["POST"])
def load_model():
    global config
    global model_path
    load_model_rq = utils.get_request_params(request)
    load_model_rq = data_def.LoadModelRequest(**load_model_rq)
    logging.warning(f"Load model request: {load_model_rq}")
    if load_model_rq.model_path == model_path and not load_model_rq.force_reload:
        return Response("Model already loaded", status=200)

    spin_up_vllm(load_model_rq.model_path, config.vllm_port)
    model_path = load_model_rq.model_path
    return Response("Model load requested", status=200)


def openai_messages_to_chat_gen_resp(response):
    choices = []
    for choice in response.choices:
        choices.append({"role": choice.message.role, "content": choice.message.content})
    return data_def.ChatGenerationResponse(choices=choices)


@app.route(Endpoints.HEALTH, methods=["GET", "POST"])
def health():
    try:
        response = requests.get(f"http://localhost:{config.vllm_port}/health")
    except requests.RequestException as e:
        return Response(str(e), status=503)
    return Response(response.text, status=response.status_code)


@app.route(Endpoints.CHAT_COMPLETIONS, methods=["POST"])
def get_chat_completion():
    global config, increment
    chat_request = utils.get_request_params(request)
    print(f"Chat request: {chat_request}")
    chat_request = data_def.ChatGenerationRequest(**chat_request)
    vllm_address = f"http://{config.self_hostname}:{config.vllm_port}/v1"

    # TODO Make this also REST API. Makes it easier to perculate up
    # response codes and error messages
    client = openai.OpenAI(base_url=vllm_address, api_key="FREE_TOKENS_FOR_ALL")
    # This is horrible. We should be using the vllm API directly.
    response = client.chat.completions.create(  # type: ignore
        model=chat_request.model,
        messages=dict(chat_request)["messages"],
        max_tokens=chat_request.max_tokens,
        temperature=chat_request.temperature,
        n=chat_request.n,
        stop=chat_request.stop,
    )
    # Iterate over response to turn choices into python primitives
    response = openai_messages_to_chat_gen_resp(response)

    return Response(response.json(), status=200, mimetype="application/json")


@app.route(Endpoints.COMPLETIONS, methods=["POST"])
def get_completion():
    global config
    completion_request = utils.get_request_params(request)
    completion_request = data_def.CompletionRequest(**completion_request)
    vllm_address = f"http://{config.self_hostname}:{config.vllm_port}/v1{Endpoints.COMPLETIONS.value}"
    response = requests.post(vllm_address, json=completion_request.model_dump())
    
    return Response(response.text, status=response.status_code, mimetype="application/json")


@app.route(Endpoints.RELEASE_GPUS, methods=["POST"])
def release_gpus():
    global model_path
    terminate_vllm()
    model_path = ""
    return Response("ok", status=200)


if __name__ == "__main__":
    config = get_config()
    import time

    n_retries = 5
    time_between_retries = 5
    time.sleep(5)
    for i in range(n_retries):
        try:
            register_worker(config.router_address, config.self_hostname, config.port)
            break
        except Exception as e:
            logging.error(f"Failed to register worker: {str(e)}")
            if i == n_retries - 1:
                raise e

        time.sleep(time_between_retries)

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        app.run(port=config.port, threaded=True, host="0.0.0.0")
    finally:
        terminate_vllm()
        logging.info("Worker terminated")
