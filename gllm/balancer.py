import multiprocessing.pool
import os
import threading

import requests

from consts import Endpoints
import data_def
from flask import Flask, Response, request
import utils

##################
# TODO:
# 1. Fix problem with workers come online after load model. Explicitly we would have to
# not send requests to these worker until they are healthy.
# 2. Allow the balancer to do retries on failed reuquests, to different workers if possible
# 2. Boot unhealthy servers that fail too many request
# 3. Allow errors (response code + text) from vllm serve to
# be propagated through worker and balancer to the end client
# 4. Add logging recording to file
# 5. Add tests
# 6. Add an argument to completions/chat-completions to provide a uniue conversation id
# subsequent requests with the conversation id should be sent to the same worker.
# This allows KV caching to make our server go VROOooOOOOOooooM
app = Flask(__name__)


model = ""


class WithLock:
    def __init__(self, entity):
        self.entity = entity
        self.lock = threading.Lock()


worker_queue_size = WithLock({})


def get_least_busy_worker():
    with worker_queue_size.lock:
        worker, _ = min(
            worker_queue_size.entity.items(), key=lambda x: x[1], default=(None, None)
        )
        return worker


@app.route(Endpoints.HEALTH, methods=["GET"])
def health():
    if len(worker_queue_size.entity) == 0:
        return Response("No workers available", status=503)
    for worker in worker_queue_size.entity.keys():
        response = requests.get(worker + Endpoints.HEALTH)
        if response.status_code != 200:
            return Response("Worker unhealthy", status=503)

    return Response("All workers healthy", status=200)


@app.route(Endpoints.ADD_WORKER, methods=["POST"])
def add_worker():
    global model, worker_queue_size

    worker_add_rq = utils.get_request_params(request)
    worker_add_rq = data_def.AddWorkerRequest(**worker_add_rq)

    with worker_queue_size.lock:
        worker_queue_size.entity[worker_add_rq.address] = 0

    return Response(model, status=200)


def register_request_with_worker(worker_address):
    with worker_queue_size.lock:
        worker_queue_size.entity[worker_address] += 1


def notify_worker_request_complete(worker_address):
    with worker_queue_size.lock:
        worker_queue_size.entity[worker_address] -= 1


@app.route(Endpoints.CHAT_COMPLETIONS, methods=["POST"])
def get_chat_completion():
    chat_request = utils.get_request_params(request)
    chat_request = data_def.ChatGenerationRequest(**chat_request)

    worker_address = get_least_busy_worker()

    if worker_address is None:
        return Response("No worker available", status=503)

    register_request_with_worker(worker_address)

    response = requests.post(
        worker_address + Endpoints.CHAT_COMPLETIONS, json=chat_request.model_dump()
    )

    notify_worker_request_complete(worker_address)

    return Response(response.text, status=response.status_code)


@app.route(Endpoints.COMPLETIONS, methods=["POST"])
def get_completion():
    completion_request = utils.get_request_params(request)
    completion_request = data_def.CompletionRequest(**completion_request)

    worker_address = get_least_busy_worker()

    if worker_address is None:
        return Response("No worker available", status=503)

    register_request_with_worker(worker_address)

    response = requests.post(
        worker_address + Endpoints.COMPLETIONS,
        json=completion_request.model_dump(),
    )

    notify_worker_request_complete(worker_address)

    return Response(response.text, status=response.status_code)


@app.route(Endpoints.LOAD_MODEL, methods=["POST"])
def load_model():
    global model
    load_model_rq = utils.get_request_params(request)
    load_model_rq = data_def.LoadModelRequest(**load_model_rq)

    model = load_model_rq.model_path

    failed = False
    for worker in worker_queue_size.entity.keys():
        response = requests.post(
            worker + Endpoints.LOAD_MODEL, json=load_model_rq.model_dump()
        )
        if response.status_code != 200:
            failed = True

    if failed:
        return Response("Failed to load model", status=500)

    return Response("Model load requested", status=200)


@app.route(Endpoints.RELEASE_GPUS, methods=["POST"])
def release_gpus():
    def release_gpu_worker(worker_address):
        response = requests.post(worker_address + Endpoints.RELEASE_GPUS)
        return response.status_code == 200

    with multiprocessing.pool.ThreadPool(len(worker_queue_size.entity)) as pool:
        results = pool.map(release_gpu_worker, worker_queue_size.entity.keys())

    if not all(results):
        return Response("Failed to release gpus", status=500)

    return Response("Gpus released", status=200)


if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    port = int(port)
    app.run(port=port, threaded=True, host="0.0.0.0")
