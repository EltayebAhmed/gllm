import os
import threading

import requests

from flask import Flask, Response, request

import data_def
import utils

##################
# TODO:
# 1. Fix problem with workers come online after load model. This would involve
# not sending requests to workers until they are healthy.
# 2. Boot unhealthy servers that fail roo many request
# 3. Allow errors (response code + text) from vllm serve to
# be propagated through worker and balancer to the end client

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


@app.route("/health", methods=["GET"])
def health():
    if len(worker_queue_size.entity) == 0:
        return Response("No workers available", status=503)
    for worker in worker_queue_size.entity.keys():
        response = requests.get(worker + "/health")
        if response.status_code != 200:
            return Response("Worker unhealthy", status=503)

    return Response("All workers healthy", status=200)


@app.route("/add_worker", methods=["POST"])
def add_worker():
    global model, worker_queue_size

    worker_add_rq = utils.get_request_params(request)
    worker_add_rq = data_def.AddWorkerRequest(**worker_add_rq)

    with worker_queue_size.lock:
        worker_queue_size.entity[worker_add_rq.address] = 0

    return Response(model, status=200)


@app.route("/chat_completion", methods=["POST"])
def get_chat_completion():
    chat_request = utils.get_request_params(request)
    chat_request = data_def.ChatGenerationRequest(**chat_request)

    worker_address = get_least_busy_worker()

    if worker_address is None:
        return Response("No worker available", status=503)

    with worker_queue_size.lock:
        worker_queue_size.entity[worker_address] += 1

    response = requests.post(
        worker_address + "/chat_completion", json=chat_request.model_dump_json()
    )

    with worker_queue_size.lock:
        worker_queue_size.entity[worker_address] -= 1

    return Response(response.text, status=response.status_code)


@app.route("/load_model", methods=["POST"])
def load_model():
    global model
    load_model_rq = utils.get_request_params(request)
    load_model_rq = data_def.LoadModelRequest(**load_model_rq)

    model = load_model_rq.model_path

    failed = False
    for worker in worker_queue_size.entity.keys():
        response = requests.post(
            worker + "/load_model", json=load_model_rq.model_dump_json()
        )
        if response.status_code != 200:
            failed = True

    if failed:
        return Response("Failed to load model", status=500)

    return Response("Model load requested", status=200)


if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    port = int(port)
    app.run(port=port, threaded=True, host="0.0.0.0")
