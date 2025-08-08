import multiprocessing.pool
import os
import threading
from typing import Optional

import requests

from flask import Blueprint, Flask, Response, request

from . import data_def
from . import utils
from .balancer_utils import LRUCache
from .consts import Endpoints, SpecialFields

##################
# TODO:
# 1. Fix problem with workers come online after load model. Explicitly we would have to
# not send requests to these worker until they are healthy.
# 2. Allow the balancer to do retries on failed reuquests, to different workers if possible
# 2. Boot unhealthy servers that fail too many request
# 3. Allow errors (response code + text) from vllm serve to
# be propagated through worker and balancer to the end client
# 4. Add logging recording to file
# 6. Add an argument to completions/chat-completions to provide a uniue conversation id
# subsequent requests with the conversation id should be sent to the same worker.
# This allows KV caching to make our server go VROOooOOOOOooooM

app = Flask(__name__)

# Create a blueprint for all endpoints
api_blueprint = Blueprint('api', __name__)

model = ""


class WithLock:
    def __init__(self, entity):
        self.entity = entity
        self.lock = threading.Lock()


worker_queue_size = WithLock({})




def _get_routing_cache_capacity() -> int:
    try:
        return int(os.environ.get("ROUTING_CACHE_SIZE", 5000))
    except Exception:
        return 5000


# Maps conversation_id -> worker address using an LRU eviction policy
conversation_routing_cache = WithLock(LRUCache(_get_routing_cache_capacity()))


def get_least_busy_worker():
    with worker_queue_size.lock:
        worker, _ = min(
            worker_queue_size.entity.items(), key=lambda x: x[1], default=(None, None)
        )
        return worker


def get_worker_for_conversation(conversation_id: Optional[str]) -> Optional[str]:
    # No conversation id: normal least-busy routing
    if not conversation_id:
        return get_least_busy_worker()

    # Try to reuse the same worker for this conversation
    with conversation_routing_cache.lock:
        cached_worker = conversation_routing_cache.entity.get(conversation_id)

    if cached_worker and cached_worker in worker_queue_size.entity:
        return cached_worker

    # Fallback: choose least busy and remember it
    worker = get_least_busy_worker()
    if worker is None:
        return None

    with conversation_routing_cache.lock:
        conversation_routing_cache.entity.put(conversation_id, worker)
    return worker


@api_blueprint.route(Endpoints.HEALTH, methods=["GET"])
def health():
    if len(worker_queue_size.entity) == 0:
        return Response("No workers available", status=503)
    for worker in worker_queue_size.entity.keys():
        response = requests.get(worker + Endpoints.HEALTH)
        if response.status_code != 200:
            return Response("Worker unhealthy", status=503)

    return Response("All workers healthy", status=200)


@api_blueprint.route(Endpoints.ADD_WORKER, methods=["POST"])
def add_worker():
    global model, worker_queue_size

    try:
        worker_add_rq = utils.get_request_params(request)
        worker_add_rq = data_def.AddWorkerRequest(**worker_add_rq)
    except Exception as e:
        return Response(f"Invalid request data: {str(e)}", status=400)

    with worker_queue_size.lock:
        worker_queue_size.entity[worker_add_rq.address] = 0

    return Response(model, status=200)


def register_request_with_worker(worker_address):
    with worker_queue_size.lock:
        worker_queue_size.entity[worker_address] += 1


def notify_worker_request_complete(worker_address):
    with worker_queue_size.lock:
        worker_queue_size.entity[worker_address] -= 1


@api_blueprint.route(Endpoints.CHAT_COMPLETIONS, methods=["POST"])
def get_chat_completion():
    chat_request = utils.get_request_params(request)
    conversation_id = chat_request.pop(SpecialFields.CONVERSATION_ID, None)   
    chat_request = data_def.ChatGenerationRequest(**chat_request)

    worker_address = get_worker_for_conversation(conversation_id)

    if worker_address is None:
        return Response("No worker available", status=503)

    register_request_with_worker(worker_address)

    response = requests.post(
        worker_address + Endpoints.CHAT_COMPLETIONS, json=chat_request.model_dump()
    )

    notify_worker_request_complete(worker_address)
    return Response(response.text, status=response.status_code, mimetype=response.headers.get('content-type', 'application/json'))


@api_blueprint.route(Endpoints.COMPLETIONS, methods=["POST"])
def get_completion():
    completion_request = utils.get_request_params(request)
    conversation_id = completion_request.pop(SpecialFields.CONVERSATION_ID, None)
    completion_request = data_def.CompletionRequest(**completion_request)

    worker_address = get_worker_for_conversation(conversation_id)

    if worker_address is None:
        return Response("No worker available", status=503)

    register_request_with_worker(worker_address)

    response = requests.post(
        worker_address + Endpoints.COMPLETIONS,
        json=completion_request.model_dump(),
    )

    notify_worker_request_complete(worker_address)
    
    return Response(response.text, status=response.status_code, mimetype="application/json")


@api_blueprint.route(Endpoints.LOAD_MODEL, methods=["POST"])
def load_model():
    global model
    load_model_rq = utils.get_request_params(request)
    load_model_rq = data_def.LoadModelRequest(**load_model_rq)

    model = load_model_rq.model_path

    failed_workers = []
    for worker in worker_queue_size.entity.keys():
        response = requests.post(
            worker + Endpoints.LOAD_MODEL, json=load_model_rq.model_dump()
        )
        if response.status_code != 200:
            failed_workers.append(worker)

    if failed_workers:
        formatted_failed_workers = "\n".join(failed_workers)
        return Response(f"Failed to load model on {len(failed_workers)} workers:\n {formatted_failed_workers}.", status=500)

    return Response("Model load requested successfully.", status=200)


@api_blueprint.route(Endpoints.RELEASE_GPUS, methods=["POST"])
def release_gpus():
    def release_gpu_worker(worker_address):
        response = requests.post(worker_address + Endpoints.RELEASE_GPUS)
        return response.status_code == 200

    # Handle case where there are no workers
    if len(worker_queue_size.entity) == 0:
        return Response("No workers to release gpus from.", status=200)

    with multiprocessing.pool.ThreadPool(len(worker_queue_size.entity)) as pool:
        results = pool.map(release_gpu_worker, worker_queue_size.entity.keys())

    if not all(results):
        return Response("Failed to release gpus", status=500)

    return Response("Gpus released", status=200)


# Register the blueprint twice - once without prefix and once with /v1 prefix
app.register_blueprint(api_blueprint)
app.register_blueprint(api_blueprint, url_prefix='/v1', name='api_v1')


if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    port = int(port)
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(port=port, threaded=True, host=host)
