# GLLM - Distributed Language Model Serving

GLLM is a distributed inference system for large language models that provides automatic load balancing across multiple GPU workers. Built on vLLM, it offers OpenAI-compatible APIs with intelligent request routing, automatic retries, and health monitoring. Dynamically load or swap models across your entire cluster with a single API call, making it perfect for high-throughput inference workloads requiring horizontal scaling and model flexibility.

## Key Features

- 🔄 Automatic load balancing with least-loaded queue selection
- 🚀 Built on vLLM for high-performance inference
- 🔌 OpenAI-compatible API endpoints
- 🔁 Automatic retries with exponential backoff
- 📊 Per-worker health monitoring
- 🔀 Dynamic model loading/switching across entire cluster from single endpoint
- 🎯 Easy CLI for cluster management
- 🐍 Python client library included

## Installation

```bash
pip install git+https://github.com/EltayebAhmed/gllm
```

**Verify installation** (optional):
```bash
gllm --version  # Should show version 0.1.0
```

## Quick Start

### Start a Complete Cluster

```bash
gllm start-cluster \
  --port 5333 \
  --workers 0:3 \
  --worker-port-range-start 8000 \
  --load-model "Qwen/Qwen2.5-7B"
```

This command starts:
- A load balancer on port 5333
- 4 workers using GPUs 0, 1, 2, 3
- Workers on ports 8000, 8100, 8200, 8300
- vLLM backends on ports 8010, 8110, 8210, 8310
- Automatically loads the Qwen model on all workers

### Python Client Usage

```python
import gllm

# Connect to cluster
client = gllm.GLLM("http://localhost:5333")

# Load model and wait for readiness
client.load_model("Qwen/Qwen2.5-7B")
client.wait_for_health()

# Get completions
response = client.get_completions(
    model="Qwen/Qwen2.5-7B",
    prompt="Tell me a joke about programming",
    max_tokens=100,
    temperature=0.7,
    return_mode="primitives"
)
print(response[0])
```

## CLI Commands Reference

### `gllm start-cluster`

Start a complete distributed cluster with workers and load balancer.

**Required Arguments**:
- `--port` (int): Port for the load balancer to listen on
- `--workers`, `-w` (str): GPU configuration for workers
- `--worker-port-range-start`, `-wp` (int): Starting port for workers

**Optional Arguments**:
- `--ip` (str, default: "0.0.0.0"): IP address for the balancer
- `--load-model` (str): Model path/name to load after startup
- `--version`: Show version and exit

**Worker Configuration Formats**:
- Single GPU: `--workers 3` → Uses GPU 3
- Range: `--workers 0:3` → Uses GPUs 0, 1, 2, 3
- List: `--workers 0,2,5` → Uses GPUs 0, 2, 5
- Multiple ranges: `--workers 0:1,4:5` → Uses GPUs 0, 1, 4, 5

**Port Assignment**:
- Worker i uses port: `worker-port-range-start + i*100`
- vLLM backend uses port: `worker_port + 10`
- Example: If `--worker-port-range-start 8000`, then:
  - Worker 0: port 8000, vLLM: 8010
  - Worker 1: port 8100, vLLM: 8110
  - Worker 2: port 8200, vLLM: 8210

**Examples**:

Basic cluster with 4 GPUs:
```bash
gllm start-cluster --port 5333 --workers 0:3 --worker-port-range-start 8000
```

Cluster with specific GPUs and auto-load model:
```bash
gllm start-cluster \
  --port 5333 \
  --workers 0,2,4,6 \
  --worker-port-range-start 8000 \
  --load-model "Qwen/Qwen2.5-7B-Instruct"
```

Single worker for testing:
```bash
gllm start-cluster --port 5333 --workers 0 --worker-port-range-start 8000
```

### `gllm start-balancer`

Start only the load balancer (for use with existing workers).

**Required Arguments**:
- `--port`, `-p` (int): Port for the balancer to listen on

**Optional Arguments**:
- `--ip` (str, default: "0.0.0.0"): IP address to bind
- `--existing-workers`, `-ew` (list): Worker URLs to register
- `--load-model` (str): Model to load on workers (requires --existing-workers)

**Examples**:

Balancer with existing workers:
```bash
gllm start-balancer \
  --port 5333 \
  --existing-workers http://localhost:8000 http://localhost:8100 \
  --load-model "Qwen/Qwen2.5-7B"
```

Empty balancer (register workers later via API):
```bash
gllm start-balancer --port 5333
```

## Python Client API

### Initialization

```python
from gllm import GLLM

# Basic initialization
client = GLLM("http://localhost:5333")

# With API key (if authentication enabled)
client = GLLM("http://localhost:5333", api_key="your-api-key")
```

### Available Methods

#### `load_model(model_identifier, force_reload=False, timeout=10)`

Load a model onto all workers.

**Parameters**:
- `model_identifier` (str): HuggingFace model name or local path
- `force_reload` (bool): Force reload if model already loaded
- `timeout` (int): Request timeout in seconds

**Example**:
```python
client.load_model("Qwen/Qwen2.5-7B-Instruct")
```

#### `get_completions(...)`

Get text completions (OpenAI `/completions` compatible).

**Parameters**:
- `model` (str): Model identifier
- `prompt` (str): Input prompt
- `max_tokens` (int, default: 16): Maximum tokens to generate
- `temperature` (float, default: 1.0): Sampling temperature
- `n` (int, default: 1): Number of completions to generate
- `top_p` (float, default: 1.0): Nucleus sampling parameter
- `frequency_penalty` (float, default: 0.0): Frequency penalty
- `presence_penalty` (float, default: 0.0): Presence penalty
- `stop` (list[str] | str, optional): Stop sequences
- `echo` (bool, default: False): Echo the prompt
- `best_of` (int, optional): Generate best_of completions and return best
- `logprobs` (int, optional): Include log probabilities
- `logit_bias` (dict, default: {}): Token biases
- `seed` (int, optional): Random seed
- `return_mode` (str, default: "openai"): "openai" | "primitives" | "raw"

**Returns**:
- If `return_mode="primitives"`: List of completion strings
- If `return_mode="openai"`: `CompletionResponse` object
- If `return_mode="raw"`: Raw response dict

**Example**:
```python
# Get 5 completions as strings
results = client.get_completions(
    model="Qwen/Qwen2.5-7B",
    prompt="Once upon a time",
    max_tokens=50,
    temperature=0.8,
    n=5,
    return_mode="primitives"
)
# results = ["completion1", "completion2", ...]
```

#### `get_chat_completion(...)`

Get chat completions (OpenAI `/chat/completions` compatible).

**Parameters**:
- `model` (str): Model identifier
- `messages` (list[dict]): Chat messages with "role" and "content"
- `max_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature
- `n` (int, default: 1): Number of completions
- `stop` (list[str] | str, optional): Stop sequences
- `return_mode` (str, default: "openai"): "openai" | "primitives"

**Returns**:
- If `return_mode="primitives"`: List of message dicts
- If `return_mode="openai"`: `ChatGenerationResponse` object

**Example**:
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the capital of France?"}
]

response = client.get_chat_completion(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=messages,
    max_tokens=100,
    temperature=0.7,
    return_mode="primitives"
)
```

#### `is_healthy(timeout=5) -> bool`

Check if the cluster is healthy.

**Returns**: `True` if healthy, `False` otherwise.

#### `wait_for_health(timeout=300, check_interval=5)`

Block until cluster is healthy or timeout.

**Parameters**:
- `timeout` (int): Maximum wait time in seconds
- `check_interval` (int): Seconds between health checks

**Raises**: `TimeoutError` if not healthy within timeout.

#### `wait_for_live(timeout=300, check_interval=5)`

Block until cluster responds (any status code).

**Parameters**:
- `timeout` (int): Maximum wait time in seconds
- `check_interval` (int): Seconds between checks

**Raises**: `TimeoutError` if no response within timeout.

#### `release_gpus(timeout=20)`

Release GPU memory on all workers.

**Raises**: `RemoteError` if request fails.

### Complete Examples

#### Example 1: Basic Completions

```python
import gllm

client = gllm.GLLM("http://localhost:5333")
client.load_model("Qwen/Qwen2.5-7B")
client.wait_for_health()

# Single completion
result = client.get_completions(
    model="Qwen/Qwen2.5-7B",
    prompt="The meaning of life is",
    max_tokens=50,
    temperature=0.7,
    n=1,
    return_mode="primitives"
)
print(result[0])
```

#### Example 2: Parallel Requests

```python
import gllm
from concurrent.futures import ThreadPoolExecutor

client = gllm.GLLM("http://localhost:5333")
client.load_model("Qwen/Qwen2.5-7B-Instruct")
client.wait_for_health()

def get_completion(prompt):
    return client.get_completions(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        return_mode="primitives"
    )[0]

prompts = [f"Tell me a joke about {topic}" for topic in ["cats", "dogs", "birds", "fish"]]

# Submit 4 parallel requests
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(get_completion, p) for p in prompts]
    results = [future.result() for future in futures]

for prompt, result in zip(prompts, results):
    print(f"\n{prompt}\n{result}\n{'-'*50}")
```

#### Example 3: Chat Conversation

```python
import gllm

client = gllm.GLLM("http://localhost:5333")
client.load_model("Qwen/Qwen2.5-7B-Instruct")
client.wait_for_health()

messages = [
    {"role": "system", "content": "You are a helpful coding assistant"},
    {"role": "user", "content": "How do I reverse a list in Python?"}
]

response = client.get_chat_completion(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=messages,
    max_tokens=200,
    temperature=0.5,
    return_mode="openai"
)

print(response.choices[0].message.content)
```

## OpenAI API Compatibility

GLLM exposes OpenAI-compatible endpoints, so you can use the official OpenAI Python client:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:5333",
    api_key="dummy"  # API key required by client but not validated
)

# Chat completion
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Write a haiku about recursion"}
    ],
    max_tokens=100,
    temperature=0.8
)
print(response.choices[0].message.content)

# Text completion
response = client.completions.create(
    model="Qwen/Qwen2.5-7B",
    prompt="The quick brown fox",
    max_tokens=50
)
print(response.choices[0].text)
```

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│            Load Balancer (Flask Server)                 │
│  - Receives inference requests                          │
│  - Routes to least-busy worker                          │
│  - Tracks queue size per worker                         │
│  - Performs health checks                               │
│  - Manages worker registration                          │
└─────────────────────────────────────────────────────────┘
                    ↓           ↓           ↓
    ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
    │   Worker 1        │ │   Worker 2        │ │   Worker 3        │
    │  (Flask Server)   │ │  (Flask Server)   │ │  (Flask Server)   │
    │  Port: 8000       │ │  Port: 8100       │ │  Port: 8200       │
    └───────────────────┘ └───────────────────┘ └───────────────────┘
            ↓                       ↓                       ↓
    ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
    │  vLLM Backend     │ │  vLLM Backend     │ │  vLLM Backend     │
    │  Port: 8010       │ │  Port: 8110       │ │  Port: 8210       │
    │  GPUs: 0-3        │ │  GPUs: 4-7        │ │  GPUs: 8-11       │
    └───────────────────┘ └───────────────────┘ └───────────────────┘
```

### Load Balancing Algorithm

1. **Request arrives** at load balancer
2. **Sample workers**: Randomly select min(n_workers, CHECK_SIZE=5) workers
3. **Select least-loaded**: Pick worker with smallest queue size
4. **Forward request**: Send to selected worker
5. **Track queue**: Increment queue counter (decremented on response)
6. **Automatic retry**: If request fails, exponential backoff retry

### Key Features

- **Thread-safe queue tracking**: Uses locks to safely track concurrent requests
- **Health monitoring**: Regular health checks to each worker
- **Graceful shutdown**: Properly terminates workers and balancer on SIGINT/SIGTERM
- **Automatic retries**: Client library has built-in exponential backoff
- **OpenAI compatibility**: Drop-in replacement for OpenAI API

### Request Flow

```
Client → Balancer → Worker → vLLM → Worker → Balancer → Client
           ↓                    ↑
    Queue tracking      Actual inference
```

## API Endpoints Reference

### OpenAI-Compatible Endpoints

#### `POST /chat/completions`

OpenAI-compatible chat completion endpoint.

**Request Body**: `ChatGenerationRequest` (Pydantic model)
```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "n": 1,
  "stop": null
}
```

**Response**: `ChatGenerationResponse` with choices array

#### `POST /completions`

OpenAI-compatible text completion endpoint.

**Request Body**: `CompletionRequest` (Pydantic model)
```json
{
  "model": "Qwen/Qwen2.5-7B",
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 1.0,
  "n": 1,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": null,
  "echo": false,
  "best_of": null,
  "logprobs": null,
  "logit_bias": {},
  "seed": null
}
```

**Response**: `CompletionResponse` with choices array

### vLLM/Health Endpoints

#### `GET /health`

Health check endpoint. Returns 200 if cluster is healthy.

**Response**: 200 OK or error status

### GLLM Custom Endpoints

#### `POST /add_worker`

Register a new worker with the balancer.

**Request Body**: `AddWorkerRequest`
```json
{
  "worker_url": "http://localhost:8300"
}
```

**Response**: Success message

#### `POST /load_model`

Load a model on all registered workers.

**Request Body**: `LoadModelRequest`
```json
{
  "model_path": "Qwen/Qwen2.5-7B-Instruct",
  "force_reload": false
}
```

**Response**: Success message or error

#### `POST /release_gpus`

Release GPU memory on all workers.

**Request Body**: None

**Response**: Success message or error

## Development TODOs

3. Make decorators with `backoff` dynamic, allow the backoff parameters to passed as a parameter.that redecorates the function with backoff on every call. This allows us to change backoff time. Do this by making the decorator return a callable object.
4. Make errors and error messages from worker propagate up from vllm all the way up through the balancer and to the client-side.
5. Currently return codes are a little chaotic (i.e. if failed: return random.randrange(500,599)). Let's make them a little more consistent.
6. Make worker try next port if vllm throws throws a socket in use.
7. Allow the use of extra_body parameters
