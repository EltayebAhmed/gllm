import enum

class Endpoints(str, enum.Enum):
    # OpenAI endpoints
    CHAT_COMPLETIONS = "/v1/chat/completions"
    COMPLETIONS = "/v1/completions"

    # VLLM OpenAI related endpoints
    HEALTH = "/health"

    # GLLM custom endpoints
    RELEASE_GPUS = "/release_gpus"

    ADD_WORKER = "/add_worker"
    LOAD_MODEL = "/load_model"

