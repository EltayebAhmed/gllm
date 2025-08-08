import enum

class Endpoints(str, enum.Enum):
    # OpenAI endpoints.
    CHAT_COMPLETIONS = "/chat/completions"
    COMPLETIONS = "/completions"

    # VLLM OpenAI Server related endpoints.
    HEALTH = "/health"

    # GLLM custom endpoints.
    RELEASE_GPUS = "/release_gpus"

    ADD_WORKER = "/add_worker"
    LOAD_MODEL = "/load_model"


class SpecialFields(str, enum.Enum):
    CONVERSATION_ID = "conversation_id"