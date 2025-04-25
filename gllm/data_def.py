from pydantic import BaseModel
from typing import Optional

class AddWorkerRequest(BaseModel):
    address: str


class Message(BaseModel):
    role: str
    content: str
    refusal: Optional[str] = None


class ChatGenerationRequest(BaseModel):
    messages: list[Message]
    model: str
    max_tokens: int
    temperature: float = 1.0
    n: int = 1
    stop: Optional[list[str]| str] = None


class LoadModelRequest(BaseModel):
    model_path: str
    force_reload: bool = False


class ChatGenerationResponse(BaseModel):
    choices: list[Message]

class CompletionRequest(BaseModel):
    model : str
    prompt: str
    best_of: Optional[int] = None
    echo: bool = False
    frequency_penalty: float = 0.0
    logit_bias: dict = {}
    logprobs: Optional[int] = None
    max_tokens: int = 16
    n : int = 1
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop: Optional[list[str]| str] = None
    temperature: float = 1.0
    top_p: float = 1.0


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: None = None # raise NotImplementedError :)
    finish_reason: str

class CompletionUsage(BaseModel):
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    completion_tokens_details: Optional[dict[str, int]] = None
    prompt_tokens_details: Optional[dict[str, int]] = None
    
class CompletionResponse(BaseModel):
    id : str
    choices: list[CompletionChoice]
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    object: str
    usage: CompletionUsage
