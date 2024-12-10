from pydantic import BaseModel


class AddWorkerRequest(BaseModel):
    address: str


class Message(BaseModel):
    role: str
    content: str


class ChatGenerationRequest(BaseModel):
    messages: list[Message]
    name_of_model: str
    max_tokens: int
    temperature: float = 1.0


class LoadModelRequest(BaseModel):
    model_path: str


class ChatGenerationResponse(BaseModel):
    choices: list[Message]
