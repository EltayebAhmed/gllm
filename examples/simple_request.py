import gllm


address = "http://localhost:5333"
model = "Qwen/Qwen2-7B-Instruct"
load_model = True

query = [
    {"role": "system", "content": "You are an AI that tells jokes"},
    {"role": "user", "content": "Tell me a joke about a cat."},
]

glm_handle = gllm.DistributionServerInterface(address)

if load_model:
    glm_handle.load_model(model)

glm_handle.wait_for_health()
results = glm_handle.get_chat_completion(
    model, query, max_tokens=100, temperature=0.5, return_mode="primitives"
)

print(results)
