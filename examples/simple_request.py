import gllm
import argparse
import openai

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple LLM request')
    parser.add_argument('--address', default="http://localhost:5333", help='Server address')
    parser.add_argument('--model', default="Qwen/Qwen2-7B-Instruct", help='Model name')
    parser.add_argument('--no-load', action='store_false', dest='load_model', help='Do not load model')
    args = parser.parse_args()

    address = args.address
    model = args.model 
    load_model = args.load_model

    query = [
        {"role": "system", "content": "You are an AI that tells jokes"},
        {"role": "user", "content": "Tell me a joke about a cat."},
    ]

    # glm_handle = gllm.GLLM(address)
    # glm_handle = openai.OpenAI(base_url=address)
    glm_handle = openai.OpenAI(
        base_url=address,
        api_key="dummy"  # API key is required but not used
    )

    response = glm_handle.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "tell me a joke"}],
        max_tokens=100,
        temperature=0.5
    )
    print("OpenAI API Response:", response.choices[0].message.content)
    print(dir(glm_handle))
    if load_model:
        glm_handle.load_model(model)

        glm_handle.wait_for_health()
    results = glm_handle.get_chat_completions(
        model, query, max_tokens=100, temperature=0.5, return_mode="primitives"
    )

    print(results)
    completions = glm_handle.get_completions(
        model, "Tell me a joke about a cat.", max_tokens=100, temperature=0.5, n=5,
        return_mode="primitives"
    )
    print(completions)

