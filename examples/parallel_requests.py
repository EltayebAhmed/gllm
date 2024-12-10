import dataclasses
import json
import os
import tqdm
import gllm
import tyro
from multiprocessing import dummy as mp
import functools




SYSTEM_PROMPT = "You are an AI assistant that responds with jokes about the animal I mention. I will show you some examples first before I give you an animal to make a joke about. Please respond with only the joke."
USER_PROMPT = """Examples
Example 1: cat
Response 1: Why was the cat sitting on the computer? Because it wanted to keep an eye on the mouse!

Example 2: dog
Response 2: Why did the dog sit in the shade? Because he didn't want to be a hot dog!

Ok now it's your turn. Please respond with a joke about the animal {animal}.
"""

def get_a_joke(animal, model_name):
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': USER_PROMPT.format(animal=animal)}]
        result = model.get_chat_completion(model_name, messages, 100, 0.5, return_mode="primitives")
        return result[0]['content']



@dataclasses.dataclass
class Config:
    model_name : str ='Qwen/Qwen2.5-7B-Instruct'
    server_address : str = "http://127.0.0.1:5333"
    load_model: bool = True
    n_threads: int = 50

if __name__ == '__main__':
    cfg = tyro.cli(Config)
    model = gllm.DistributionServerInterface(cfg.server_address)

    if cfg.load_model:
        model.load_model(cfg.model_name)

    model.wait_for_health()

    system_prompt = "You are a chatbot that responds to any message containing one word with a joke containing that word."

    animals = [
        "cat", "dog", "bird", "fish", "elephant", "giraffe", "lion", "tiger", "bear", "wolf",
        "fox", "rabbit", "deer", "moose", "horse", "cow", "pig", "sheep", "goat", "chicken",
        "duck", "turkey", "parrot", "penguin", "owl", "eagle", "hawk", "falcon", "raven", "crow",
    ]

    results = []
    f = functools.partial(get_a_joke, model_name=cfg.model_name)
    with mp.Pool(cfg.n_threads) as pool:
        for i, result in enumerate(tqdm.tqdm(pool.imap_unordered(f, animals))):
            results.append(result)

    print(results)
