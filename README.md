Instructions for Installation
####

1. On the base machine, from the root directory of the repo run
```>>> docker compose -p <your_name> build
```
2. If running your user scripts in docker be sure to use `--network=host`
3. In the environment you will be running your scripts run
```>>> pip install -r requirements.txt
```
4. When running your script ensure that the root folder of the repo is in your `PYTHONPATH`
5. Spin up gllm servers by running this on the root machine
``` >>> docker compose -p <your_name> up
```
6. To modify GPU assignment or number of GPUs add or remove workers to compose.yml


# TODOS
## Before release
1. Make work with OpenAI client
2. Add cli tool
## Post release

3. Make cursed decorator that redecorates the function with backoff on every call. This allows us to change backoff time. Do this by making the decorator return a callable object. 
4. Make errors and error messages propagate up from vllm all the way up to balancer 
5. Currently return codes are a little cursed (i.e. if failed: return random.randrange(500,599)). Let's make them a little more consistent.
6. Make worker try next port if vllm throws throws a socket in use. 
7. Allow the use of extra_body parameters
