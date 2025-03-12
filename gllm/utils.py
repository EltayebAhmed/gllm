import json
import flask

def get_request_params(request: flask.request) -> dict:
    return json.loads(request.get_json())
