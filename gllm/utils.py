import flask

def get_request_params(request: flask.request) -> dict:
    return request.get_json()
