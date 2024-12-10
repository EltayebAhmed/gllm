import json


def get_request_params(request):
    return json.loads(request.get_json())
