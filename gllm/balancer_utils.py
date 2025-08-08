from typing import Optional
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = max(int(capacity), 1)
        self._data: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
