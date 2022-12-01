from publish.utils.cache import BaseCache
from typing import Callable

class BasicDictCache(BaseCache):
    def __init__(
        self,
        key_function: Callable,
        include_key_in_fxn_call: bool = False
    ):
        self._cache = {}
        self._key_function = key_function
        self._include_key_in_fxn_call = include_key_in_fxn_call

    def get(self, key, *args, **kwargs):
        if key not in self._cache:
            if self._include_key_in_fxn_call:
                kwargs['cache_key'] = key

            self._cache[key] = self._key_function(*args, **kwargs)

        return self._cache[key]
