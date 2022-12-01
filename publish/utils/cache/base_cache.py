from abc import ABC, abstractmethod

class BaseCache(ABC):
    @abstractmethod
    def get(self, key, *args, **kwargs):
        pass
