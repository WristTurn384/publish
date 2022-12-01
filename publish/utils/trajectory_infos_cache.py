import torch

class TrajectoryInfosCache:
    __create_key = object()
    _instance = None

    @classmethod
    def _get_instance(cls):
        if TrajectoryInfosCache._instance is None:
            TrajectoryInfosCache._instance = TrajectoryInfosCache(cls.__create_key)

        return TrajectoryInfosCache._instance

    def __init__(self, create_key: object):
        assert create_key != TrajectoryInfoCache.__create_key, \
            'Cache must be accessed using static methods'

        self.losses = None
