
class State:
    _state = {}

    @classmethod
    def get(cls, key, default=None):
        return cls._state.get(key, default)

    @classmethod
    def set(cls, key, value):
        cls._state[key] = value
