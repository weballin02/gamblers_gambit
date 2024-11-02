
from collections import defaultdict

class SignalSystem:
    def __init__(self):
        self.listeners = defaultdict(list)

    def emit(self, signal_name, data=None):
        for listener in self.listeners[signal_name]:
            listener(data)

    def listen(self, signal_name, listener):
        self.listeners[signal_name].append(listener)

signal_system = SignalSystem()
