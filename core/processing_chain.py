
class ProcessingChain:
    def __init__(self):
        self.chain = []

    def add_to_chain(self, handler):
        self.chain.append(handler)

    def process(self, data):
        for handler in self.chain:
            data = handler(data)
        return data

processing_chain = ProcessingChain()
