class ContextError(Exception):
    def __init__(self, message: str = "ContextError"):
        self.message = message
        super().__init__(self.message)