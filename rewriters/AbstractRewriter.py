class AbstractRewriter:

    def __init__(self, **kwargs):
        self.history = None

    def reset_history(self):
        self.history = None

    def _add_to_history(self, previous, **kwargs):
        raise NotImplementedError()

    def rewrite(self, query, **kwargs):
        raise NotImplementedError()