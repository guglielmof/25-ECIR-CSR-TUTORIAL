class AbstractBiencoder:

    def encode_queries(self, text):
        raise NotImplementedError

    def encode_documents(self, text):
        raise NotImplementedError