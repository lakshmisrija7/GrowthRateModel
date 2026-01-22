class SentimentFetcherError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class SentimentConnectionError(SentimentFetcherError):
    pass

class SentimentPayloadError(SentimentFetcherError):
    pass

class SentimentResponseError(SentimentFetcherError):
    pass
