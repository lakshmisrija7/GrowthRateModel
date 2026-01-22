class SectorFetcherError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class SectorConnectionError(SectorFetcherError):
    pass

class SectorPayloadError(SectorFetcherError):
    pass

class SectorResponseError(SectorFetcherError):
    pass
