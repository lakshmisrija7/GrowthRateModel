class VARModelError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class VARConnectionError(VARModelError):
    pass

class VARPayloadError(VARModelError):
    pass

class VARResponseError(VARModelError):
    pass
