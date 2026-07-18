class ARIMAModelError(Exception):
    pass

class ConnectionError(ARIMAModelError):
    pass

class RequestError(ARIMAModelError):
    pass

class ResponseError(ARIMAModelError):
    pass

class ValuationError(ARIMAModelError):
    pass
