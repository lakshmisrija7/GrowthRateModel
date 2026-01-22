class TechnicalAnalysisError(Exception):
    pass

class ConnectionError(TechnicalAnalysisError):
    pass

class RequestError(TechnicalAnalysisError):
    pass

class ResponseError(TechnicalAnalysisError):
    pass
