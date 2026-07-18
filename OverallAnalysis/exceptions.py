class OverallAnalysisError(Exception):
    pass

class ConnectionError(OverallAnalysisError):
    pass

class RequestError(OverallAnalysisError):
    pass

class ResponseError(OverallAnalysisError):
    pass
