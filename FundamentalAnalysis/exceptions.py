class FundamentalAnalysisError(Exception):
    pass

class ConnectionError(FundamentalAnalysisError):
    pass

class RequestError(FundamentalAnalysisError):
    pass

class ResponseError(FundamentalAnalysisError):
    pass
