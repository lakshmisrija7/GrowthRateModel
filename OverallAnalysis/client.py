import json
import websockets
from .config import WS_URL, API_KEY
from .logger import get_logger
from .exceptions import ConnectionError, RequestError, ResponseError
from .request import OverallAnalysisRequest
from .response import OverallAnalysisResponse

logger = get_logger(__name__)


class OverallAnalysisClient:
    """
    WebSocket client for fetching overall analysis data.
    Follows the same connect / fetch / disconnect lifecycle as all other analysis clients.
    """

    def __init__(self, url: str = WS_URL, api_key: str = API_KEY):
        self.url = url
        self.api_key = api_key
        self.connection = None

    async def connect(self):
        try:
            logger.info(f"Connecting to WebSocket URL: {self.url}")
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
                headers["X_API_KEY"] = self.api_key
            self.connection = await websockets.connect(
                self.url,
                additional_headers=headers,
                max_size=None
            )
            logger.info("Successfully connected to WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to {self.url}: {e}")
            raise ConnectionError(f"Connection failed: {e}") from e

    async def disconnect(self):
        if self.connection:
            try:
                logger.info("Closing WebSocket connection")
                await self.connection.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error while closing connection: {e}")
            finally:
                self.connection = None

    async def fetch_overall_analysis(
        self,
        symbol: str,
        from_date: str,
        to_date: str
    ) -> OverallAnalysisResponse:
        """
        Send GET_OVERALL_ANALYSIS request and return the parsed response.

        Args:
            symbol:    Ticker symbol (e.g. "LLY")
            from_date: Start timestamp in milliseconds as a string
            to_date:   End timestamp in milliseconds as a string

        Returns:
            OverallAnalysisResponse instance
        """
        if not self.connection:
            raise ConnectionError("Not connected to WebSocket. Call connect() first.")

        request = OverallAnalysisRequest(symbol, from_date, to_date)
        request_data = json.dumps(request.to_dict())

        try:
            logger.info(f"Sending overall analysis request for symbol {symbol}")
            await self.connection.send(request_data)
            logger.info("Request sent successfully")
        except Exception as e:
            logger.error(f"Failed to send request: {e}")
            raise RequestError(f"Failed to send request: {e}") from e

        try:
            logger.info("Waiting for response")
            raw_response = await self.connection.recv()
            logger.info("Response received")
            return OverallAnalysisResponse(raw_response)
        except Exception as e:
            logger.error(f"Failed to receive response: {e}")
            raise ResponseError(f"Failed to receive response: {e}") from e
