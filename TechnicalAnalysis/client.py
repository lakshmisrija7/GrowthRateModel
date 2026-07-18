import asyncio
import json
import websockets
from .logger import get_logger
from .exceptions import ConnectionError, RequestError, ResponseError
from .request import TechnicalRequest
from .response import TechnicalAnalysisResponse

logger = get_logger(__name__)

class TechnicalWebSocketClient:
    def __init__(self, url: str, api_key: str = None):
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
            self.connection = await websockets.connect(self.url, additional_headers=headers)
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

    async def fetch_technical_analysis(self, symbol: str, from_date: int, to_date: int) -> TechnicalAnalysisResponse:
        if not self.connection:
            raise ConnectionError("Not connected to WebSocket. Call connect() first.")
        
        request = TechnicalRequest(symbol, from_date, to_date)
        request_data = json.dumps(request.to_dict())
        
        try:
            logger.info(f"Sending technical analysis request for symbol {symbol}")
            await self.connection.send(request_data)
            logger.info("Request sent successfully")
        except Exception as e:
            logger.error(f"Failed to send request: {e}")
            raise RequestError(f"Failed to send request: {e}") from e

        try:
            logger.info("Waiting for response")
            raw_response = await self.connection.recv()
            logger.info("Response received")
            return TechnicalAnalysisResponse(raw_response)
        except Exception as e:
            logger.error(f"Failed to receive response: {e}")
            raise ResponseError(f"Failed to receive response: {e}") from e
