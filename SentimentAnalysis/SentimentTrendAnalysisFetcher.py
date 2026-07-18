import asyncio
import json
import websockets
from concurrent.futures import ThreadPoolExecutor
from .config import URL, HEADERS
from .logger import SentimentLogger
from .exceptions import SentimentFetcherError, SentimentConnectionError, SentimentResponseError
from .request import SentimentTrendRequest
from .response import SentimentTrendResponse

class SentimentTrendAnalysisFetcher:
    def __init__(self, url: str = URL, headers: dict = HEADERS, logger: SentimentLogger = None):
        self._url = url
        self._headers = headers
        self._logger = logger or SentimentLogger()

    @property
    def url(self) -> str:
        return self._url

    async def fetch_async(self, category: str, from_date: str, to_date: str) -> SentimentTrendResponse:
        request = SentimentTrendRequest(category=category, from_date=from_date, to_date=to_date)
        payload = request.to_dict()
        self._headers = self._headers
        self._logger.info(f"Connecting to {self._url} for sentiment trend analysis of {category}")
        try:
            async with websockets.connect(self._url, extra_headers=self._headers, max_size=None) as websocket:
                self._logger.info("Sending request payload")
                await websocket.send(json.dumps(payload))
                response = await websocket.recv()
                self._logger.info("Received response successfully")
                return SentimentTrendResponse.from_dict(json.loads(response))
        except websockets.exceptions.WebSocketException as e:
            self._logger.error(f"WebSocket connection error: {str(e)}")
            raise SentimentConnectionError(f"WebSocket connection error: {str(e)}") from e
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to decode response JSON: {str(e)}")
            raise SentimentResponseError(f"Failed to decode response JSON: {str(e)}") from e
        except Exception as e:
            self._logger.error(f"Unexpected error: {str(e)}")
            raise SentimentFetcherError(f"Unexpected error occurred: {str(e)}") from e

    def fetch(self, category: str, from_date: str, to_date: str) -> SentimentTrendResponse:
        try:
            return asyncio.run(self.fetch_async(category, from_date, to_date))
        except RuntimeError:
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.fetch_async(category, from_date, to_date))
                finally:
                    new_loop.close()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()
