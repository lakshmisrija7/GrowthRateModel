import asyncio
import json
import websockets
from concurrent.futures import ThreadPoolExecutor
from .config import URL, HEADERS
from .logger import SectorLogger
from .exceptions import SectorFetcherError, SectorConnectionError, SectorResponseError
from .request import SectorAnalysisRequest
from .response import SectorAnalysisResponse

class SectorAnalysisFetcher:
    def __init__(self, url: str = URL, headers: dict = HEADERS, logger: SectorLogger = None):
        self._url = url
        self._headers = headers
        self._logger = logger or SectorLogger()

    @property
    def url(self) -> str:
        return self._url

    async def fetch_async(self, company_name: str) -> SectorAnalysisResponse:
        request = SectorAnalysisRequest(company_name=company_name)
        payload = request.to_dict()
        self._logger.info(f"Connecting to {self._url} for sector analysis of {company_name}")
        try:
            async with websockets.connect(self._url, extra_headers=self._headers) as websocket:
                self._logger.info("Sending request payload")
                await websocket.send(json.dumps(payload))
                response = await websocket.recv()
                self._logger.info("Received response successfully")
                return SectorAnalysisResponse.from_dict(json.loads(response))
        except websockets.exceptions.WebSocketException as e:
            self._logger.error(f"WebSocket connection error: {str(e)}")
            raise SectorConnectionError(f"WebSocket connection error: {str(e)}") from e
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to decode response JSON: {str(e)}")
            raise SectorResponseError(f"Failed to decode response JSON: {str(e)}") from e
        except Exception as e:
            self._logger.error(f"Unexpected error: {str(e)}")
            raise SectorFetcherError(f"Unexpected error occurred: {str(e)}") from e

    def fetch(self, company_name: str) -> SectorAnalysisResponse:
        try:
            return asyncio.run(self.fetch_async(company_name))
        except RuntimeError:
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.fetch_async(company_name))
                finally:
                    new_loop.close()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()
