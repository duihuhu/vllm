import requests
from typing import Tuple, Dict
from enum import Enum

class EngineType(Enum):
    EPREFILL = "ep"
    EDECODE = "ed"
    EPD = "epd"

class CommonHeader:
    def __init__(
        self,
        host: str,
        port: int,
        engine_type: EngineType=None
    ) -> None:
        self.host = host
        self.port = port
        self.engine_type = engine_type
    
    def __json__(self) -> Dict:
        if self.engine_type:
            return {
                "host": self.host,
                "port": str(self.port),
                "engine_type": self.engine_type.name
            }
        else:
            return {
                "host": self.host,
                "port": str(self.port)
            }

class CommData:
    def __init__(self, headers, payload) -> None:
        self.headers = headers
        self.payload = payload
        
class CommEngine:
    @staticmethod
    def send_to(entry_point: Tuple[str, int], func_name: str, data: CommData):
        api_url = f"http//{entry_point[0]}:{entry_point[1]}/{func_name}"
        response = requests.post(api_url, headers=data.headers, json=data.payload, stream=True)
        return response