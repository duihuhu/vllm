import requests
from typing import Tuple, Dict
from enum import Enum
import aiohttp
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

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

    @staticmethod
    async def async_send_to(entry_point: Tuple[str, int], func_name: str, data: CommData):
        api_url = f"http//{entry_point[0]}:{entry_point[1]}/{func_name}"
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=api_url, json=data.payload,
                                    headers=data.headers) as response:
                return await response.text()
class CacheMeta:
    def __init__(
        self,
        cmeta_host: str = None,
        cmeta_port: str = None,
        cmeta_ranks: list = [],
        cmeta_kv_len: int = 0,
    ) -> None:
        self.cmeta_host = cmeta_host
        self.cmeta_port = cmeta_port
        self.cmeta_ranks = cmeta_ranks
        self.cmeta_kv_len = cmeta_kv_len
        self.cached_len = 0 
        self.is_ready = False
        
    def __json__(self):
        return {
            "cmeta_host": self.cmeta_host,
            "cmeta_port": self.cmeta_port,
            "cmeta_ranks": self.cmeta_ranks,
            "cmeta_kv_len": self.cmeta_kv_len,
            "cached_len": self.cached_len,
        }
        
class QueryCacheMeta():
    def __init__(
        self,
        request_id: str =None,
        prompt_token_ids: list = [],
    ) -> None:
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
    def __json__(self):
        return {
            "request_id": self.request_id,
            "prompt_token_ids": self.prompt_token_ids
        }


class QueryMeta:
    def __init__(
        self,
        cache_meta: CacheMeta,
        opp_host: str = None,
        opp_port: int = 0,
        opp_ranks: list = [],
        request_id: str =None,
        prompt_token_ids: list = [],
    ) -> None:
        self.cache_meta = cache_meta
        self.opp_host = opp_host
        self.opp_port = opp_port
        self.opp_ranks = opp_ranks
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
              
    def __json__(self):
        return {
            "cache_meta": self.cache_meta.__json__(),
            "opp_host": self.opp_host,
            "opp_port": self.opp_port,
            "opp_ranks": self.opp_ranks,
            "request_id": self.request_id,
            "prompt_token_ids": self.prompt_token_ids
        }
