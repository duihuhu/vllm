import requests
from typing import Tuple, Dict, Optional
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
        api_url = f"http://{entry_point[0]}:{entry_point[1]}/{func_name}"
        response = requests.post(api_url, headers=data.headers, json=data.payload, stream=True)
        return response
    
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
        
    def __json__(self):
        return {
            "cmeta_host": self.cmeta_host,
            "cmeta_port": self.cmeta_port,
            "cmeta_ranks": self.cmeta_ranks,
            "cmeta_kv_len": self.cmeta_kv_len,
            "cached_len": self.cached_len,
        }
        
    
class QueryMeta:
    def __init__(
        self,
        cache_meta: CacheMeta,
        local_host: str = None,
        local_port: int = 0,
        local_ranks: list = [],
        request_id: str =None,
        prompt_token_ids: list = [],
    ) -> None:
        self.cache_meta = cache_meta
        self.local_host = local_host
        self.local_port = local_port
        self.local_ranks = local_ranks
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
              
    def __json__(self):
        return {
            "cache_meta": self.cache_meta.__json__(),
            "local_host": self.local_host,
            "local_port": self.local_port,
            "local_ranks": self.local_ranks,
            "local_ranks": self.local_ranks,
            "request_id": self.request_id,
            "prompt_token_ids": self.prompt_token_ids
        }
