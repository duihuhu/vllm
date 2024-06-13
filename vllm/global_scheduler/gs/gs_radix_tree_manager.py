"""index manager in xds_rtc."""
from typing import Dict, List, Tuple

from vllm.global_scheduler.gs.gs_radix_cache import RadixCache 

MAX_BLOCKS_ALLOWED_PER_SEQ = 65535


class RadixTreeManager:
    def __init__(
        self,
        block_size: int
    ) -> None:
        """IndexManager.

        Args: None
        """
        self.tree_cache = RadixCache(block_size)
        self.block_size = block_size
        self.max_blocks_allowed = MAX_BLOCKS_ALLOWED_PER_SEQ  # 性能测试用

    ##### Public API #####
    def match(self, token_ids: List[int]) -> bool:
        """optimizing the seq with radix tree inference
        """
        nodes = self._match_prefix(token_ids)
        nodes = nodes[:self.max_blocks_allowed]

        return nodes

    def insert(self, token_ids: List[int], instance: str) -> bool:
        # 添加lora id作为前缀
        # 推理完成后，保存token和block到radix tree
        self._insert(token_ids, instance)
        return True

    # 查找匹配点位置，并返回匹配点的value和Node
    def _match_prefix(self, key: List[int]):
        splited_key: List[Tuple] = []
        height = 0
        while len(key) >= self.block_size:
            # lora id == 0说明不使用lora
            # 在第0层添加lora id，分开存储不同lora的cache
            if height == 0:
                splited_key.append(tuple(key[:self.block_size]))
                height += 1
            else:
                splited_key.append(tuple(key[:self.block_size]))
            key = key[self.block_size:]
        if len(splited_key) == 0:
            return []
        return self.tree_cache.match_prefix(splited_key)

    # 根据原始key插入block到radix tree， prompt_token_ids按block切分，与value中的block一一映射
    # 返回插入到radix tree的路径段数
    def _insert(self, key: List[int], instance: str):
        splited_key: List[Tuple] = []
        height = 0
        while len(key) >= self.block_size:
            # lora id == 0说明不使用lora
            # 在第0层添加lora id，分开存储不同lora的cache
            if height == 0:
                splited_key.append(tuple(key[:self.block_size]))
                height += 1
            else:
                splited_key.append(tuple(key[:self.block_size]))
            key = key[self.block_size:]
        if len(splited_key) == 0:
            return 0
        return self.tree_cache.insert(splited_key, instance)
