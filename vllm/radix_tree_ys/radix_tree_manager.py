"""index manager in xds_rtc."""
from typing import Dict, List, Optional, Union, Tuple
from vllm.block import PhysicalTokenBlock
from vllm.sequence import Sequence
from vllm.logger import init_logger
from vllm.radix_tree_ys.radix_cache import (RadixCache, TreeNode, kvCacheProgressStatus)


logger = init_logger(__name__)


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
    def match(self, seq: Sequence) -> bool:
        """optimizing the seq with radix tree inference
        """
        # 添加lora id作为前缀
        lora_id = seq.lora_request.lora_id if seq.lora_request else 0
        # 推理前，根据token匹配前缀
        nodes = self._match_prefix(seq.get_token_ids(), lora_id=lora_id)
        nodes = nodes[:self.max_blocks_allowed]
        seq.data.cache_token_len = len(nodes)*self.block_size
        # seq.data.set_num_computed_tokens(len(nodes)*self.block_size)
        seq.cache_nodes = nodes
        # seq.data.prefix_block_ids = [
        #     node.value.physicalTokenBlock.block_number for node in nodes]
        # seq.data.prefix_blocks = [
        #     node.value.physicalTokenBlock for node in nodes]
        for node in nodes:
            node.value.physicalTokenBlock.ref_count += 1
            seq.data.prefix_block_ids.append(node.value.physicalTokenBlock.block_number)
            seq.data.prefix_blocks.append(node.value.physicalTokenBlock)
        # 检查匹配到的block是不是正在swap，如果正在swap则匹配失败
        for node in nodes:
            if node.value.progressStatus != kvCacheProgressStatus.STABLE:
                return False
        logger.debug(
            f"matched: {len(nodes)} nodes. passing {min(len(nodes), self.max_blocks_allowed)} blocks")
        return True

    def insert(self, seq: Sequence, cpu_free_call_back) -> bool:
        # 添加lora id作为前缀
        lora_id = seq.lora_request.lora_id if seq.lora_request else 0
        # 推理完成后，保存token和block到radix tree
        self._insert(seq.data.prompt_token_ids+seq.data.output_token_ids[:-1],
                     seq.cache_blocks_to_insert, cpu_free_call_back, lora_id=lora_id)
        return True

    # 淘汰函数说明：
    # 淘汰目标对象：叶子节点按访问时间LRU排序，把最长时间不访问且引用计数为0的block，从cpu cache淘汰
    # 入参：num_tokens， 预期触发同步淘汰的数量
    #       evict_callback， 淘汰函数
    # 出参：
    # 1、num_evicted == num_tokens
    # 2、num_evicted < num_tokens,说明叶子节点正在被引用，不足以淘汰，需要swap。
    def evict(self, num_nodes, device, evict_callback) -> int:
        return self.tree_cache.evict(num_nodes, device, evict_callback)

    # swap函数说明：
    # swap目标对象：叶子节点按访问时间LRU排序，把最长时间不访问且引用计数为0的block，从gpu cache swap out到cpu cache
    # 入参：num_tokens， 预期触发异步swap的数量
    #       modify_callback， swap函数
    # 出参：
    # 1、num_swaped == num_tokens，成功触发异步swap
    # 2、num_swaped < num_tokens,说明叶子节点正在被引用，不足以swap，需要先减引用计数，再次swap。
    def swap_out(self, num_nodes: int) -> List[TreeNode]:
        return self.tree_cache.swap_out(num_nodes)

    def refresh_nodes_status(self, nodes: List[TreeNode]):
        if nodes:
            return self.tree_cache.refresh_node_status(nodes)
        return None

    def get_num_nodes(self, device=None) -> int:
        return self.tree_cache.get_num_nodes(device)

    def get_num_nodes_can_swap_out(self) -> int:
        return self.tree_cache.get_num_nodes_can_swap_out()

    def _set_max_blocks_allowed(self, value):
        self.max_blocks_allowed = value

    # 查找匹配点位置，并返回匹配点的value和Node
    def _match_prefix(self, key: List[int], lora_id: int = 0):
        splited_key: List[Tuple] = []
        height = 0
        while len(key) > self.block_size:
            # lora id == 0说明不使用lora
            # 在第0层添加lora id，分开存储不同lora的cache
            if height == 0:
                splited_key.append(tuple([lora_id] + key[:self.block_size]))
                height += 1
            else:
                splited_key.append(tuple(key[:self.block_size]))
            key = key[self.block_size:]
        if len(splited_key) == 0:
            return []
        return self.tree_cache.match_prefix(splited_key)

    # 根据原始key插入block到radix tree， prompt_token_ids按block切分，与value中的block一一映射
    # 返回插入到radix tree的路径段数
    def _insert(self, key: List[int], value: List[PhysicalTokenBlock], cpu_free_call_back, lora_id: int = 0):
        splited_key: List[Tuple] = []
        height = 0
        while len(key) >= self.block_size:
            if height == 0:
                splited_key.append(tuple([lora_id] + key[:self.block_size]))
                height += 1
            else:
                splited_key.append(tuple(key[:self.block_size]))
            key = key[self.block_size:]
        if len(splited_key) == 0:
            return 0
        value = value[:len(splited_key)]
        return self.tree_cache.insert(splited_key, value, cpu_free_call_back)
