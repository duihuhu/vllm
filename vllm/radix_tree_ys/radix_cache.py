import heapq
import time
import string
import enum
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List
from vllm.utils import Device
from vllm.block import PhysicalTokenBlock

from vllm.logger import init_logger

logger = init_logger(__name__)

class kvCacheProgressStatus(enum.Enum):
    """Status of a SequenceData in RTC."""
    # 稳定状态
    STABLE = enum.auto()
    # 正在换出
    SWAPPING_OUT = enum.auto()
    # 正在换入
    SWAPPING_IN = enum.auto()
    
    def __str__(self) -> str:
        if self == kvCacheProgressStatus.STABLE:
            return "STABLE"
        elif self == kvCacheProgressStatus.SWAPPING_OUT:
            return "SWAPPING_OUT"
        else:
            return "SWAPPING_IN"

class TreeNodeValue:
    def __init__(
        self, 
        physicalTokenBlock: PhysicalTokenBlock, 
        progressStatus: kvCacheProgressStatus = kvCacheProgressStatus.STABLE):
        self.physicalTokenBlock = physicalTokenBlock
        self.progressStatus = progressStatus

class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.value: TreeNodeValue = None
        self.last_access_time = time.time()

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


#按位比较key和seq的单个元素
# 如果seq比key长，则zip以key为准
def match(key: Tuple, seq: Tuple):
    return key == seq

class RadixCache:
    def __init__(self, block_size: int):
        self.root_node = TreeNode()
        self.root_node.value = TreeNodeValue(PhysicalTokenBlock(None, -1, block_size, -1, -1))
        self.block_size = block_size
        

    ##### Public API #####
    # 查找匹配点位置，并返回匹配点的value
    def match_prefix(self, key : List[Tuple]):
        nodes: List[TreeNode] = []

        self._match_prefix_helper(self.root_node, key, nodes)
        return nodes

    ##### Public API #####
    # 返回插入到radix tree的路径段数
    def insert(
        self, 
        key : List[Tuple], 
        block_table : List[PhysicalTokenBlock],
        cpu_free_call_back,
    ) -> int:
        return self._insert_helper(self.root_node, key, block_table, cpu_free_call_back)

    def pretty_print(self):
        logger.info("root node")
        self._print_helper(self.root_node, 4)
        logger.info(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self.get_num_nodes() * self.block_size

    #按num_tokens个数淘汰,node的value存的是内存的索引，不是真的内存。kvcache修改函数：self._free_value_block
    def evict(self, num_nodes, device: Device, evict_callback):

        leaves = self._collect_leaves_for_evict(device)
        # 将list用最小堆，按最长访问时间排序
        heapq.heapify(leaves)

        def evict_node(node: TreeNode, device: Device) -> int:
            num_node_evicted = 0
            for child in node.children.values():
                num_node_evicted += evict_node(child)
            evict_callback(node)
            if node.value.physicalTokenBlock.device == device:
                num_node_evicted += 1
            return num_node_evicted
        
        def get_num_child(node: TreeNode, device: Device = None):
            num_child = 0
            for child in node.children.values():
                if not device or child.value.physicalTokenBlock.device == device:
                    num_child += 1
            return num_child
        
        num_evicted = 0
        #两种条件退出：
        # 1、 满足淘汰数量num_evicted < num_tokens后，退出
        # 2、len(leaves) == 0
        while num_evicted < num_nodes and len(leaves):
            #取最小堆的最小项, len(leaves)会变成退出条件
            x = heapq.heappop(leaves)

            if x == self.root_node:
                #所有节点都删完了，根节点不删除，退出
                logger.info("only root node left")
                break
            # 引用计数为1才淘汰
            if x.value.physicalTokenBlock.ref_count > 1:
                continue

            num_evicted += evict_node(x, device)
            # TODO check this function
            self._delete_leaf(x)

            #如果叶子节点删完了，父节点变叶子节点，进入淘汰候选list
            if (get_num_child(x.parent, device) == 0 and 
                x.parent.value.physicalTokenBlock.ref_count == 1 and 
                x.parent.value.physicalTokenBlock.device == device and 
                x.parent.value.progressStatus == kvCacheProgressStatus.STABLE):
                heapq.heappush(leaves, x.parent)
        return num_evicted

    #按num_tokens个数swap,node的value存的是内存的索引，不是真的内存。kvcache释放函数：self._modify_tree_value_block
    #TODO::radix_tree 暂时还没定淘汰接口
    def swap_out(self, num_tokens):

        nodes = self._collect_nodes_for_swap()
        # 将list用最小堆，按最长访问时间排序
        heapq.heapify(nodes)
        
        def get_num_child(node: TreeNode):
            num_child = 0
            for child in node.children.values():
                if (child.value.physicalTokenBlock.device == Device.GPU and 
                    child.value.progressStatus == kvCacheProgressStatus.STABLE):
                    num_child += 1
            return num_child
        
        swap_out_nodes: List[TreeNode] = []
        while len(swap_out_nodes) < num_tokens and len(nodes):
            #取最小堆的最小项
            x: TreeNode = heapq.heappop(nodes)

            if x == self.root_node:
                break

            swap_out_nodes.append(x)
            x.value.progressStatus = kvCacheProgressStatus.SWAPPING_OUT
            
            if (get_num_child(x.parent) == 0 and 
                x.parent.value.physicalTokenBlock.ref_count == 1 and 
                x.parent.value.progressStatus == kvCacheProgressStatus.STABLE):
                heapq.heappush(nodes, x.parent)

            #TODO：：1.如果叶子节点swap完了，如果还不满足swap要求，此时再调一次swap？
            # 2.blocks swap的顺序， 可能不是严格按时间排序。
            # 3.方案可能造成每个前缀序列都有一小部分在cpu上，cost model复杂。
            
        return swap_out_nodes

    ##### Internal Helper Functions #####
    def _match_prefix_helper(self, node: TreeNode, key : List[Tuple], nodes: List[TreeNode]):
        node.last_access_time = time.time()

        is_match = False
        for c_key, child in node.children.items():
            is_match = match(c_key, key[0])
            if is_match:
                nodes.append(child)
                key.pop(0)
                if len(key) == 0:
                    #如果匹配到底了，退出
                    break
                self._match_prefix_helper(child, key, nodes)
                break

    def _insert_helper(
        self, 
        node: TreeNode, 
        key : List[Tuple], 
        block_table : List[PhysicalTokenBlock],
        cpu_free_call_back,
    ) -> int:
        node.last_access_time = time.time()

        if not key: return 0

        #child 是一个node
        is_match = False
        for c_key, child in node.children.items():
            is_match = match(c_key, key[0])

            # c_key被完全匹配了，说明需要新增或者就是完全匹配
            if is_match:
                if (child.value.progressStatus == kvCacheProgressStatus.SWAPPING_IN or 
                    child.value.progressStatus == kvCacheProgressStatus.SWAPPING_OUT):
                    return 0
                if child.value.physicalTokenBlock.device == Device.CPU:
                    cpu_free_call_back([child.value.physicalTokenBlock])
                    child.value.progressStatus = kvCacheProgressStatus.STABLE
                    child.value.physicalTokenBlock = block_table[0]
                    block_table[0].ref_count += 1
                # if child.value.physicalTokenBlock.block_number != block_table[0].block_number:
                #往下insert
                key.pop(0)
                block_table.pop(0)
                return 1 + self._insert_helper(child, key, block_table, cpu_free_call_back)

        #所有叶子节点完全不匹配，新增一个child
        if not is_match:
            new_node = TreeNode()
            new_node.parent = node
            block = block_table.pop(0)
            #TODO check why add ref_count in there
            block.ref_count += 1
            new_node.value = TreeNodeValue(block)
            node.children[key.pop(0)] = new_node

            # 往下insert
            if not key and not block_table:
                return 1
            return 1 + self._insert_helper(new_node, key, block_table, cpu_free_call_back)
        return 0

    def _print_helper(self, node: TreeNode, indent):
        for key, child in node.children.items():
            logger.info(" " * indent + f"block id: {child.value.physicalTokenBlock.block_number},"
                  f" device: {child.value.physicalTokenBlock.device}, "
                  f"ref count: {child.value.physicalTokenBlock.ref_count}, "
                  f"status: {child.value.progressStatus}")
            self._print_helper(child, indent=indent + 4)

    def _delete_leaf(self, node: TreeNode):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]

    def _clean_tree(self, free_call_back): #调试用接口
        def clean_child(node: TreeNode):
            child_list = []
            for child in node.children.values():
                child_list.append(child)

            for child in child_list:
                clean_child(child)

            if node != self.root_node:
                free_call_back([node.value.physicalTokenBlock])
                self._delete_leaf(node)

        clean_child(self.root_node)

    #dfs遍历叶子节点
    def _collect_leaves_for_evict(self, device: Device):
        ret_list: List[TreeNode] = []

        def dfs_(cur_node: TreeNode):
            num_child = 0
            for x in cur_node.children.values():
                if x.value.physicalTokenBlock.device == device:
                    num_child += 1
            
            if (num_child == 0 and 
                cur_node.value.physicalTokenBlock.device == device and 
                cur_node.value.progressStatus == kvCacheProgressStatus.STABLE):
                if cur_node.value.physicalTokenBlock.ref_count == 1:
                    ret_list.append(cur_node)
                return

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list
    
    def _collect_nodes_for_swap(self):
        ret_list = []

        def dfs_(cur_node: TreeNode):
            gpu_child: List[TreeNode] = []
            for child_node in cur_node.children.values():
                if (child_node.value.physicalTokenBlock.device == Device.GPU and 
                    child_node.value.progressStatus == kvCacheProgressStatus.STABLE):
                    gpu_child.append(child_node)
            
            # if len(gpu_child) == 0 and child_node.value.physicalTokenBlock.ref_count == 1:
            if len(gpu_child) == 0 and cur_node.value.physicalTokenBlock.ref_count == 1:
                ret_list.append(cur_node)

            for x in gpu_child:
                dfs_(x)

        dfs_(self.root_node)
        return ret_list
    
    def refresh_node_status(self, nodes: List[TreeNode]):
        for node in nodes:
            node.value.progressStatus = kvCacheProgressStatus.STABLE
            
    def get_num_nodes(self, device: Device = None) -> int:
        def dfs_(cur_node: TreeNode):
            num_nodes = 0
            if not device or cur_node.value.physicalTokenBlock.device == device:
                num_nodes += 1
            
            for child_node in cur_node.children.values():
                print("child_node ", child_node.value.physicalTokenBlock.ref_count)
                num_nodes += dfs_(child_node)
            
            return num_nodes
                
        return dfs_(self.root_node)
    
    def get_num_nodes_can_swap_out(self) -> int:
        def dfs_(cur_node: TreeNode):
            num_nodes = 0
            print("cur_node.value.physicalTokenBlock.ref_count ",  cur_node.value.physicalTokenBlock.block_number, cur_node.value.physicalTokenBlock.ref_count)
            if cur_node.value.physicalTokenBlock.device == Device.CPU:
                return 0
            if cur_node.value.physicalTokenBlock.ref_count == 1:
                num_nodes += 1
            
            for child_node in cur_node.children.values():
                num_nodes += dfs_(child_node)
            
            return num_nodes
        
        return dfs_(self.root_node)
            

if __name__ == "__main__":
    tree = RadixCache(disable=False)

    #构造tree value，目的是使字符串key和value一一对应，方便调试
    block = PhysicalTokenBlock(Device.GPU, 1, 0)
    valueA = TreeNodeValue(block, kvCacheProgressStatus.STABLE)
    block = PhysicalTokenBlock(Device.GPU, 2, 0)
    valueB = TreeNodeValue(block, kvCacheProgressStatus.STABLE)
    block = PhysicalTokenBlock(Device.GPU, 3, 0)
    valueC = TreeNodeValue(block, kvCacheProgressStatus.STABLE)
    block = PhysicalTokenBlock(Device.GPU, 4, 0)
    valueD = TreeNodeValue(block, kvCacheProgressStatus.STABLE)
    block = PhysicalTokenBlock(Device.GPU, 5, 0)
    valueE = TreeNodeValue(block, kvCacheProgressStatus.STABLE)
    block = PhysicalTokenBlock(Device.GPU, 6, 0)
    valueF = TreeNodeValue(block, kvCacheProgressStatus.STABLE)
    block = PhysicalTokenBlock(Device.GPU, 7, 0)
    valueG = TreeNodeValue(block, kvCacheProgressStatus.STABLE)
    block = PhysicalTokenBlock(Device.GPU, 8, 0)
    valueH = TreeNodeValue(block, kvCacheProgressStatus.STABLE)
    block = PhysicalTokenBlock(Device.GPU, 9, 0)
    valueI = TreeNodeValue(block, kvCacheProgressStatus.STABLE)
    block = PhysicalTokenBlock(Device.GPU, 10, 0)
    valueJ = TreeNodeValue(block, kvCacheProgressStatus.STABLE)

    #构造radix tree
    tree.insert(["HelloHelloHelloA", "HelloHelloHelloB", "HelloHelloHelloC"], [valueA, valueB, valueC])
    tree.insert(["HelloHelloHelloA", "HelloHelloHelloD", "HelloHelloHelloE"], [valueA, valueD, valueE])
    tree.insert(["HelloHelloHelloF", "HelloHelloHelloG", "HelloHelloHelloH"], [valueF, valueG, valueH])
    #打印radix tree
    tree.pretty_print()
    #匹配radix tree，完全匹配
    value, last_node = tree.match_prefix(["HelloHelloHelloA", "HelloHelloHelloB", "HelloHelloHelloC"])
    print(value)
    print(last_node.value.__dict__)
    #匹配radix tree，部分匹配
    value, last_node = tree.match_prefix(["HelloHelloHelloA", "HelloHelloHelloI", "HelloHelloHelloJ"])
    print(value)
    print(last_node.value.__dict__)
    #匹配radix tree，完全匹配，输入比路径长
    value, last_node = tree.match_prefix(["HelloHelloHelloA", "HelloHelloHelloB", "HelloHelloHelloC","HelloHelloHelloD"])
    print(value)
    print(last_node.value.__dict__)
    #匹配radix tree，完全匹配
    value, last_node = tree.match_prefix(["HelloHelloHelloF", "HelloHelloHelloG", "HelloHelloHelloH"])
    print(value)
    print(last_node.value.__dict__)

    #swap测试
    def _swap_callback(x):
       print("swap", x.__dict__)
       x.progressStatus = kvCacheProgressStatus.SWAPPING_OUT
       return 1

    expect_num_swaped = 2
    num_swaped = tree.swap(expect_num_swaped, _swap_callback)
    if expect_num_swaped != num_swaped:
        print("not equal ! num_swaped", num_swaped)
        print("not equal ! expect_num_swaped", expect_num_swaped)
    #swap返回的数量不够，对H减引用技术，继续swap
    expect_num_swaped = expect_num_swaped - num_swaped
    num_swaped = tree.swap(expect_num_swaped, _swap_callback)
    if expect_num_swaped != num_swaped:
        print("not equal ! num_swaped", num_swaped)
        print("not equal ! expect_num_swaped", expect_num_swaped)
    else:
        print("equal ! num_swaped", num_swaped)
        print("equal ! expect_num_swaped", expect_num_swaped)

    #淘汰测试
    def _evict_callback(x):
       print("evict", x.__dict__)
       return 1
    tree.evict(5, _evict_callback)

    tree.pretty_print()
