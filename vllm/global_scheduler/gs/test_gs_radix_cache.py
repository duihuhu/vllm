import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List

class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.last_access_time = time.time()
        self.instances = []

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


#按位比较key和seq的单个元素
# 如果seq比key长，则zip以key为准
def match(key: Tuple, seq: Tuple):
    return key == seq

class RadixCache:
    def __init__(self, block_size: int):
        self.root_node = TreeNode()
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
        instance: str
    ) -> int:
        return self._insert_helper(self.root_node, key, instance)

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
        instance: str
    ) -> int:
        node.last_access_time = time.time()

        if not key: return 0

        #child 是一个node
        is_match = False
        for c_key, child in node.children.items():
            is_match = match(c_key, key[0])

            # c_key被完全匹配了，说明需要新增或者就是完全匹配
            if is_match:
                child.instances.append(instance)
                key.pop(0)
                return 1 + self._insert_helper(child, key, instance)

        #所有叶子节点完全不匹配，新增一个child
        if not is_match:
            new_node = TreeNode()
            new_node.parent = node
            new_node.instances.append(instance)
            node.children[key.pop(0)] = new_node
            
            return 1 + self._insert_helper(new_node, key, instance)
        return 0

    def _print_helper(self, node, indent):
        for key, child in node.children.items():
            print(" " * indent, len(key), key[:10])
            self._print_helper(child, indent=indent + 2)


    def _delete_leaf(self, node: TreeNode):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]

    def _clean_tree(self): #调试用接口
        def clean_child(node: TreeNode):
            child_list = []
            for child in node.children.values():
                child_list.append(child)

            for child in child_list:
                clean_child(child)

            if node != self.root_node:
                self._delete_leaf(node)

        clean_child(self.root_node)


