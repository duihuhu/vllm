import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple
import sys
sys.setrecursionlimit(100000)

class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.value = None
        self.ref_counter = 0
        self.last_access_time = time.time()
        self.node_addr = [] 
        
    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


class RadixCache:
    def __init__(self, disable=False):
        self.reset()
        self.disable = disable

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.value = []
        self.root_node.ref_counter = 1
        self.evictable_size_ = 0

    def match_prefix(self, key):
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node)
        # if value:
        #     print(value)
            # value = torch.concat(value)
            # value.append(value)
        return value, last_node[0]

    def only_match_prefix(self, key):
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        
        last_node_matched_len = [0]
        self._only_match_prefix_helper(self.root_node, key, value, last_node, last_node_matched_len)
        # if value:
        #     print(value)
            # value = torch.concat(value)
            # value.append(value)
        return value, last_node[0], last_node_matched_len[0]
    
    def _only_match_prefix_helper(self, node, key, value, last_node, last_node_matched_len):
        node.last_access_time = time.time()

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len != 0:
                if prefix_len < len(c_key):
                    # for val in child.value[:prefix_len]:
                    #     val.ref_count += 1
                    value.extend(child.value[:prefix_len])
                    last_node[0] = child
                    last_node_matched_len[0] = prefix_len
                else:
                    last_node_matched_len[0] = prefix_len
                    # for val in child.value:
                    #     val.ref_count += 1
                    value.extend(child.value)
                    last_node[0] = child
                    self._only_match_prefix_helper(child, key[prefix_len:], value, last_node, last_node_matched_len)
                # break
            
    def insert(self, key, value=None, addr=None):
        if self.disable:
            return len(key)

        last_len = [0]
        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, addr, last_len), last_len[0]

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens, evict_callback):
        if self.disable:
            raise RuntimeError()

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.ref_counter > 0:
                continue

            num_evicted += evict_callback(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            if node.ref_counter == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.ref_counter += 1
            node = node.parent
        return delta

    def dec_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            if node.ref_counter == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.ref_counter -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    ##### Internal Helper Functions #####
    def _match_prefix_helper(self, node, key, value, last_node):

        node.last_access_time = time.time()
        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len != 0:
                if prefix_len < len(c_key):
                    new_node = self._split_node(c_key, child, prefix_len)
                    value.append(new_node.value)
                    last_node[0] = new_node
                else:
                    value.append(child.value)
                    last_node[0] = child
                    self._match_prefix_helper(child, key[prefix_len:], value, last_node)
                break

    def _split_node(self, key, child, split_len):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:]: child}
        new_node.parent = child.parent
        new_node.ref_counter = child.ref_counter
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len]] = new_node
        del new_node.parent.children[key]
        return new_node

    def _insert_helper(self, node, key, value, addr, last_node_matched_len):
        node.last_access_time = time.time()

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)

            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    child.node_addr.append(addr)
                    last_node_matched_len[0] = prefix_len
                    return prefix_len, child
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    pre_len, last_node = self._insert_helper(child, key, value, addr, last_node_matched_len)
                    return prefix_len + pre_len, last_node

            if prefix_len:
                new_node = self._split_node(c_key, child, prefix_len)
                pre_len, last_node =  self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:], addr, last_node_matched_len
                )
                return prefix_len + pre_len, last_node

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.value = value
            node.children[key] = new_node
            new_node.node_addr.append(addr)
            node.node_addr.append(addr)
            self.evictable_size_ += len(value)
            last_node_matched_len[0] = len(value)
            return len(key), new_node
        return 0, node

    def _print_root_node(self):
        for c_key, child in self.root_node.children.items():
            print("c_key ", c_key)
            
    def _print_helper(self, node, indent):
        for key, child in node.children.items():
            print(" " * indent, len(key), key[:10], f"r={child.ref_counter}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(k)

    def _total_size_helper(self, node):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list


if __name__ == "__main__":
    # tree = RadixCache(disable=False)

    # tree.insert("HelloA")
    # tree.insert("HelloB")
    # tree.insert("Hello_L.A.!")
    # # tree.insert("Hello_world! Happy")
    # # tree.insert("I love you!")
    # tree.pretty_print()
    
    a = tuple([508, 366, 2874])
    b = tuple([508, 366, 2874, 263, 716, 29889])
    c = tuple([508, 366, 2874, 263, 716, 29889])
    tree = RadixCache(disable=False)
    last_len = [0]
    matched_len, last_node = tree._insert_helper(tree.root_node, a, a, last_len)
    
    tree._insert_helper(last_node.parent, b, b[:-1], last_len)
    tree.pretty_print()
    blocks, last_node, last_node_matched_len = tree.only_match_prefix(c)
    
    print(len(blocks))

    tree.pretty_print()
    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
