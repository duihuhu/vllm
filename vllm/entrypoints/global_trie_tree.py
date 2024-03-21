class TrieNode:
  def __init__(self):
      self.children = {}
      self.request_ids = set()

class Trie:
  def __init__(self):
    self.root = TrieNode()

  def insert(self, tokens, request_id):
    node = self.root
    for token in tokens:
      if token not in node.children:
        node.children[token] = TrieNode()
        node.children[token].request_ids.add(request_id)
      else:
        node.children[token].request_ids.add(request_id)
      node = node.children[token]
      
  def delete(self, tokens, request_id):
    node = self.root
    for token in tokens:
      if token in node.children:
        node.children[token].request_ids.remove(request_id)
        if node.children[token].request_ids:
          node = node.children[token]
        else:
          #no other request has this path
          del node.children[token]
          break
        
  def search(self, tokens):
    node = self.root
    prefix_len = 0
    start_Node = TrieNode()
    end_Node = TrieNode()
    for token in tokens:
      if token in node.children and prefix_len == 0:
        start_Node = node.children[token]
        end_Node = node.children[token]
        prefix_len = prefix_len + 1
      elif token in node.children and prefix_len != 0 :
        end_Node = node.children[token]
        prefix_len = prefix_len + 1 
      elif token not in node.children:
          break
      node = node.children[token]
      
    req_res = start_Node.request_ids & end_Node.request_ids
    return req_res, prefix_len 
  
  # def udpate(self, tokens, request_id):
  #   if reqs_relation_table.get(request_id):
  #     prio_req = reqs_relation_table[request_id]
  #     if prio_req.type == "part":
  #       self.insert(tokens=tokens, request_id=request_id)
  #     else:
  #       prio_request_id = prio_req.request_id
  #       node = self.root
  #       for token in tokens:
  #         if token not in node.children:
  #           node.children[token] = TrieNode()
  #           node.children[token].request_ids.add(request_id)
  #         else:
  #           node.children[token].request_ids.add(request_id)
  #           node.children[token].request_ids.remove(prio_request_id)
  #         node = node.children[token]
  #   else:
  #     self.insert(tokens=tokens, request_id=request_id)
  


# # 建立公共前缀树
# trie = Trie()
# listA = [1, 2, 3, 4]
# listB = [1, 5, 7, 8]
# listC = [0, 4, 1, 2]

# trie.insert(listA,"1111")
# trie.insert(listB,"2222")
# trie.insert(listC, "3333")

# # trie.delete(listA, "1111")

# req_res, prefix_len = trie.search([1,2])
# print(req_res, prefix_len)
# #todo choose one request_id in req_res