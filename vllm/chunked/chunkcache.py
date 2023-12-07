from typing import List

class Block:
    def __init__(self,
                 block_id: int) -> None:
        self.block_id = block_id
        self.used = False

class ChunkCacheBlocks:
    def __init__(self,
                 blocks_num: int) -> None:
        self.Blocks: List[Block] = []
        for i in range(blocks_num):
            self.Blocks.append(Block(block_id = i))
    
    def can_allocate(self) -> bool:
        if len(self.Blocks) >= 1:
            return True
        else:
            return False
    
    def allocate_block(self) -> Block:
        if self.can_allocate():
            block = self.Blocks.pop()
            block.used = True
            return block
        else:
            return None
    
    def free_block(self, block: Block):
        block.used = False
        self.Blocks.append(block)