import pyarrow._plasma as plasma_object

class PlasmaClient:
    def __init__(self, plasma_store_socket_name) -> None:
        self.plasma_client_ = plasma_object.connect(plasma_store_socket_name)
    
    def allocate_object_id(self):
        obj_id = plasma_object.ObjectID.from_random()
        return obj_id
    
    def create(self, object_id, length):
        obj = self.plasma_client_.create(object_id, length)
        return obj
    
    def seal(self, object_id):
        self.plasma_client_.seal(object_id)
  
    def get_buffers(self, object_id):
        return self.plasma_client_.get_buffers([object_id])
plasma_client = PlasmaClient("/tmp/plasma_store")