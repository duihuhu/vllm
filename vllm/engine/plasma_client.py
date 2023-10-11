import pyarrow._plasma as plasma_object

class PlasmaClient:
    def __init__(self, plasma_store_socket_name) -> None:
        self.plasma_client_ = plasma_object.connect(plasma_store_socket_name)
    
    def allocate_object_id():
        obj_id = plasma_object.ObjectID.from_random()
        return obj_id
    
    def create(self, object_id, length):
        obj = self.plasma_client_.create(object_id, length)
        return obj