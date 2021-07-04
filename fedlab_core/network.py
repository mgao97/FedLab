import torch.distributed as dist
from torch.multiprocessing import Process


class DistNetwork(object):
    """Manage torch.distributed network
    
    Args:
        address (tuple): Address of this server in form of ``(SERVER_ADDR, SERVER_IP)``
        world_size (int): the size of this distributed group (including server).
        rank (int): the rank of process in distributed group.
        dist_backend (str or Backend): :attr:`backend` of ``torch.distributed``. Valid values include ``mpi``, ``gloo``,
        and ``nccl``. Default: ``"gloo"``.
    """
    def __init__(self, address, world_size, rank, dist_backend='gloo'):
        super(DistNetwork, self).__init__()
        self.address = address
        self.rank = rank
        self.world_size = world_size
        self.dist_backend = dist_backend

    def init_network_connection(self):
        print("torch.distributed initializeing processing group with ip address {}:{}, rank {}, world size: {}, backend: {}".format(self.address[0],self.address[1],self.rank, self.world_size, self.dist_backend))
        dist.init_process_group(backend=self.dist_backend,
                                init_method='tcp://{}:{}'.format(
                                    self.address[0],
                                    self.address[1]),
                                rank=self.rank,
                                world_size=self.world_size)
    
    def show_configuration(self):
        info_str = "ip address {}:{}, rank {}, world size: {}, backend: {}".format(self.address[0],self.address[1],self.rank, self.world_size, self.dist_backend)
        print(info_str)
        return info_str