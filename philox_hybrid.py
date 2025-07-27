import torch
import mmap
import os

class AsyncQRNGPhilox:
    def __init__(self, seed_file: str, buffer_size: int = 1 << 20):
        self.buffer_size = buffer_size
        # memory-map the true-random seed buffer
        f = open(seed_file, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()
        # initialize Philox generator on CUDA
        self.gen = torch.cuda.random.BitGeneratorPhilox(device=None)
        # seed it once with first 8 bytes of QRNG
        seed = int.from_bytes(self.mm.read(8), "little")
        self.gen.manual_seed(seed)
        # reset the mmap pointer
        self.mm.seek(0)

    def rand(self, n: int):
        # generate n random floats via Philox
        return torch.rand(n, generator=self.gen, device="cuda")
