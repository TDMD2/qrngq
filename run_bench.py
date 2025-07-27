#!/usr/bin/env python3
import time
import torch

from philox_hybrid import AsyncQRNGPhilox

def measure(fn, n=10_000_000):
    torch.cuda.synchronize()
    t0 = time.time()
    fn(n)
    torch.cuda.synchronize()
    return n / (time.time() - t0)

def baseline(n):
    # cuRAND-backed default RNG
    _ = torch.cuda.FloatTensor(n).random_()

def hybrid(n):
    # Your QRNG-seeded Philox generator
    gen = AsyncQRNGPhilox(
        seed_file="qr_seed.bin",
        buffer_size=1 << 20  # 1 MiB of pre-fetched random bits
    )
    # generate n floats
    _ = gen.rand(n)

if __name__ == "__main__":
    for name, fn in [("cuRAND", baseline), ("QRNG+", hybrid)]:
        thr = measure(fn)
        print(f"{name} throughput: {thr:,.0f} rands/sec")
