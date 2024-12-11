import numpy as np
import struct
import os
from typing import List


class HakesVt:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b

    def apply(self, x: np.ndarray):
        return np.dot(x, self.A.T) + self.b

    @classmethod
    def deserialize(cls, s: bytes):
        d_out = struct.unpack("<i", s[:4])[0]
        d_in = struct.unpack("<i", s[4:8])[0]
        A_start = 8
        b_start = 8 + d_out * d_in * 4
        b_end = b_start + d_out * 4
        A = np.frombuffer(s[A_start:b_start], dtype="<f").reshape(d_out, d_in)
        b = np.frombuffer(s[b_start:b_end], dtype="<f")
        return HakesVt(A, b), s[b_end:]

    def serialize(self) -> bytes:
        s = (
            struct.pack("<ii", self.A.shape[0], self.A.shape[1])
            + self.A.tobytes()
            + self.b.tobytes()
        )
        return s


class HakesVts:
    def __init__(self, vt_list: List[HakesVt]):
        self.vt_list = vt_list

    def apply(self, x: np.ndarray):
        for vt in self.vt_list:
            x = vt.apply(x)
        return x

    @classmethod
    def deserialize(cls, s: bytes):
        vt_list = []
        n = struct.unpack("i", s[:4])[0]
        s = s[4:]
        for _ in range(n):
            vt, s = HakesVt.deserialize(s)
            vt_list.append(vt)
        return HakesVts(vt_list), s

    def serialize(self) -> bytes:
        s = struct.pack("<i", len(self.vt_list))
        for vt in self.vt_list:
            s += vt.serialize()
        return s


class HakesIVF:
    def __init__(self, ivf_centroids: np.ndarray):
        self.ivf_centroids = ivf_centroids

    @classmethod
    def deserialize(cls, s: bytes):
        _ = struct.unpack("<i", s[:4])[0] == 1
        nlist = struct.unpack("<i", s[4:8])[0]
        d = struct.unpack("<i", s[8:12])[0]
        ivf_end = 12 + nlist * d * 4
        ivf_centroids = np.frombuffer(s[12:ivf_end], dtype="<f").reshape(nlist, d)
        return HakesIVF(ivf_centroids), s[ivf_end:]

    def apply(self, target: np.ndarray):
        return np.argmax(np.dot(target, self.ivf_centroids.T), axis=-1)

    def serialize(self) -> bytes:
        return (
            struct.pack(
                "<iii", 0, self.ivf_centroids.shape[0], self.ivf_centroids.shape[1]
            )
            + self.ivf_centroids.tobytes()
        )


class HakesPQ:
    def __init__(self, d, m, nbits: int, codebook: np.ndarray):
        self.d = d
        self.m = m
        self.nbits = nbits
        self.codebook = codebook

    @classmethod
    def deserialize(cls, s: bytes):
        d = struct.unpack("<i", s[:4])[0]
        m = struct.unpack("<i", s[4:8])[0]
        nbits = struct.unpack("<i", s[8:12])[0]
        codebook_end = 12 + d * (1 << nbits) * 4
        codebook = np.frombuffer(s[12:codebook_end], dtype="<f")
        return HakesPQ(d, m, nbits, codebook), s[codebook_end:]

    def serialize(self) -> bytes:
        return struct.pack("<iii", self.d, self.m, self.nbits) + self.codebook.tobytes()


class HakesIndexParams:
    def __init__(self, vts: HakesVts, ivf_vts: HakesVts, ivf: HakesIVF, pq: HakesPQ):
        self.vts = vts
        self.ivf_vts = ivf_vts
        self.ivf = ivf
        self.pq = pq

    @classmethod
    def deserialize(cls, s: bytes):
        vts, s = HakesVts.deserialize(s)
        ivf_vts, s = HakesVts.deserialize(s)
        ivf, s = HakesIVF.deserialize(s)
        pq, _ = HakesPQ.deserialize(s)
        return HakesIndexParams(vts, ivf_vts, ivf, pq)

    @classmethod
    def load_from_dir(cls, path: bytes):
        vts = None
        ivf_vts = None
        ivf = None
        pq = None
        if not os.path.exists(path):
            raise ValueError(f"Index directory {path} does not exist")
        for f in os.listdir(path):
            if f == "pre-transform.bin":
                with open(os.path.join(path, f), "rb") as f:
                    vts = HakesVts.deserialize(f.read())
            elif f == "ivf.bin":
                with open(os.path.join(path, f), "rb") as f:
                    ivf = HakesIVF.deserialize(f.read())
            elif pq == "pq.bin":
                with open(os.path.join(path, f), "rb") as f:
                    pq = HakesPQ.deserialize(f.read())
        if ivf_vts is None:
            ivf_vts = HakesVts([])

        return HakesIndexParams(vts, ivf, pq)

    def apply_dr(self, x: np.ndarray):
        return self.vts.apply(x)

    def get_partition(self, target: np.ndarray):
        vecs_t = self.vts.apply(target)
        return self.ivf.apply(vecs_t)

    def serialize(self) -> bytes:
        return (
            self.vts.serialize()
            + self.ivf_vts.serialize()
            + self.ivf.serialize()
            + self.pq.serialize()
        )
