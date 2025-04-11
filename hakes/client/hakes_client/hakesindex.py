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

    def __repr__(self) -> str:
        return f"HakesVt: A.shape: {self.A.shape}, b.shape: {self.b.shape}"


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

    def __repr__(self) -> str:
        return f"HakesVts: {len(self.vt_list)} VTs, " + ", ".join(
            [str(vt) for vt in self.vt_list]
        )


class HakesIVF:
    def __init__(self, ivf_centroids: np.ndarray, metric="ip"):
        self.ivf_centroids = ivf_centroids
        self.metric = metric

    @classmethod
    def deserialize(cls, s: bytes):
        d = struct.unpack("<i", s[:4])[0]
        _ = struct.unpack("<Q", s[4:12])[0]
        metric_type = struct.unpack("B", s[12:13])[0]
        if metric_type == 0:
            metric = "l2"
        else:
            metric = "ip"
        nlist = struct.unpack("<i", s[13:17])[0]
        ivf_end = 17 + nlist * d * 4
        ivf_centroids = np.frombuffer(s[17:ivf_end], dtype="<f").reshape(nlist, d)
        return HakesIVF(ivf_centroids, metric), s[ivf_end:]

    def apply(self, target: np.ndarray):
        if self.metric == "ip":
            return np.argmax(np.dot(target, self.ivf_centroids.T), axis=-1)
        else:
            return np.argmin(
                np.linalg.norm(target[:, None] - self.ivf_centroids, axis=-1), axis=-1
            )

    def serialize(self) -> bytes:
        return (
            struct.pack(
                "<iQBi",
                self.ivf_centroids.shape[1],
                0,
                0 if self.metric == "l2" else 1,
                self.ivf_centroids.shape[0],
            )
            + self.ivf_centroids.tobytes()
        )

    def __repr__(self) -> str:
        return f"HakesIVF: ivf_centroids.shape: {self.ivf_centroids.shape}, metric: {self.metric}"


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

    def __repr__(self) -> str:
        return f"HakesPQ: d: {self.d}, m: {self.m}, nbits: {self.nbits}, codebook.shape: {self.codebook.shape}"


class HakesIndexParams:
    def __init__(self, vts: HakesVts, ivf: HakesIVF, pq: HakesPQ):
        self.vts = vts
        self.ivf = ivf
        self.pq = pq

    @classmethod
    def deserialize(cls, s: bytes):
        vts, s = HakesVts.deserialize(s)
        ivf, s = HakesIVF.deserialize(s)
        pq, _ = HakesPQ.deserialize(s)
        return HakesIndexParams(vts, ivf, pq)

    @classmethod
    def load_from(cls, path: bytes):
        if not os.path.exists(path):
            raise ValueError(f"Index {path} does not exist")
        with open(path, "rb") as f:
            s = f.read()
            return cls.deserialize(s)

    def apply_dr(self, x: np.ndarray):
        return self.vts.apply(x)

    def get_partition(self, target: np.ndarray):
        vecs_t = self.vts.apply(target)
        return self.ivf.apply(vecs_t)

    def serialize(self) -> bytes:
        return self.vts.serialize() + self.ivf.serialize() + self.pq.serialize()

    def _repr__(self) -> str:
        return f"HakesIndexParams: {self.vts}, {self.ivf}, {self.pq}"
