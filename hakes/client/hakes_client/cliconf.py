from typing import List

import numpy as np
from hakes_client.hakesindex import HakesVts, HakesIVF


class ClientConfig:
    def __init__(self, addrs: List[str], preference: int = 0):
        self.addrs = addrs
        self.n = len(addrs)
        self.preference = preference
        print(
            f"ClientConfig: mod route policy among servers: ({self.addrs}), n {self.n} preference: {self.preference}"
        )

    def __repr__(self) -> str:
        return f"Config: mod route policy among servers: ({self.addrs})"

    def get_server_address(self, id):
        return self.addrs[self.get_server_id(id)]

    def get_server_id(self, id):
        return id % self.n

    def get_server_address_by_id(self, idx):
        return self.addrs[idx]

    def server_count(self):
        return self.n

    def get_preferred_id(self):
        return self.preference


class ClientConfigPA(ClientConfig):
    def __init__(
        self, addrs: List[str], vts: HakesVts, ivf: HakesIVF, preference: int = 0
    ):
        super().__init__(addrs, preference)
        self.vts = vts
        self.ivf = ivf
        shard_centroids = self.ivf.ivf_centroids[: self.n]
        self.pa = [[i] for i in range(self.n)]
        avg_len = len(self.ivf.ivf_centroids) // self.n
        for idx, c in enumerate(self.ivf.ivf_centroids[self.n :]):
            assign_score = np.dot(c, shard_centroids.T)
            assign = np.argsort(-assign_score)
            for i in assign:
                if len(self.pa[i]) < avg_len:
                    self.pa[i].append(idx + self.n)
                    break

        self.reverse_pa = {}
        for i in range(len(self.pa)):
            for j in range(len(self.pa[i])):
                self.reverse_pa[self.pa[i][j]] = i

    def __repr__(self) -> str:
        return f"PAConfig: mod route policy among servers: ({self.addrs}): {self.pa}"

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.vts.serialize())
            f.write(self.ivf.serialize())

    @classmethod
    def load(cls, cfg: ClientConfig, path: str):
        with open(path, "rb") as f:
            vts, s = HakesVts.deserialize(f.read())
            ivf, _ = HakesIVF.deserialize(s)
        return cls(cfg.addrs, vts, ivf, cfg.preference)

    def get_ivf_assign(self, vecs: np.ndarray):
        return self.ivf.apply(self.vts.apply(vecs))

    def get_server_id(self, assignment: int):
        return self.reverse_pa[assignment]
