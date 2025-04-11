from typing import List

import numpy as np
from hakes_client.hakesindex import HakesVts, HakesIVF


class ShardingBaselineConfig:
    """
    ShardingBaselineConfig is used for baseline Sharded-HNSW where the index and data are coupled
    """

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


class ClientConfig:
    def __init__(
        self,
        index_worker_groups: List[List[str]],
        refine_workers: List[str],
        preference: int = 0,
    ):
        if len(index_worker_groups) == 0:
            raise ValueError("No index workers provided")
        if len(refine_workers) == 0:
            raise ValueError("No refine workers provided")
        self.index_worker_groups = index_worker_groups
        self.index_worker_group_count = len(index_worker_groups)
        self.refine_workers = refine_workers
        self.refine_worker_count = len(refine_workers)
        self.preference = preference
        print(
            f"ClientConfig: index worker groups: {len(self.index_worker_groups)}\n{self.index_worker_groups}\nmod route policy among refine workers: ({self.refine_workers}), preference: {self.preference}"
        )

    def __repr__(self) -> str:
        return f"ClientConfig: index worker groups: {len(self.index_worker_groups)}\n{self.index_worker_groups}\nmod route policy among refine workers: ({self.refine_workers}), preference: {self.preference}"

    def get_index_worker_group_id(self, id):
        return id % self.index_worker_group_count

    def get_refine_worker_id(self, id):
        return id % self.refine_worker_count

    def get_index_worker_group_id(self, id):
        return id % self.index_worker_group_count

    def get_preferred_id(self):
        return self.preference


class ClientConfigPA(ClientConfig):
    def __init__(
        self,
        index_worker_groups: List[List[str]],
        refine_workers: List[str],
        vts: HakesVts,
        ivf: HakesIVF,
        preference: int = 0,
    ):
        super().__init__(index_worker_groups, refine_workers, preference)
        self.vts = vts
        self.ivf = ivf
        shard_centroids = self.ivf.ivf_centroids[: self.refine_worker_count]
        self.pa = [[i] for i in range(self.refine_worker_count)]
        avg_len = len(self.ivf.ivf_centroids) // self.refine_worker_count
        for idx, c in enumerate(self.ivf.ivf_centroids[self.refine_worker_count :]):
            assign_score = np.dot(c, shard_centroids.T)
            assign = np.argsort(-assign_score)
            for i in assign:
                if len(self.pa[i]) < avg_len:
                    self.pa[i].append(idx + self.refine_worker_count)
                    break

        self.reverse_pa = {}
        for i in range(len(self.pa)):
            for j in range(len(self.pa[i])):
                self.reverse_pa[self.pa[i][j]] = i
        print(
            f"ClientConfig: mod route policy among servers: ({self.refine_workers}): {self.pa}"
        )

    def __repr__(self) -> str:
        return f"PAConfig: mod route policy among servers: ({self.refine_workers}): {self.pa}"

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.vts.serialize())
            f.write(self.ivf.serialize())

    @classmethod
    def load(cls, cfg: ClientConfig, path: str):
        with open(path, "rb") as f:
            vts, s = HakesVts.deserialize(f.read())
            ivf, _ = HakesIVF.deserialize(s)
        return cls(
            cfg.index_worker_groups, cfg.refine_workers, vts, ivf, cfg.preference
        )

    def get_ivf_assign(self, vecs: np.ndarray):
        return self.ivf.apply(self.vts.apply(vecs))

    def get_refine_worker_id(self, assignment: int):
        return self.reverse_pa[assignment]
