import logging
import requests
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from hakes_client.hakesindex import HakesIndexParams
from hakes_client.message import (
    prepare_add_request,
    prepare_extended_add_request,
    prepare_search_request,
    prepare_rerank_request,
    parse_search_response,
    parse_extended_add_response,
    prepare_delete_request,
    parse_delete_response,
    parse_get_index_response,
    prepare_update_index_request,
    parse_update_index_response,
)
from hakes_client.cliconf import ShardingBaselineConfig, ClientConfig, ClientConfigPA


class Client:
    def __init__(self, url):
        self.url = url

    def add(self, n: int, d: int, vecs: np.ndarray, ids: List[int]):
        data = prepare_add_request(n, d, vecs, ids)
        try:
            response = requests.post(self.url + "/add", json=data)
        except Exception as e:
            logging.warning(f"add failed on {self.url}: {e}")
            return None
        if response.status_code != 200:
            logging.warning(
                f"Failed to call server, status code: {response.status_code} {response.text}"
            )
            return None
        return json.loads(response.text)

    def add_ext(
        self,
        n: int,
        d: int,
        vecs: np.ndarray,
        ids: List[int],
        assign: List[int] = None,
        add_to_refine_only=False,
    ):
        data = prepare_extended_add_request(n, d, vecs, ids, assign, add_to_refine_only)
        try:
            response = requests.post(self.url + "/add", json=data)
        except Exception as e:
            logging.warning(f"add failed on {self.url}: {e}")
            return None
        if response.status_code != 200:
            logging.warning(
                f"Failed to call server, status code: {response.status_code} {response.text}"
            )
            return None
        return parse_extended_add_response(json.loads(response.text))

    def search(
        self,
        query: np.ndarray,
        k: int,
        nprobe: int,
        k_factor: int = 1,
        metric_type: int = 1,
        filter_invalid: bool = True,
        require_pa: bool = False,
    ) -> Dict:
        data = prepare_search_request(
            query.shape[0],
            query.shape[1],
            query,
            k,
            nprobe,
            k_factor,
            metric_type,
            require_pa,
        )
        print(f"search data: {data}")
        try:
            response = requests.post(self.url + "/search", json=data)
        except Exception as e:
            logging.warning(f"search failed on {self.url}: {e}")
            return None
        if response.status_code != 200:
            logging.warning(
                f"Failed to call server, status code: {response.status_code} {response.text}"
            )
            return None
        return parse_search_response(json.loads(response.text), filter_invalid)

    def rerank(
        self,
        query: np.ndarray,
        k: int,
        k_base_count: List[int],
        base_ids: List[int],
        base_scores: List[float],
        metric_type: int = 1,
        filter_invalid: bool = True,
    ) -> Dict:
        if len(k_base_count) == 0:
            return None
        data = prepare_rerank_request(
            query.shape[0],
            query.shape[1],
            query,
            k,
            metric_type,
            k_base_count,
            base_ids,
            base_scores,
        )
        try:
            response = requests.post(self.url + "/rerank", json=data)
        except Exception as e:
            logging.warning(f"rerank failed on {self.url}: {e}")
            return None
        if response.status_code != 200:
            logging.warning(
                f"Failed to call server, status code: {response.status_code} {response.text}"
            )
            return None
        return parse_search_response(json.loads(response.text), filter_invalid)

    def delete(self, n: int, ids: List[int]):
        data = prepare_delete_request(n, ids)
        try:
            response = requests.post(self.url + "/delete", json=data)
        except Exception as e:
            logging.warning(f"delete failed on {self.url}: {e}")
            return None
        if response.status_code != 200:
            logging.warning(
                f"Failed to call server, status code: {response.status_code} {response.text}"
            )
            return None
        return parse_delete_response(json.loads(response.text))

    def checkpoint(self) -> str:
        try:
            response = requests.post(self.url + "/checkpoint")
        except Exception as e:
            logging.warning(f"checkpoint failed on {self.url}: {e}")
            return None
        if response.status_code != 200:
            logging.warning(
                f"Failed to call server, status code: {response.status_code} {response.text}"
            )
            return None
        return "checkpoint success"

    def get_index(self) -> Dict:
        try:
            response = requests.get(self.url + "/get_index")
        except Exception as e:
            logging.warning(f"get_index_params failed on {self.url}: {e}")
            return None
        if response.status_code != 200:
            logging.warning(
                f"Failed to call server, status code: {response.status_code} {response.text}"
            )
            return None
        return parse_get_index_response(json.loads(response.text))

    def update_index(self, params: HakesIndexParams) -> Dict:
        data = prepare_update_index_request(params)
        try:
            response = requests.post(self.url + "/update_index", json=data)
        except Exception as e:
            logging.warning(f"update_index failed on {self.url}: {e}")
            return None
        if response.status_code != 200:
            logging.warning(
                f"Failed to call server, status code: {response.status_code} {response.text}"
            )
            return None
        return parse_update_index_response(json.loads(response.text))


class ClientShardedHNSW:
    """
    Client for sharded HNSW baseline
    """

    def __init__(self, cfg: ShardingBaselineConfig):
        self.cfg = cfg
        self.clients = [Client(addr) for addr in cfg.addrs]
        self.pool = ThreadPoolExecutor(max_workers=1000)

    def add(self, n, d, vecs, ids):
        """
        Add vectors to the the designated shard and its hnsw index
        """
        # fast path for single request
        if n == 1:
            idx = self.cfg.get_server_id(ids[0])
            return self.clients[idx].add(1, d, vecs, ids)

        # build vector batch for each client
        vec_batches = [[] for _ in range(self.cfg.n)]
        id_batches = [[] for _ in range(self.cfg.n)]
        for i in range(n):
            idx = self.cfg.get_server_id(ids[i])
            vec_batches[idx].append(vecs[i])
            id_batches[idx].append(ids[i])

        # send requests to each server with the threadpool
        futures = [
            self.pool.submit(
                self.clients[i].add,
                len(vec_batches[i]),
                d,
                np.array(vec_batches[i]),
                id_batches[i],
            )
            for i in range(self.cfg.n)
        ]

        # wait for all requests to finish and collect results
        results = [f.result() for f in futures]
        for res in results:
            if res is None:
                return None
        return results

    def search(
        self,
        query: np.ndarray,
        k: int,
        nprobe: int,
        k_factor: int = 1,
        metric_type: int = 1,
    ):
        """
        Search vectors from all shards and merge the results
        """
        # send requests to each server with the threadpool
        futures = [
            self.pool.submit(
                self.clients[i].search, query, k, nprobe, k_factor, metric_type
            )
            for i in range(self.cfg.n)
        ]

        # wait for all requests to finish and collect results
        results = [f.result() for f in futures]
        # merge the results
        nq = query.shape[0]
        collated_dist_ll = [[] for _ in range(nq)]
        collated_id_ll = [[] for _ in range(nq)]
        for i in range(len(results)):
            if results[i] is None:
                logging.warning(
                    f"skipping empty search result on server {self.clients[i].url}"
                )
                continue
            for j in range(nq):
                collated_dist_ll[j].extend(results[i]["scores"][j])
                collated_id_ll[j].extend(results[i]["ids"][j])
        final_result = []
        for j in range(nq):
            # sort collated ids based on collated dist
            # for ip, we sort them in descending order so flip the argsort return
            # -1 entries have a default distance of FLOAT_MIN
            sorted_idx = np.flip(np.argsort(collated_dist_ll[j]))
            collated_id_ll[j] = np.array(collated_id_ll[j])[sorted_idx]
            collated_dist_ll[j] = np.array(collated_dist_ll[j])[sorted_idx]
            final_result.append(
                {"ids": collated_id_ll[j][:k], "scores": collated_dist_ll[j][:k]}
            )
        return final_result

    def checkpoint(self):
        """
        Checkpoint
        """
        # send requests to each server with the threadpool
        futures = [self.pool.submit(client.checkpoint) for client in self.clients]

        # wait for all requests to finish and collect results
        results = [f.result() for f in futures]
        for res in results:
            if res is None:
                return None
        return results


class HakesClient:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        # The most common case should be 1 index worker group
        # and index workers in the group replicate the filter stage index
        self.index_worker_groups = [
            [Client(addr) for addr in group] for group in cfg.index_worker_groups
        ]
        self.refine_workers = [Client(addr) for addr in cfg.refine_workers]
        self.pool = ThreadPoolExecutor(max_workers=1000)

    def add(self, n, d, vecs, ids):
        """
        Add vectors to the distributed HakesService V3
            1. add to the target refine index server
            2. add to all base index servers
        """
        # fast path for single request
        if vecs.shape[0] != n:
            logging.warning(f"add failed: vecs shape {vecs.shape} != n {n}")
            return None

        # build vector batch for each client
        vec_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        id_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        for i in range(n):
            refine_worker_idx = self.cfg.get_refine_worker_id(ids[i])
            vec_batches[refine_worker_idx].append(vecs[i])
            id_batches[refine_worker_idx].append(ids[i])

        # send requests to each server with the threadpool
        refine_worker_futures = [
            (
                self.pool.submit(
                    self.refine_workers[i].add_ext,
                    len(vec_batches[i]),
                    d,
                    vec_batches[i],
                    id_batches[i],
                    add_to_refine_only=True,
                )
                if len(vec_batches[i]) != 0
                else None
            )
            for i in range(self.cfg.refine_worker_count)
        ]

        index_worker_vec_batches = [
            [] for _ in range(self.cfg.index_worker_group_count)
        ]
        index_worker_id_batches = [[] for _ in range(self.cfg.index_worker_group_count)]
        for i in range(n):
            index_worker_group_idx = self.cfg.get_index_worker_group_id(ids[i])
            index_worker_vec_batches[index_worker_group_idx].append(vecs[i])
            index_worker_id_batches[index_worker_group_idx].append(ids[i])

        preferred_index_worker = self.cfg.get_preferred_id()
        index_worker_future = [
            (
                self.pool.submit(
                    self.index_worker_groups[i][preferred_index_worker].add_ext,
                    len(index_worker_vec_batches[i]),
                    d,
                    index_worker_vec_batches[i],
                    index_worker_id_batches[i],
                )
                if len(index_worker_vec_batches[i]) != 0
                else None
            )
            for i in range(self.cfg.index_worker_group_count)
        ]

        index_worker_group_round2_futures = []
        for i in range(self.cfg.index_worker_group_count):
            index_worker_count = len(self.index_worker_groups[i])
            vecs_t_d = 0
            # wait for requests to the first round index worker finish
            if index_worker_future[i] is None:
                continue
            res = index_worker_future[i].result()
            if res is None:
                logging.warning(
                    f"add failed on {self.index_worker_groups[i][preferred_index_worker].url}"
                )
                continue
            # add the transformed vectors to other servers
            assign = res["assign"]
            vecs_t_d = res["vecs_t_d"]
            transformed_vecs = res["vecs_t"]
            if vecs_t_d == 0:
                logging.warning(f"add failed")
                return None

            # broadcast the transformed vectors to all the rest servers
            futures = []
            for j in range(index_worker_count):
                if j == preferred_index_worker:
                    continue
                self.pool.submit(
                    self.index_worker_groups[i][j].add_ext,
                    len(index_worker_vec_batches[i]),
                    vecs_t_d,
                    np.array(transformed_vecs),
                    index_worker_id_batches[i],
                    assign,
                )
            index_worker_group_round2_futures.append(futures)

        # wait for all requests to finish and collect results
        results = []
        for f in refine_worker_futures:
            if f is not None:
                results.append(f.result())
        for fs in index_worker_group_round2_futures:
            for f in fs:
                if f is not None:
                    results.append(f.result())
        return results

    def search(
        self,
        query: np.ndarray,
        k: int,
        nprobe: int,
        k_factor: int = 1,
        metric_type: int = 1,
    ):
        """
        Search vectors in the distributed HakesService
        """
        if len(query.shape) != 2:
            logging.warning(f"search failed: query shape {query.shape} != 2")
            return None

        # send request to the preferred server
        preferred_server = self.cfg.get_preferred_id()
        # do not filter invalid entries as we need the shape for rerank
        futures = [
            self.pool.submit(
                self.index_worker_groups[i][preferred_server].search,
                query,
                k,
                nprobe,
                k_factor,
                metric_type,
            )
            for i in range(self.cfg.index_worker_group_count)
        ]

        results = [f.result() for f in futures]
        print(f"results: {results}")
        ids = [[] for _ in range(query.shape[0])]
        scores = [[] for _ in range(query.shape[0])]
        for j in range(query.shape[0]):
            for i in range(self.cfg.index_worker_group_count):
                if results[i] is None:
                    continue
                ids[j].extend(results[i]["ids"][j])
                scores[j].extend(results[i]["scores"][j])
            cur_idx = np.array(ids[j])
            cur_score = np.array(scores[j])
            idx = np.argsort(cur_score if metric_type == 1 else -cur_score)
            ids[j] = cur_idx[idx]
            scores[j] = cur_score[idx]
            ids[j] = ids[j][: k * k_factor]
            scores[j] = scores[j][: k * k_factor]

        # rerank the results split the results into their target servers
        nq = query.shape[0]

        # build rerank requests input
        base_id_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        base_dist_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        k_base_count_batches = [
            [0 for _ in range(nq)] for _ in range(self.cfg.refine_worker_count)
        ]

        for i in range(nq):
            for j in range(len(ids[i])):
                server_idx = self.cfg.get_refine_worker_id(ids[i][j])
                base_id_batches[server_idx].append(ids[i][j])
                base_dist_batches[server_idx].append(scores[i][j])
                k_base_count_batches[server_idx][i] += 1

        # send rerank request to each server
        futures = [
            self.pool.submit(
                self.refine_workers[i].rerank,
                query,
                k,
                k_base_count_batches[i],
                base_id_batches[i],
                base_dist_batches[i],
                metric_type,
            )
            for i in range(self.cfg.refine_worker_count)
        ]
        results = [future.result() for future in futures]

        collated_id_ll = [[] for _ in range(nq)]
        collated_dist_ll = [[] for _ in range(nq)]
        for i in range(self.cfg.refine_worker_count):
            if results[i] is None:
                continue
            for j in range(nq):
                collated_id_ll[j].extend(results[i]["ids"][j])
                collated_dist_ll[j].extend(results[i]["scores"][j])

        # sort the ids based on scores
        final_result = []
        for j in range(nq):
            sorted_idx = np.flip(np.argsort(collated_dist_ll[j]))
            collated_id_ll[j] = np.array(collated_id_ll[j])[sorted_idx]
            collated_dist_ll[j] = np.array(collated_dist_ll[j])[sorted_idx]
            final_result.append(
                {"ids": collated_id_ll[j][:k], "scores": collated_dist_ll[j][:k]}
            )
        return final_result

    def delete(self, n, ids):
        # issue delete to corresponding index worker group and refine worker
        id_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        for i in range(n):
            refine_worker_idx = self.cfg.get_refine_worker_id(ids[i])
            id_batches[refine_worker_idx].append(ids[i])

        # send requests to each server with the threadpool
        refine_worker_futures = [
            (
                self.pool.submit(
                    self.refine_workers[i].delete, len(id_batches[i]), id_batches[i]
                )
                if len(id_batches[i]) != 0
                else None
            )
            for i in range(self.cfg.refine_worker_count)
        ]

        id_batches = [[] for _ in range(self.cfg.index_worker_group_count)]
        for i in range(n):
            index_worker_group_idx = self.cfg.get_index_worker_group_id(ids[i])
            id_batches[index_worker_group_idx].append(ids[i])
        index_worker_futures = [
            (
                self.pool.submit(
                    client.delete,
                    len(id_batches[i]),
                    id_batches[i],
                )
                if len(id_batches[i]) != 0
                else None
            )
            for i in range(self.cfg.index_worker_group_count)
            for client in self.index_worker_groups[i]
        ]
        # wait for all requests to finish and collect results
        results = [f.result() for f in refine_worker_futures]
        for res in results:
            if res is None:
                return None
        results = [f.result() for f in index_worker_futures]
        for res in results:
            if res is None:
                return None
        return results

    def checkpoint(self):
        """
        Checkpoint the distributed HakesService V3
        """
        # send requests to each server with the threadpool
        refine_worker_futures = [
            self.pool.submit(client.checkpoint) for client in self.refine_workers
        ]
        index_worker_futures = [
            self.pool.submit(client.checkpoint)
            for group in self.index_worker_groups
            for client in group
        ]

        # wait for all requests to finish and collect results
        results = [f.result() for f in refine_worker_futures]
        for res in results:
            if res is None:
                return None
        results = [f.result() for f in index_worker_futures]
        return results

    def get_index(self) -> Tuple[int, HakesIndexParams]:
        preferred_server = self.cfg.get_preferred_id()
        # do not filter invalid entries as we need the shape for rerank
        result = self.index_worker_groups[0][preferred_server].get_index()
        if result is None:
            logging.warning(
                f"get_index failed on {self.index_worker_groups[0][preferred_server].url}"
            )
            return None
        if "index_version" not in result or "params" not in result:
            logging.warning(
                f"get_index failed on {self.index_worker_groups[0][preferred_server].url} ({result['msg']})"
            )
            return None
        return result["index_version"], result["params"]

    def update_index(self, params: HakesIndexParams) -> int:
        # send requests to each server with the threadpool
        futures = [
            self.pool.submit(client.update_index, params)
            for group in self.index_worker_groups
            for client in group
        ]

        # wait for all requests to finish and collect results
        results = [f.result() for f in futures]
        index_version = -1
        for res in results:
            if (
                res is None
                or "index_version" not in res
                or (res["index_version"] != index_version and index_version != -1)
            ):
                return -1
            else:
                index_version = res["index_version"]
        return index_version


class HakesClientPA(HakesClient):

    def __init__(self, cfg: ClientConfig):
        super().__init__(cfg)

    def connect(self):
        # fetch the index and build the partition allocation config
        self.index_version, index = self.get_index()
        self.vts = index.vts
        self.cfg = ClientConfigPA(
            self.cfg.index_worker_groups,
            self.cfg.refine_workers,
            index.vts,
            index.ivf,
            self.cfg.preference,
        )

    # generate only once for a Hakes deployment such that partition allocation among nodes are fixed
    def init_and_export_pa_config(self, path=None):
        _, index = self.get_index()
        # use the fetched index to build the sharding config
        cfg = ClientConfigPA(
            self.cfg.index_worker_groups,
            self.cfg.refine_workers,
            index.vts,
            index.ivf,
            self.cfg.preference,
        )
        if path is not None:
            cfg.save(path)
        self.cfg = cfg
        self.vts = index.vts
        return cfg

    def load_pa_config(self, path):
        cfg = ClientConfigPA.load(self.cfg, path)
        self.cfg = cfg
        self.vts = cfg.vts
        return cfg

    def add(self, n, d, vecs, ids):
        # find ivf allocations
        # vecs_t, ivf_assign = self.index.get_partition(vecs)
        if len(vecs.shape) == 1:
            vecs = vecs.reshape(1, -1)
        ivf_assign = self.cfg.get_ivf_assign(vecs)
        vecs_t = self.vts.apply(vecs)

        # build batches based on partitions
        vec_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        id_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        assign_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        for i in range(n):
            refine_worker_idx = self.cfg.get_refine_worker_id(ivf_assign[i])
            vec_batches[refine_worker_idx].append(vecs[i])
            id_batches[refine_worker_idx].append(ids[i])
            assign_batches[refine_worker_idx].append(ivf_assign[i])

        # send requests to each server with the threadpool
        refine_worker_futures = [
            (
                self.pool.submit(
                    self.refine_workers[i].add_ext,
                    len(vec_batches[i]),
                    d,
                    vec_batches[i],
                    id_batches[i],
                    assign_batches[i],
                    add_to_refine_only=True,
                )
                if len(vec_batches[i]) != 0
                else None
            )
            for i in range(self.cfg.refine_worker_count)
        ]

        index_worker_vec_batches = [
            [] for _ in range(self.cfg.index_worker_group_count)
        ]
        index_worker_id_batches = [[] for _ in range(self.cfg.index_worker_group_count)]
        index_worker_assign_batches = [
            [] for _ in range(self.cfg.index_worker_group_count)
        ]
        for i in range(n):
            index_worker_group_idx = self.cfg.get_index_worker_group_id(ids[i])
            index_worker_vec_batches[index_worker_group_idx].append(vecs_t[i])
            index_worker_id_batches[index_worker_group_idx].append(ids[i])
            index_worker_assign_batches[index_worker_group_idx].append(ivf_assign[i])

        preferred_index_worker = self.cfg.get_preferred_id()
        index_worker_group_futures = []
        for i in range(self.cfg.index_worker_group_count):
            index_worker_count = len(self.index_worker_groups[i])
            futures = [
                (
                    self.pool.submit(
                        self.index_worker_groups[i][j].add_ext,
                        len(index_worker_vec_batches[i]),
                        vecs_t.shape[1],
                        index_worker_vec_batches[i],
                        index_worker_id_batches[i],
                        index_worker_assign_batches[i],
                    )
                    if len(index_worker_vec_batches[i]) != 0
                    else None
                )
                for j in range(index_worker_count)
            ]
            index_worker_group_futures.append(futures)

        # wait for all futures to finish
        results = []
        for f in refine_worker_futures:
            if f is not None:
                results.append(f.result())
        for fs in index_worker_group_futures:
            for f in fs:
                if f is not None:
                    results.append(f.result())
        return results

    def search(
        self,
        query: np.ndarray,
        k: int,
        nprobe: int,
        k_factor: int = 1,
        metric_type: int = 1,
    ):
        if len(query.shape) != 2:
            logging.warning(f"search failed: query shape {query.shape} != 2")
            return None

        preferred_server = self.cfg.get_preferred_id()
        futures = [
            self.pool.submit(
                self.index_worker_groups[i][preferred_server].search,
                query,
                k,
                nprobe,
                k_factor,
                metric_type,
                require_pa=True,
            )
            for i in range(self.cfg.index_worker_group_count)
        ]
        results = [f.result() for f in futures]
        print(f"results: {results}")

        ids = [[] for _ in range(query.shape[0])]
        scores = [[] for _ in range(query.shape[0])]
        pas = [[] for _ in range(query.shape[0])]
        for j in range(query.shape[0]):
            for i in range(self.cfg.index_worker_group_count):
                if results[i] is None:
                    continue
                ids[j].extend(results[i]["ids"][j])
                scores[j].extend(results[i]["scores"][j])
                pas[j].extend(results[i]["pas"][j])
            cur_idx = np.array(ids[j])
            cur_score = np.array(scores[j])
            cur_pas = np.array(pas[j])
            print(cur_score)
            print(cur_pas)
            idx = np.argsort(cur_score if metric_type == 1 else -cur_score)
            ids[j] = cur_idx[idx]
            scores[j] = cur_score[idx]
            pas[j] = cur_pas[idx]
            print(pas[j])
            ids[j] = ids[j][: k * k_factor]
            scores[j] = scores[j][: k * k_factor]
            pas[j] = pas[j][: k * k_factor]
        nq = query.shape[0]

        # build rerank requests input
        base_id_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        base_dist_batches = [[] for _ in range(self.cfg.refine_worker_count)]
        k_base_count_batches = [
            [0 for _ in range(nq)] for _ in range(self.cfg.refine_worker_count)
        ]

        print(f"ids: {ids}")
        print(f"scores: {scores}")
        print(f"pas: {pas}")

        for i in range(nq):
            for j in range(len(ids[i])):
                server_idx = self.cfg.get_refine_worker_id(pas[i][j])
                base_id_batches[server_idx].append(ids[i][j])
                base_dist_batches[server_idx].append(scores[i][j])
                k_base_count_batches[server_idx][i] += 1

        futures = [
            self.pool.submit(
                self.refine_workers[i].rerank,
                query,
                k,
                k_base_count_batches[i],
                base_id_batches[i],
                base_dist_batches[i],
                metric_type,
            )
            for i in range(self.cfg.refine_worker_count)
        ]
        results = [future.result() for future in futures]

        collated_id_ll = [[] for _ in range(nq)]
        collated_dist_ll = [[] for _ in range(nq)]
        for i in range(self.cfg.refine_worker_count):
            if results[i] is None:
                continue
            for j in range(nq):
                collated_id_ll[j].extend(results[i]["ids"][j])
                collated_dist_ll[j].extend(results[i]["scores"][j])

        # sort the ids based on scores
        final_result = []
        for j in range(nq):
            sorted_idx = np.flip(np.argsort(collated_dist_ll[j]))
            collated_id_ll[j] = np.array(collated_id_ll[j])[sorted_idx]
            collated_dist_ll[j] = np.array(collated_dist_ll[j])[sorted_idx]
            final_result.append(
                {"ids": collated_id_ll[j][:k], "scores": collated_dist_ll[j][:k]}
            )
        return final_result

    def delete(self, n, ids):
        # broadcast to refineworker with the threadpool
        refine_worker_futures = [
            (
                self.pool.submit(
                    self.refine_workers[i].delete,
                    n,
                    ids,
                )
            )
            for i in range(self.cfg.refine_worker_count)
        ]

        id_batches = [[] for _ in range(self.cfg.index_worker_group_count)]
        for i in range(n):
            index_worker_group_idx = self.cfg.get_index_worker_group_id(ids[i])
            id_batches[index_worker_group_idx].append(ids[i])
        index_worker_futures = [
            (
                self.pool.submit(
                    client.delete,
                    len(id_batches[i]),
                    id_batches[i],
                )
                if len(id_batches[i]) != 0
                else None
            )
            for i in range(self.cfg.index_worker_group_count)
            for client in self.index_worker_groups[i]
        ]
        # wait for all requests to finish and collect results
        results = [f.result() for f in refine_worker_futures]
        for res in results:
            if res is None:
                return None
        results = [f.result() for f in index_worker_futures]
        for res in results:
            if res is None:
                return None
        return results
