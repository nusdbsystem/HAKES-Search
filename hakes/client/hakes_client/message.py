"""
  This file contains the message format for the client and server to communicate.
  ref: faiss-ex/serving/message
"""

import numpy as np
from typing import Dict

from hakes_client.hakesindex import HakesIndexParams


def prepare_add_request(n: int, d: int, vecs: np.ndarray, ids: np.ndarray):
    data = {
        "n": n,
        "d": d,
        "vecs": np.ascontiguousarray(vecs, dtype="<f").tobytes().hex(),
        "ids": np.ascontiguousarray(ids, dtype="<q").tobytes().hex(),
    }
    return data


def prepare_extended_add_request(
    n: int,
    d: int,
    vecs: np.ndarray,
    ids: np.ndarray,
    assign: np.ndarray = None,
    add_to_refine_only: bool = False,
):
    data = {
        "n": n,
        "d": d,
        "vecs": np.ascontiguousarray(vecs, dtype="<f").tobytes().hex(),
        "ids": np.ascontiguousarray(ids, dtype="<q").tobytes().hex(),
    }
    data["assigned"] = False
    data["add_to_refine_only"] = add_to_refine_only
    if assign is not None:
        data["assigned"] = True
        data["assign"] = np.ascontiguousarray(assign, dtype="<q").tobytes().hex()
    return data


def parse_extended_add_response(resp: Dict) -> Dict:
    n = resp["n"]
    if "assign" in resp:
        assign = resp["assign"]
        assign = bytes.fromhex(assign)
        resp["assign"] = np.frombuffer(assign, dtype=np.int64).reshape(n, -1)
    if "vecs_t" in resp:
        vecs_t = resp["vecs_t"]
        vecs_t = bytes.fromhex(vecs_t)
        resp["vecs_t"] = np.frombuffer(vecs_t, dtype=np.float32).reshape(n, -1)
    return resp


# L2: 0, IP: 1
def prepare_search_request(
    n: int,
    d: int,
    vecs: np.ndarray,
    k: int,
    nprobe: int,
    k_factor: int,
    metric_type: int,
    requrie_pa: bool = False,
):
    # flatten the vecs
    vecs = [x for vec in vecs for x in vec]
    data = {
        "n": n,
        "d": d,
        "vecs": np.ascontiguousarray(vecs, dtype="<f").tobytes().hex(),
        "k": k,
        "nprobe": nprobe,
        "k_factor": k_factor,
        "metric_type": metric_type,
        "require_pa": requrie_pa,
    }
    return data


def parse_search_response(resp: Dict, filter_invalid: bool = True) -> Dict:
    n = resp["n"]
    k = resp["k"]
    if "ids" in resp and "scores" in resp:
        ids = resp["ids"]
        ids = bytes.fromhex(ids)
        ids = np.frombuffer(ids, dtype=np.int64).reshape(n, k)
        scores = resp["scores"]
        scores = bytes.fromhex(scores)
        scores = np.frombuffer(scores, dtype=np.float32).reshape(n, k)
        if not filter_invalid:
            resp["ids"] = ids.tolist()
            resp["scores"] = scores.tolist()
            return resp
        valid_ids = [[] for _ in range(n)]
        valid_scores = [[] for _ in range(n)]
        for i in range(n):
            for j in range(k):
                if ids[i][j] != -1:
                    valid_ids[i].append(ids[i][j])
                    valid_scores[i].append(scores[i][j])

        resp["ids"] = valid_ids
        resp["scores"] = valid_scores
    if "require_pa" in resp and resp["require_pa"]:
        pa = resp["pas"]
        pa = bytes.fromhex(pa)
        pa = np.frombuffer(pa, dtype=np.int64).reshape(n, -1)
        resp["pas"] = pa
    return resp


def prepare_rerank_request(
    n: int,
    d: int,
    vecs: np.ndarray,
    k: int,
    metric_type: int,
    k_base_count: np.ndarray,
    base_labels: np.ndarray,
    base_distances: np.ndarray,
):
    # flatten the vecs
    vecs = [x for vec in vecs for x in vec]
    data = {
        "n": n,
        "d": d,
        "vecs": np.ascontiguousarray(vecs, dtype="<f").tobytes().hex(),
        "k": k,
        "metric_type": metric_type,
        "k_base_count": np.ascontiguousarray(k_base_count, dtype="<q").tobytes().hex(),
        "base_labels": np.ascontiguousarray(base_labels, dtype="<q").tobytes().hex(),
        "base_distances": np.ascontiguousarray(base_distances, dtype="<f")
        .tobytes()
        .hex(),
    }
    return data


def parse_get_index_response(resp: Dict) -> Dict:
    status = resp["status"]
    if status == False:
        return {"msg": resp["msg"]}

    return {
        "index_version": resp["index_version"],
        "params": HakesIndexParams.deserialize(bytes.fromhex(resp["params"])),
    }


def prepare_update_index_request(params: HakesIndexParams):
    data = {"params": params.serialize().hex()}
    return data


def parse_update_index_response(resp: Dict) -> Dict:
    status = resp["status"]
    if status == False:
        return {"msg": resp["msg"]}
    else:
        return {"index_version": resp["index_version"]}
