import argparse
import copy
import numpy as np
import random
import sys
import time
import torch
import tqdm

from hakes.dataset import HakesDataset
from hakes.index import HakesIndex
from hakes.train import train_model
from hakes.dataset import load_data_bin, load_neighbors_bin

# control randomness
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# # add the project path to python path if not installed
# project_path = "."
# sys.path.append(project_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, help="Number of total data", required=True)
    parser.add_argument("--d", type=int, help="Dimension of vectors", required=True)
    parser.add_argument(
        "--train_n", type=int, help="Number of training data", required=True
    )
    parser.add_argument(
        "--train_neighbor_count",
        type=int,
        help="Number of neighbors per train vector in the file",
        required=True,
    )
    parser.add_argument(
        "--use_train_nn_count",
        type=int,
        help="Number of top neighbors to use per query out of train_neighbor_count in training",
        required=True,
    )
    parser.add_argument(
        "--NQ", type=int, help="Number of validation set data count", required=True
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Data path that should contains the overall data and training data (sampled data and their approximate neighbors)",
        required=True,
    )
    parser.add_argument(
        "--index_path",
        type=str,
        help="Path of base index, in which there should be a base_index directory that can be loaded and served by HAKES",
        required=True,
    )
    parser.add_argument(
        "--epoch", type=int, help="Number of epoch to train", required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training iteration",
        required=True,
    )
    parser.add_argument(
        "--vt_lr", type=float, help="Learning rate for IVF", required=True
    )
    parser.add_argument(
        "--pq_lr",
        type=float,
        help="Learning rate for product quantization",
        required=True,
    )
    parser.add_argument(
        "--lamb",
        type=float,
        help="Control the weight of vt loss, -1 to rescale against pq loss",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="The GPU device to use if any. (use cpu by default) example: cuda:0",
        required=True,
    )
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        help="Path to save the index trained per epoch",
        required=True,
    )
    args = parser.parse_args()
    return args


def run(
    N,
    d,
    train_n,
    train_neighbor_count,
    use_train_nn_count,
    NQ,
    data_path,
    index_path,
    epoch,
    batch_size,
    vt_lr,
    pq_lr,
    lamb,
    device,
    save_path_prefix,
):
    data = load_data_bin(f"{data_path}/train.bin", N, d)
    sampled_data = load_data_bin(f"{index_path}/sampleQuery_100000.bin", train_n, d)
    neighbor_data = load_neighbors_bin(
        f"{index_path}/sample100NN_100000.bin", train_n, train_neighbor_count
    )

    # prepare the training dataset
    train_dataset = HakesDataset(
        data,
        sampled_data,
        neighbor_data[:, 0:use_train_nn_count],
    )

    # load the index
    hakes_index = HakesIndex.load(f"{index_path}/base_index", "ip")
    fixed_assignment = True
    shared_vts = True
    hakes_index.set_fixed_assignment(fixed_assignment)  # fix the assignment
    hakes_index.set_share_vts(shared_vts)

    # load test data
    test_data_tensor = torch.tensor(load_data_bin(f"{data_path}/test.bin", NQ, d))
    test_neighbor_data = load_neighbors_bin(f"{data_path}/neighbors.bin", NQ, 100)
    test_neighbor_data_tensor = torch.tensor(
        data[test_neighbor_data[:, :use_train_nn_count]]
    )

    # load reference index
    reference_index = HakesIndex.load(f"{index_path}/base_index", "ip")
    reference_index.set_fixed_assignment(fixed_assignment)  # fix the assignment
    reference_index.set_share_vts(shared_vts)

    for e in tqdm.tqdm(range(epoch)):
        start_time = time.time()
        # train the model for an epoch
        train_model(
            model=hakes_index,
            dataset=train_dataset,
            epochs=1,
            batch_size=batch_size,
            lr_params={"vt": vt_lr, "pq": pq_lr, "ivf": 0, "ivf_vt": 0},
            loss_weight={
                "vt": lamb if lamb != -1 else "rescale",
                "pq": 1,
                "ivf": 0,
                "ivf_vt": "0",
            },
            temperature=1,
            loss_method="hakes",  # vt kl-div, pq kl-div loss with the fixed assignment
            device=device,
        )
        train_finish_time = time.time()

        # evaluate test kl-div
        hakes_index.to("cpu")
        scores = hakes_index.report_distance(
            test_data_tensor, test_neighbor_data_tensor, "pq"
        ).squeeze(1)
        raw_scores = hakes_index.report_distance(
            test_data_tensor, test_neighbor_data_tensor, "raw"
        ).squeeze(1)
        ref_scores = reference_index.report_distance(
            test_data_tensor, test_neighbor_data_tensor, "pq"
        ).squeeze(1)
        kl_div = hakes_index.kldiv_loss(torch.tensor(raw_scores), torch.tensor(scores))
        ref_kl_div = reference_index.kldiv_loss(
            torch.tensor(raw_scores), torch.tensor(ref_scores)
        )
        print(f"Epoch {e}, KL-div: {kl_div}, Ref KL-div: {ref_kl_div}")

        # copy the vt, ivf, pq to the reference index
        reference_index.vts = copy.deepcopy(hakes_index.vts)
        reference_index.pq = copy.deepcopy(hakes_index.pq)
        reference_index.ivf = copy.deepcopy(hakes_index.ivf)

        update_centroids_start = time.time()
        # update ivf centroids
        recenter_sample_rate = 0.1
        recenter_sample_count = int(N * recenter_sample_rate)
        recenter_indices = np.random.choice(N, recenter_sample_count, replace=False)
        recenter_data = data[recenter_indices]

        grouped_vectors = [[] for _ in range(hakes_index.ivf.nlist)]
        calc_assign_batch_size = 1024 * 10
        for i in tqdm.trange(0, recenter_sample_count, calc_assign_batch_size):
            data_batch = torch.tensor(
                recenter_data[
                    i : min(i + calc_assign_batch_size, recenter_sample_count)
                ]
            )
            if shared_vts:
                data_batch = hakes_index.vts(data_batch)
            assignment = hakes_index.ivf.get_assignment(data_batch)
            # assign to the corresponding groups
            for j in range(len(assignment)):
                grouped_vectors[assignment[j].item()].append(i + j)

        new_centroids = []
        for i in tqdm.trange(len(grouped_vectors)):
            vecs = recenter_data[grouped_vectors[i]]
            vecs_tensor = torch.tensor(vecs)
            vt_vecs = hakes_index.vts(vecs_tensor)
            rep = torch.mean(vt_vecs, dim=0)
            normalized_rep = rep / torch.norm(rep)
            new_centroids.append(normalized_rep.detach().cpu().numpy())

        reference_index.ivf.update_centers(np.array(new_centroids))
        update_centroids_finish = time.time()

        # save the index
        save_path = f"{save_path_prefix}/epoch-{e}"
        reference_index.save_as_hakes_index(f"{save_path}/base_index")
        with open(f"{save_path}/train.txt", "w") as f:
            f.write(f"index_path: {index_path}\n")
            f.write(f"epoch {e}\n")
            f.write(f"vt_lr: {vt_lr}\n")
            f.write(f"pq_lr: {pq_lr}\n")
            f.write(f"lamb: {lamb}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(
                f"train_n: {train_n}, train_neighbor_count: {train_neighbor_count}, use_train_nn_count: {use_train_nn_count}\n"
            )
            f.write(f"KL-div: {kl_div}, Ref KL-div: {ref_kl_div}\n")
            f.write(f"recenter sample rate {recenter_sample_rate}\n")
            f.write(f"train time: {train_finish_time - start_time}\n")
            f.write(
                f"centroids update time: {update_centroids_finish - update_centroids_start}\n"
            )
            f.write(f"Total time: {time.time() - start_time}\n")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run(
        args.N,
        args.d,
        args.train_n,
        args.train_neighbor_count,
        args.use_train_nn_count,
        args.NQ,
        args.data_path,
        args.index_path,
        args.epoch,
        args.batch_size,
        args.vt_lr,
        args.pq_lr,
        args.lamb,
        args.device,
        args.save_path_prefix,
    )
