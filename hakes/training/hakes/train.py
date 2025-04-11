import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.optim.adamw import AdamW
from tqdm import trange

from typing import Dict

from hakes.index import HakesIndex
from hakes.dataset import HakesDataset, HakesDataCollator

"""
  HAKES training loops through the sampled vector - ANN pairs to update index parameters.
"""


def train_model(
    model: HakesIndex,
    dataset: HakesDataset,
    epochs: int = 10,
    batch_size: int = 512,
    warmup_steps_ratio: float = 0.1,
    lr_params: Dict[str, float] = {"vt": 1e-4, "pq": 1e-4, "ivf": 1e-4},
    loss_weight: Dict[str, float] = {"vt": 0.0, "pq": 1.0, "ivf": 1.0},
    temperature: float = 1.0,
    loss_method: str = "hakes",
    max_grad_norm: float = -1,
    device="cpu",
    checkpoint_path="./checkpoint",
):
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, collate_fn=HakesDataCollator()
    )
    model.to(device)
    num_train_steps = len(dataloader) * epochs

    # setup optimizer
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if "vts" in n and "ivf" not in n],
            "weight_decay": 0.999,
            "lr": lr_params["vt"],
        },
        {
            "params": [p for n, p in param_optimizer if "ivf_vts" in n],
            "weight_decay": 0.999,
            "lr": lr_params["ivf_vt"] if "ivf_vt" in lr_params else 0,
        },
        {
            "params": [p for n, p in param_optimizer if "ivf" in n and "vts" not in n],
            "weight_decay": 0.999,
            "lr": lr_params["ivf"],
        },
        {
            "params": [p for n, p in param_optimizer if "pq.codebooks" in n],
            "weight_decay": 0.999,
            "lr": lr_params["pq"],
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    # setup learning rate scheduler
    num_warmup_steps = int(num_train_steps * warmup_steps_ratio)
    lr_lambda = lambda step: (
        float(step) / float(max(1, num_warmup_steps))
        if step < num_warmup_steps
        else max(
            0.0,
            float(num_train_steps - step)
            / float(max(1, num_train_steps - num_warmup_steps)),
        )
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # train
    global_step = 0
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        for step, sample in enumerate(dataloader):
            # to update, we are not using dict
            query_data = sample["query_data"].to(device)
            pos_data = sample["pos_data"].to(device)
            # check the effect of fix_emb='doc', cross_device_sample=True
            batch_vt_loss, batch_pq_loss, batch_ivf_vt_loss, batch_ivf_loss = model(
                query_data=query_data,
                pos_data=pos_data,
                temperature=temperature,
                loss_method=loss_method,
            )
            vt_rescale = (
                (
                    (batch_pq_loss / batch_vt_loss).item() * loss_weight["pq"]
                    if batch_vt_loss != 0.0
                    else 0
                )
                if loss_weight["vt"] == "rescale"
                else float(loss_weight["vt"])
            )
            ivf_vt_rescale = (
                (
                    (batch_ivf_loss / batch_ivf_vt_loss).item() * loss_weight["ivf"]
                    if batch_ivf_vt_loss != 0.0
                    else 0
                )
                if loss_weight["ivf_vt"] == "rescale"
                else float(loss_weight["ivf_vt"])
            )

            # print(f"vt_rescale: {vt_rescale}, ivf_vt_rescale: {ivf_vt_rescale}")
            # print(f"vt loss: {batch_vt_loss}, pq loss: {batch_pq_loss}, ivf_vt loss: {batch_ivf_vt_loss}, ivf loss: {batch_ivf_loss}")
            # print(f"vt_rescale: {type(vt_rescale)}, ivf_vt_rescale: {type(ivf_vt_rescale)} {type(loss_weight['pq'])}, {type(loss_weight['ivf'])}")
            # print(f"vt loss: {type(batch_vt_loss)}, pq loss: {type(batch_pq_loss)}, ivf_vt loss: {type(batch_ivf_vt_loss)}, ivf loss: {type(batch_ivf_loss)}")

            batch_loss = (
                vt_rescale * batch_vt_loss
                + loss_weight["pq"] * batch_pq_loss
                + ivf_vt_rescale * batch_ivf_vt_loss
                + loss_weight["ivf"] * batch_ivf_loss
            )
            batch_loss.backward()
            print(
                f"batch loss = {vt_rescale} * {batch_vt_loss} + {loss_weight['pq']} * {batch_pq_loss} + {ivf_vt_rescale} * {batch_ivf_vt_loss} + {loss_weight['ivf']} * {batch_ivf_loss}"
            )
            print(f"epoch {epoch} step {step} batch_loss: {batch_loss}")

            if max_grad_norm != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            if model.metric == "ip":
                model.ivf.normalize_centers()

            global_step += 1
            # if global_step % logging_steps == 0:
            #     step_num = logging_steps
            #     logging.info(
            #         f"Step {step_num}: loss {loss / step_num:.4f}, vt_loss {vt_loss / step_num:.4f}, ivf_loss {ivf_loss / step_num:.4f}, pq_loss {pq_loss / step_num:.4f}"
            #     )
            #     loss, vt_loss, ivf_loss, pq_loss = 0.0, 0.0, 0.0, 0.0
            #     ckpt_path = os.path.join(
            #         checkpoint_path, f"epoch_{epoch}_step_{global_step}"
            #     )
            #     os.makedirs(ckpt_path, exist_ok=True)
            #     model.save(ckpt_path)
            #     logging.info(f"Checkpoint saved at {ckpt_path}")
            # return  # early termination for testing
