import os

os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"

import functools as ft
import itertools as it
from queue import Empty
from typing import Any, Dict

import ray
import typer
from ray import tune as rt
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch import multiprocessing as mp

from training import _launch_mp

app = typer.Typer()


def _handle_distributed(config: Dict[str, Any], use_tune: bool = False):
    import torch_xla.distributed.xla_multiprocessing as xmp

    mp.set_start_method("spawn")
    queue = mp.Queue()
    context = xmp.spawn(_launch_mp, args=(config, 8, queue), nprocs=8, join=False)

    try:
        while True:
            if not all(p.is_alive() for p in context.processes):
                break

            try:
                res = queue.get(block=True, timeout=1)
            except Empty:
                pass
            else:
                if isinstance(res, dict):
                    print(f"Got result: {res}")
                    if use_tune:
                        rt.report(**res)
                else:
                    raise NotImplementedError(f"No idea how to handle {res}")

    finally:
        queue.close()
        context.join()


def run_train(config: Dict[str, Any], distributed: bool = True, use_tune: bool = False):
    if distributed:
        _handle_distributed(config, use_tune=use_tune)
    else:
        _launch_mp(0, config)


@app.command()
def train(not_distributed: bool = False):
    run_train({}, distributed=not not_distributed)


@app.command()
def tune(address: str = None, not_distributed: bool = False):
    resources = {"tpu": 1} if address is None else None
    ray.init(
        address=address,
        resources=resources,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": "/home/ubuntu/transformers-mup",
                "XRT_TPU_CONFIG": "localservice;0;localhost:51011",
            }
        },
    )
    print(resources)
    search_space = {
        "learning_rate": rt.loguniform(1e-6, 1e-2),
        "warmup_ratio": rt.uniform(0, 0.1),
        "num_train_epochs": rt.uniform(20, 50),
    }
    train_fun = ft.partial(run_train, distributed=not not_distributed, use_tune=True)
    analysis = rt.run(
        train_fun,
        config=search_space,
        num_samples=-1,
        time_budget_s=60 * 60,
        log_to_file=True,
        resources_per_trial={"cpu": 96, "custom_resources": {"TPU": 1}},
        search_alg=HyperOptSearch(metric="eval_loss", mode="min"),
        scheduler=ASHAScheduler(metric="eval_loss", mode="min"),
        local_dir="/tmp",
    )


if __name__ == "__main__":
    app()
