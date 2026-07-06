from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model_samples import MongoSequenceSampleStore
from .training_data import (
    GlobalFeatureVectorizer,
    MongoWorldModelDataset,
    ShardedWorldModelDataset,
    collate_world_model_batch,
    infer_schema,
    load_shard_manifest,
)
from .world_model import WorldModelCVAE, WorldModelConfig, kl_divergence

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a self-supervised CS2 CVAE world model.")
    parser.add_argument("--config", default="config.train.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--shard-dir", help="Train from exported local .pt shards instead of MongoDB.")
    parser.add_argument(
        "--mongo-batch-size",
        type=int,
        help="How many Mongo sample documents to fetch per cursor batch.",
    )
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity/team.")
    parser.add_argument("--wandb-run-name", help="Weights & Biases run name.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], help="Weights & Biases mode.")
    parser.add_argument("--wandb-tag", action="append", default=None, help="Weights & Biases tag. Can be repeated.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _load_json(args.config)
    _apply_overrides(config, args)
    train(config)
    return 0


def train(config: dict[str, Any]) -> None:
    training_cfg = config["training"]
    seed = int(training_cfg.get("seed", 1337))
    random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(training_cfg.get("device", "auto"))
    log.info("Training on %s", device)

    event_label_names = list(config["dataset"].get("event_labels") or [])
    vectorizer = GlobalFeatureVectorizer()
    shard_dir = config["dataset"].get("shard_dir")
    store = None if shard_dir else MongoSequenceSampleStore(config["mongo"])
    meta = _load_training_meta(config, store)
    if not meta:
        raise RuntimeError(
            "No ml_meta document found in Mongo. Run the demo parser first so the training code can infer sample shapes."
        )
    schema = infer_schema(meta, event_label_names, vectorizer)
    model_cfg = WorldModelConfig(
        player_feature_dim=schema.player_feature_dim,
        target_feature_dim=schema.target_feature_dim,
        global_feature_dim=schema.global_feature_dim,
        future_steps=schema.future_steps,
        max_players=schema.max_players,
        event_label_dim=schema.event_label_dim,
        **config["model"],
    )
    model = WorldModelCVAE(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )

    train_loader = _build_loader(config, vectorizer, event_label_names, fold="train")
    val_loader = _build_loader(config, vectorizer, event_label_names, fold="val")
    train_total_batches, val_total_batches = _estimate_epoch_batches(config, store)
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints/world_model"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_val = float("inf")
    wandb_ctx = _init_wandb(config, model_cfg, schema, device, model)

    try:
        for epoch in range(1, int(training_cfg.get("epochs", 20)) + 1):
            train_metrics, global_step = _run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                training_cfg,
                global_step=global_step,
                train_mode=True,
                limit_batches=training_cfg.get("limit_train_batches"),
                total_batches=train_total_batches,
                wandb_ctx=wandb_ctx,
                epoch=epoch,
            )
            val_metrics, _ = _run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                training_cfg=training_cfg,
                global_step=global_step,
                train_mode=False,
                limit_batches=training_cfg.get("limit_val_batches"),
                total_batches=val_total_batches,
                wandb_ctx=None,
                epoch=epoch,
            )
            log.info("epoch=%s train=%s val=%s", epoch, _fmt(train_metrics), _fmt(val_metrics))
            latest_path = checkpoint_dir / "latest.pt"
            _save_checkpoint(latest_path, model, optimizer, model_cfg, schema, vectorizer, config, epoch, global_step, val_metrics)
            val_loss = val_metrics.get("loss", train_metrics["loss"])
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                _save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    model_cfg,
                    schema,
                    vectorizer,
                    config,
                    epoch,
                    global_step,
                    val_metrics,
                )
            _log_wandb_epoch(
                wandb_ctx,
                epoch=epoch,
                global_step=global_step,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
                best_val=best_val,
            )
            if bool((config.get("wandb") or {}).get("log_checkpoints", False)):
                _save_wandb_file(wandb_ctx, latest_path)
                if is_best:
                    _save_wandb_file(wandb_ctx, checkpoint_dir / "best.pt")
    finally:
        _finish_wandb(wandb_ctx)


def _build_loader(
    config: dict[str, Any],
    vectorizer: GlobalFeatureVectorizer,
    event_label_names: list[str],
    *,
    fold: str,
) -> DataLoader:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    num_workers = int(training_cfg.get("num_workers", 0))
    shard_dir = dataset_cfg.get("shard_dir")
    if num_workers > 0 and not shard_dir:
        log.warning(
            "num_workers=%s with Mongo IterableDataset can duplicate Mongo reads; use 0 unless the dataset reader is partitioned.",
            num_workers,
        )
    if shard_dir:
        dataset = ShardedWorldModelDataset(
            shard_dir,
            fold=fold,
            shuffle_shards=bool(dataset_cfg.get("shuffle_shards", fold == "train")),
            shuffle_samples=bool(dataset_cfg.get("shuffle_samples", fold == "train")),
            seed=int(training_cfg.get("seed", 1337)),
        )
    else:
        dataset = MongoWorldModelDataset(
            config["mongo"],
            split=dataset_cfg.get("split", "train"),
            fold=fold,
            val_fraction=float(dataset_cfg.get("val_fraction", 0.1)),
            event_label_names=event_label_names,
            vectorizer=vectorizer,
            mongo_batch_size=int(dataset_cfg.get("mongo_batch_size", 512)),
        )
    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 64)),
        num_workers=num_workers,
        collate_fn=collate_world_model_batch,
    )


def _run_epoch(
    model: WorldModelCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training_cfg: dict[str, Any],
    *,
    global_step: int,
    train_mode: bool,
    limit_batches: int | None,
    total_batches: int | None = None,
    wandb_ctx: dict[str, Any] | None = None,
    epoch: int | None = None,
) -> tuple[dict[str, float], int]:
    model.train(train_mode)
    totals: dict[str, float] = {}
    count = 0
    progress = tqdm(loader, desc="train" if train_mode else "val", leave=False, total=total_batches)
    iterator = iter(progress)
    next_start = time.perf_counter()
    batch_idx = 0
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_wait_seconds = time.perf_counter() - next_start
        batch_idx += 1
        if limit_batches and batch_idx > int(limit_batches):
            break
        step_start = time.perf_counter()
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        if train_mode:
            global_step += 1
        with torch.set_grad_enabled(train_mode):
            output = model(
                batch["history"],
                batch["history_mask"],
                batch["global_features"],
                future=batch["future"],
                future_mask=batch["future_mask"],
                sample=train_mode,
            )
            losses = compute_losses(output, batch, training_cfg, global_step)
            if train_mode and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("grad_clip_norm", 1.0)))
                optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
        step_seconds = time.perf_counter() - step_start
        totals["data_wait_seconds"] = totals.get("data_wait_seconds", 0.0) + data_wait_seconds
        totals["step_seconds"] = totals.get("step_seconds", 0.0) + step_seconds
        if train_mode and batch_idx % int(training_cfg.get("log_every", 50)) == 0:
            loss_values = {key: float(value.detach().cpu()) for key, value in losses.items()}
            progress.set_postfix(
                loss=loss_values["loss"],
                data=f"{data_wait_seconds:.3f}s",
                step=f"{step_seconds:.3f}s",
            )
            _log_wandb_step(
                wandb_ctx,
                global_step=global_step,
                epoch=epoch,
                batch_idx=batch_idx,
                metrics=loss_values,
                data_wait_seconds=data_wait_seconds,
                step_seconds=step_seconds,
                learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
            )
        next_start = time.perf_counter()
    if count == 0:
        return {"loss": float("inf")}, global_step
    return {key: value / count for key, value in totals.items()}, global_step


def _load_training_meta(config: dict[str, Any], store: MongoSequenceSampleStore | None) -> dict[str, Any]:
    shard_dir = config["dataset"].get("shard_dir")
    if shard_dir:
        manifest = load_shard_manifest(shard_dir)
        return manifest.get("dataset", {}).get("meta") or {}
    if store is None:
        return {}
    return store.get_meta() or {}


def _estimate_epoch_batches(config: dict[str, Any], store: MongoSequenceSampleStore | None) -> tuple[int | None, int | None]:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    val_fraction = max(0.0, min(1.0, float(dataset_cfg.get("val_fraction", 0.1))))
    batch_size = max(1, int(training_cfg.get("batch_size", 64)))
    shard_dir = dataset_cfg.get("shard_dir")
    if shard_dir:
        try:
            manifest = load_shard_manifest(shard_dir)
            counts = manifest.get("counts") or {}
            train_samples = int(counts.get("train", 0))
            val_samples = int(counts.get("val", 0))
            total_samples = train_samples + val_samples
        except Exception as exc:
            log.warning("Could not estimate epoch length from shard manifest: %s", exc)
            return None, None
    else:
        if store is None:
            return None, None
        try:
            total_samples = store.count_samples(split=dataset_cfg.get("split", "train"))
        except Exception as exc:
            log.warning("Could not estimate epoch length from Mongo: %s", exc)
            return None, None
        train_samples = int(total_samples * (1.0 - val_fraction))
        val_samples = total_samples - train_samples
    train_batches = math.ceil(train_samples / batch_size)
    val_batches = math.ceil(val_samples / batch_size)
    if training_cfg.get("limit_train_batches") is not None:
        train_batches = min(train_batches, int(training_cfg["limit_train_batches"]))
    if training_cfg.get("limit_val_batches") is not None:
        val_batches = min(val_batches, int(training_cfg["limit_val_batches"]))
    log.info(
        "Estimated epoch size: samples=%s train_batches~=%s val_batches~=%s batch_size=%s val_fraction=%.3f",
        total_samples,
        train_batches,
        val_batches,
        batch_size,
        val_fraction,
    )
    return train_batches, val_batches


def compute_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    training_cfg: dict[str, Any],
    global_step: int,
) -> dict[str, torch.Tensor]:
    future_mask = batch["future_mask"].unsqueeze(-1)
    pos_sq = (output["future_pos"] - batch["future"]).pow(2) * future_mask
    position_loss = pos_sq.sum() / (future_mask.sum() * batch["future"].shape[-1]).clamp_min(1.0)

    alive_target = batch["future_alive"].squeeze(-1)
    alive_loss_raw = F.binary_cross_entropy_with_logits(
        output["future_alive_logits"],
        alive_target,
        reduction="none",
    )
    alive_loss = (alive_loss_raw * batch["future_mask"]).sum() / batch["future_mask"].sum().clamp_min(1.0)

    event_loss = F.binary_cross_entropy_with_logits(output["event_logits"], batch["event_labels"])
    winner_loss_raw = F.cross_entropy(output["winner_logits"], batch["winner_label"], reduction="none")
    winner_loss = (winner_loss_raw * batch["winner_known"]).sum() / batch["winner_known"].sum().clamp_min(1.0)
    kl = kl_divergence(
        output["posterior_mu"],
        output["posterior_logvar"],
        output["prior_mu"],
        output["prior_logvar"],
    )
    beta = _kl_beta(training_cfg, global_step)
    loss = (
        float(training_cfg.get("position_loss_weight", 1.0)) * position_loss
        + float(training_cfg.get("alive_loss_weight", 0.2)) * alive_loss
        + float(training_cfg.get("event_loss_weight", 0.1)) * event_loss
        + float(training_cfg.get("winner_loss_weight", 0.1)) * winner_loss
        + beta * kl
    )
    return {
        "loss": loss,
        "position": position_loss,
        "alive": alive_loss,
        "event": event_loss,
        "winner": winner_loss,
        "kl": kl,
        "kl_beta": torch.tensor(beta, device=loss.device),
    }


def _kl_beta(training_cfg: dict[str, Any], global_step: int) -> float:
    beta_max = float(training_cfg.get("kl_beta_max", 0.1))
    warmup = max(1, int(training_cfg.get("kl_warmup_steps", 2000)))
    return beta_max * min(1.0, global_step / warmup)


def _save_checkpoint(
    path: Path,
    model: WorldModelCVAE,
    optimizer: torch.optim.Optimizer,
    model_cfg: WorldModelConfig,
    schema: Any,
    vectorizer: GlobalFeatureVectorizer,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_cfg.to_dict(),
            "schema": schema.__dict__,
            "global_vectorizer": {
                "numeric_keys": vectorizer.numeric_keys,
                "event_keys": vectorizer.event_keys,
                "names": vectorizer.names,
            },
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def _init_wandb(
    config: dict[str, Any],
    model_cfg: WorldModelConfig,
    schema: Any,
    device: torch.device,
    model: WorldModelCVAE,
) -> dict[str, Any] | None:
    wandb_cfg = config.get("wandb") or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging is enabled, but `wandb` is not installed. "
            "Install it with `.venv/bin/python -m pip install wandb` or disable W&B."
        ) from exc
    init_kwargs: dict[str, Any] = {
        "project": wandb_cfg.get("project", "cs2-demo-world-model"),
        "config": {
            "train_config": config,
            "model_config": model_cfg.to_dict(),
            "schema": schema.__dict__,
            "device": str(device),
        },
    }
    if wandb_cfg.get("entity"):
        init_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg.get("name"):
        init_kwargs["name"] = wandb_cfg["name"]
    if wandb_cfg.get("mode"):
        init_kwargs["mode"] = wandb_cfg["mode"]
    if wandb_cfg.get("tags"):
        init_kwargs["tags"] = list(wandb_cfg["tags"])
    run = wandb.init(**init_kwargs)
    if bool(wandb_cfg.get("watch_model", False)):
        wandb.watch(model, log=wandb_cfg.get("watch_log", "gradients"), log_freq=int(wandb_cfg.get("watch_log_freq", 100)))
    return {"wandb": wandb, "run": run}


def _log_wandb_epoch(
    ctx: dict[str, Any] | None,
    *,
    epoch: int,
    global_step: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    learning_rate: float,
    best_val: float,
) -> None:
    if ctx is None:
        return
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "learning_rate": learning_rate,
        "best_val/loss": best_val,
    }
    payload.update({f"train/{key}": value for key, value in train_metrics.items()})
    payload.update({f"val/{key}": value for key, value in val_metrics.items()})
    ctx["wandb"].log(payload, step=global_step)


def _log_wandb_step(
    ctx: dict[str, Any] | None,
    *,
    global_step: int,
    epoch: int | None,
    batch_idx: int,
    metrics: dict[str, float],
    data_wait_seconds: float,
    step_seconds: float,
    learning_rate: float,
) -> None:
    if ctx is None:
        return
    payload = {
        "epoch": epoch,
        "train/batch_idx": batch_idx,
        "train/data_wait_seconds": data_wait_seconds,
        "train/step_seconds": step_seconds,
        "learning_rate": learning_rate,
    }
    payload.update({f"train/{key}": value for key, value in metrics.items()})
    ctx["wandb"].log(payload, step=global_step)


def _save_wandb_file(ctx: dict[str, Any] | None, path: Path) -> None:
    if ctx is None or not path.exists():
        return
    ctx["wandb"].save(str(path))


def _finish_wandb(ctx: dict[str, Any] | None) -> None:
    if ctx is None:
        return
    ctx["wandb"].finish()


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.device:
        config["training"]["device"] = args.device
    if args.checkpoint_dir:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir
    if args.shard_dir:
        config["dataset"]["shard_dir"] = args.shard_dir
    if args.mongo_batch_size is not None:
        config["dataset"]["mongo_batch_size"] = args.mongo_batch_size
    if args.limit_train_batches is not None:
        config["training"]["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        config["training"]["limit_val_batches"] = args.limit_val_batches
    if args.wandb:
        config.setdefault("wandb", {})["enabled"] = True
    if args.wandb_project:
        config.setdefault("wandb", {})["project"] = args.wandb_project
    if args.wandb_entity:
        config.setdefault("wandb", {})["entity"] = args.wandb_entity
    if args.wandb_run_name:
        config.setdefault("wandb", {})["name"] = args.wandb_run_name
    if args.wandb_mode:
        config.setdefault("wandb", {})["mode"] = args.wandb_mode
    if args.wandb_tag:
        config.setdefault("wandb", {})["tags"] = args.wandb_tag


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fmt(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))


if __name__ == "__main__":
    raise SystemExit(main())
