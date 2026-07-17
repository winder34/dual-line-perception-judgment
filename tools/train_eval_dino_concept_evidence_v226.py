#!/usr/bin/env python3
"""Train v226a Parent/Fine evidence projectors and audit frozen TEST once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.dual_line.representation import (
    ConceptLayout,
    DinoConceptEvidenceProjector,
    symmetric_kl,
    true_class_attention_concentration,
)


def _load(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    return {key: z[key] for key in z.files}


def _split(labels: np.ndarray, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = int(round(len(indices) * ratio))
        val.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    rng.shuffle(train)
    rng.shuffle(val)
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def _random_valid_mask(batch: int, tiles: int, drop_rate: float, device: torch.device) -> torch.Tensor:
    mask = torch.rand((batch, tiles), device=device) >= float(drop_rate)
    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty, torch.randint(tiles, (int(empty.sum()),), device=device)] = True
    return mask


def _loss(
    model: DinoConceptEvidenceProjector,
    cls_token: torch.Tensor,
    tile_tokens: torch.Tensor,
    fine_target: torch.Tensor,
    parent_target: torch.Tensor,
    *,
    drop_rate: float,
    concentration_limit: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean = model(cls_token, tile_tokens)
    mask = _random_valid_mask(len(fine_target), tile_tokens.shape[1], drop_rate, tile_tokens.device)
    masked = model(cls_token, tile_tokens, valid_tile_mask=mask)
    class_loss = F.cross_entropy(clean.parent_logits, parent_target) + F.cross_entropy(clean.fine_logits, fine_target)
    tile_loss = F.cross_entropy(clean.parent_tile_pooled_logits, parent_target) + F.cross_entropy(
        clean.fine_tile_pooled_logits, fine_target
    )
    masked_loss = F.cross_entropy(masked.parent_tile_pooled_logits, parent_target) + F.cross_entropy(
        masked.fine_tile_pooled_logits, fine_target
    )
    consistency = symmetric_kl(clean.parent_tile_pooled_logits, masked.parent_tile_pooled_logits) + symmetric_kl(
        clean.fine_tile_pooled_logits, masked.fine_tile_pooled_logits
    )
    parent_conc = true_class_attention_concentration(clean.parent_spatial_attention, parent_target)
    fine_conc = true_class_attention_concentration(clean.fine_spatial_attention, fine_target)
    diversity = F.relu(parent_conc - concentration_limit).square() + F.relu(
        fine_conc - concentration_limit
    ).square()
    total = class_loss + 0.50 * tile_loss + 0.25 * masked_loss + 0.15 * consistency + 0.10 * diversity
    return total, {
        "class_loss": float(class_loss.detach()),
        "tile_loss": float(tile_loss.detach()),
        "masked_loss": float(masked_loss.detach()),
        "consistency": float(consistency.detach()),
        "parent_concentration": float(parent_conc.detach()),
        "fine_concentration": float(fine_conc.detach()),
    }


@torch.no_grad()
def _evaluate(
    model: DinoConceptEvidenceProjector,
    cache: dict[str, Any],
    indices: np.ndarray,
    layout: ConceptLayout,
    device: torch.device,
    batch_size: int,
    counterfactual: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    fine_all = torch.as_tensor(cache["label"], dtype=torch.long)
    parent_all = layout.parent_targets(fine_all)
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(
        torch.as_tensor(cache["cls_token"]).index_select(0, index_tensor),
        torch.as_tensor(cache["tile_tokens"]).index_select(0, index_tensor),
        fine_all.index_select(0, index_tensor),
        parent_all.index_select(0, index_tensor),
        index_tensor,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    records: dict[str, list[np.ndarray]] = {
        key: []
        for key in [
            "index",
            "fine_label",
            "parent_label",
            "fine_pred",
            "parent_pred",
            "fine_cls_pred",
            "parent_cls_pred",
            "fine_tile_pred",
            "parent_tile_pred",
            "fine_prob",
            "parent_prob",
            "fine_cls_prob",
            "parent_cls_prob",
            "fine_tile_prob",
            "parent_tile_prob",
            "fine_pred_attention",
            "parent_pred_attention",
            "fine_attention_concentration",
            "parent_attention_concentration",
            "fine_tile_evidence",
            "parent_tile_evidence",
            "fine_mask_flip_rate",
            "parent_mask_flip_rate",
            "fine_mask_prob_drop_mean",
            "parent_mask_prob_drop_mean",
            "fine_mask_prob_drop_max",
            "parent_mask_prob_drop_max",
        ]
    }
    for cls_token, tile_tokens, fine_target, parent_target, source_idx in loader:
        cls_token = cls_token.to(device=device, dtype=torch.float32)
        tile_tokens = tile_tokens.to(device=device, dtype=torch.float32)
        fine_target = fine_target.to(device)
        parent_target = parent_target.to(device)
        out = model(cls_token, tile_tokens)
        fine_prob = F.softmax(out.fine_logits, dim=-1)
        parent_prob = F.softmax(out.parent_logits, dim=-1)
        fine_cls_prob = F.softmax(out.fine_cls_logits, dim=-1)
        parent_cls_prob = F.softmax(out.parent_cls_logits, dim=-1)
        fine_tile_prob = F.softmax(out.fine_tile_pooled_logits, dim=-1)
        parent_tile_prob = F.softmax(out.parent_tile_pooled_logits, dim=-1)
        fine_pred = fine_prob.argmax(dim=-1)
        parent_pred = parent_prob.argmax(dim=-1)
        row = torch.arange(len(fine_pred), device=device)
        fine_attn = out.fine_spatial_attention[row, fine_pred]
        parent_attn = out.parent_spatial_attention[row, parent_pred]
        if counterfactual:
            tile_count = int(tile_tokens.shape[1])
            repeated_cls = cls_token.repeat_interleave(tile_count, dim=0)
            repeated_tiles = tile_tokens.repeat_interleave(tile_count, dim=0)
            masks = torch.ones(
                (len(fine_pred) * tile_count, tile_count),
                dtype=torch.bool,
                device=device,
            )
            hidden = torch.arange(tile_count, device=device).repeat(len(fine_pred))
            masks[torch.arange(len(masks), device=device), hidden] = False
            masked = model(repeated_cls, repeated_tiles, valid_tile_mask=masks)
            masked_fine_prob = F.softmax(masked.fine_logits, dim=-1).reshape(
                len(fine_pred), tile_count, -1
            )
            masked_parent_prob = F.softmax(masked.parent_logits, dim=-1).reshape(
                len(parent_pred), tile_count, -1
            )
            fine_selected = masked_fine_prob.gather(
                2,
                fine_pred[:, None, None].expand(-1, tile_count, 1),
            ).squeeze(-1)
            parent_selected = masked_parent_prob.gather(
                2,
                parent_pred[:, None, None].expand(-1, tile_count, 1),
            ).squeeze(-1)
            fine_drop = fine_prob[row, fine_pred][:, None] - fine_selected
            parent_drop = parent_prob[row, parent_pred][:, None] - parent_selected
            fine_flip = (masked_fine_prob.argmax(-1) != fine_pred[:, None]).float().mean(-1)
            parent_flip = (masked_parent_prob.argmax(-1) != parent_pred[:, None]).float().mean(-1)
            fine_drop_mean = fine_drop.mean(-1)
            parent_drop_mean = parent_drop.mean(-1)
            fine_drop_max = fine_drop.max(-1).values
            parent_drop_max = parent_drop.max(-1).values
        else:
            fine_flip = parent_flip = torch.zeros(len(fine_pred), device=device)
            fine_drop_mean = parent_drop_mean = torch.zeros(len(fine_pred), device=device)
            fine_drop_max = parent_drop_max = torch.zeros(len(fine_pred), device=device)
        values = {
            "index": source_idx.numpy(),
            "fine_label": fine_target.cpu().numpy(),
            "parent_label": parent_target.cpu().numpy(),
            "fine_pred": fine_pred.cpu().numpy(),
            "parent_pred": parent_pred.cpu().numpy(),
            "fine_cls_pred": out.fine_cls_logits.argmax(-1).cpu().numpy(),
            "parent_cls_pred": out.parent_cls_logits.argmax(-1).cpu().numpy(),
            "fine_tile_pred": out.fine_tile_pooled_logits.argmax(-1).cpu().numpy(),
            "parent_tile_pred": out.parent_tile_pooled_logits.argmax(-1).cpu().numpy(),
            "fine_prob": fine_prob.cpu().numpy(),
            "parent_prob": parent_prob.cpu().numpy(),
            "fine_cls_prob": fine_cls_prob.cpu().numpy(),
            "parent_cls_prob": parent_cls_prob.cpu().numpy(),
            "fine_tile_prob": fine_tile_prob.cpu().numpy(),
            "parent_tile_prob": parent_tile_prob.cpu().numpy(),
            "fine_pred_attention": fine_attn.cpu().numpy(),
            "parent_pred_attention": parent_attn.cpu().numpy(),
            "fine_attention_concentration": fine_attn.max(-1).values.cpu().numpy(),
            "parent_attention_concentration": parent_attn.max(-1).values.cpu().numpy(),
            "fine_tile_evidence": out.fine_tile_evidence.cpu().numpy().astype(np.float16),
            "parent_tile_evidence": out.parent_tile_evidence.cpu().numpy().astype(np.float16),
            "fine_mask_flip_rate": fine_flip.cpu().numpy(),
            "parent_mask_flip_rate": parent_flip.cpu().numpy(),
            "fine_mask_prob_drop_mean": fine_drop_mean.cpu().numpy(),
            "parent_mask_prob_drop_mean": parent_drop_mean.cpu().numpy(),
            "fine_mask_prob_drop_max": fine_drop_max.cpu().numpy(),
            "parent_mask_prob_drop_max": parent_drop_max.cpu().numpy(),
        }
        for key, value in values.items():
            records[key].append(np.asarray(value))
    merged = {key: np.concatenate(parts, axis=0) for key, parts in records.items()}
    order = np.argsort(merged["index"])
    merged = {key: value[order] for key, value in merged.items()}
    fine_correct = merged["fine_pred"] == merged["fine_label"]
    parent_correct = merged["parent_pred"] == merged["parent_label"]
    metrics = {
        "n": int(len(fine_correct)),
        "fine_accuracy": float(fine_correct.mean()),
        "parent_accuracy": float(parent_correct.mean()),
        "fine_cls_accuracy": float((merged["fine_cls_pred"] == merged["fine_label"]).mean()),
        "parent_cls_accuracy": float((merged["parent_cls_pred"] == merged["parent_label"]).mean()),
        "fine_tile_accuracy": float((merged["fine_tile_pred"] == merged["fine_label"]).mean()),
        "parent_tile_accuracy": float((merged["parent_tile_pred"] == merged["parent_label"]).mean()),
        "states": {
            "both_valid": int(np.sum(parent_correct & fine_correct)),
            "parent_only": int(np.sum(parent_correct & ~fine_correct)),
            "fine_only": int(np.sum(~parent_correct & fine_correct)),
            "neither": int(np.sum(~parent_correct & ~fine_correct)),
        },
        "fine_attention_concentration_mean": float(merged["fine_attention_concentration"].mean()),
        "parent_attention_concentration_mean": float(merged["parent_attention_concentration"].mean()),
    }
    return metrics, merged


def _save_packet(
    path: Path,
    cache: dict[str, Any],
    arrays: dict[str, np.ndarray],
    layout: ConceptLayout,
) -> None:
    index = arrays["index"].astype(np.int64)
    payload = dict(arrays)
    payload["name"] = np.asarray(cache["name"])[index]
    payload["fine_classes"] = np.asarray(layout.fine_classes, dtype=str)
    payload["parent_classes"] = np.asarray(layout.parent_classes, dtype=str)
    np.savez(path, **payload)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_cache = _load(args.train_cache)
    test_cache = _load(args.test_cache)
    fine_classes = [str(x) for x in train_cache["class_names"].tolist()]
    if fine_classes != [str(x) for x in test_cache["class_names"].tolist()]:
        raise ValueError("TRAIN/TEST class names differ")
    parent_map = json.loads(args.parent_map.read_text(encoding="utf-8"))
    layout = ConceptLayout.from_parent_map(fine_classes, parent_map)
    train_idx, val_idx = _split(train_cache["label"].astype(np.int64), args.val_ratio, args.seed)

    model = DinoConceptEvidenceProjector(
        embedding_dim=int(train_cache["cls_token"].shape[-1]),
        parent_classes=len(layout.parent_classes),
        fine_classes=len(layout.fine_classes),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataset = TensorDataset(
        torch.as_tensor(train_cache["cls_token"])[train_idx],
        torch.as_tensor(train_cache["tile_tokens"])[train_idx],
        torch.as_tensor(train_cache["label"], dtype=torch.long)[train_idx],
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    best_state = None
    best_val = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for cls_token, tile_tokens, fine_target in loader:
            cls_token = cls_token.to(device=device, dtype=torch.float32)
            tile_tokens = tile_tokens.to(device=device, dtype=torch.float32)
            fine_target = fine_target.to(device)
            parent_target = layout.parent_targets(fine_target)
            loss, _ = _loss(
                model,
                cls_token,
                tile_tokens,
                fine_target,
                parent_target,
                drop_rate=args.tile_drop_rate,
                concentration_limit=args.concentration_limit,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(fine_target)
            seen += len(fine_target)
        val_metrics, _ = _evaluate(model, train_cache, val_idx, layout, device, args.eval_batch_size)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, seen),
            "val_parent_accuracy": val_metrics["parent_accuracy"],
            "val_fine_accuracy": val_metrics["fine_accuracy"],
            "val_fine_tile_accuracy": val_metrics["fine_tile_accuracy"],
        }
        history.append(row)
        print(f"[epoch {epoch:02d}] {json.dumps(row)}", flush=True)
        score = float(val_metrics["parent_accuracy"] + val_metrics["fine_accuracy"])
        if score > best_val:
            best_val = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)

    train_metrics, train_arrays = _evaluate(model, train_cache, train_idx, layout, device, args.eval_batch_size)
    val_metrics, val_arrays = _evaluate(model, train_cache, val_idx, layout, device, args.eval_batch_size)
    test_indices = np.arange(len(test_cache["label"]), dtype=np.int64)
    test_metrics, test_arrays = _evaluate(model, test_cache, test_indices, layout, device, args.eval_batch_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state": best_state,
        "layout": {
            "fine_classes": list(layout.fine_classes),
            "parent_classes": list(layout.parent_classes),
            "fine_to_parent_index": list(layout.fine_to_parent_index),
        },
        "config": {
            "embedding_dim": int(train_cache["cls_token"].shape[-1]),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
    }
    torch.save(checkpoint, args.out_dir / "dino_concept_evidence_v226.pt")
    _save_packet(args.out_dir / "train_evidence_packet.npz", train_cache, train_arrays, layout)
    _save_packet(args.out_dir / "val_evidence_packet.npz", train_cache, val_arrays, layout)
    _save_packet(args.out_dir / "test_evidence_packet.npz", test_cache, test_arrays, layout)
    summary = {
        "mode": "v226a_dino_parent_fine_evidence_projector",
        "uses_gate": False,
        "backbone_frozen": True,
        "test_used_for_model_selection": False,
        "layout": checkpoint["layout"],
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_indices)),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", type=Path, required=True)
    parser.add_argument("--test_cache", type=Path, required=True)
    parser.add_argument("--parent_map", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--tile_drop_rate", type=float, default=0.25)
    parser.add_argument("--concentration_limit", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=226)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
