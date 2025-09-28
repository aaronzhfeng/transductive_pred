from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple

import click
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from graph2transforms.data.corpus import EditStepDataset, ReactionSequenceDataset, collate_steps
from graph2transforms.models.graph_transformer import EditModel
from graph2transforms.edits.actions import Action, ActionType, BondOrder, BondStereo
from graph2transforms.core.rollout import rw_mol_from_mapped_reactants, apply_single_action, get_map_to_idx
from graph2transforms.features.featurize import serialize_graph


def action_exact_match(pred: Dict[str, Any], tgt: Dict[str, Any]) -> bool:
    if pred.get("type") != tgt.get("type"):
        return False
    t = pred["type"]
    # Compare minimal args per type
    if t in (ActionType.ADD_BOND.value, ActionType.CHANGE_BOND.value, ActionType.DEL_BOND.value):
        for k in ["i", "j"]:
            if int(pred.get("args", {}).get(k, -9999)) != int(tgt.get("args", {}).get(k, -9999)):
                return False
        if t == ActionType.ADD_BOND.value:
            return (str(pred["args"].get("order")) == str(tgt["args"].get("order"))) and (str(pred["args"].get("stereo")) == str(tgt["args"].get("stereo")))
        if t == ActionType.CHANGE_BOND.value:
            return (str(pred["args"].get("new_order")) == str(tgt["args"].get("new_order"))) and (str(pred["args"].get("new_stereo")) == str(tgt["args"].get("new_stereo")))
        return True
    if t in (ActionType.H_GAIN.value, ActionType.H_LOSS.value, ActionType.CHANGE_ATOM.value):
        return int(pred.get("args", {}).get("i", -9999)) == int(tgt.get("args", {}).get("i", -9999))
    if t == ActionType.STOP.value:
        return True
    # For DELETE_SUBGRAPH/REMOVE_LG we skip exact set equality here
    return False


def build_pred_action(model: EditModel, node_feats: torch.LongTensor, device: torch.device) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        out = model(node_feats.to(device))
        type_id = int(torch.argmax(out["type"]).item())
        type_str = ActionType(type_id).value
        # Default args
        args: Dict[str, Any] = {}
        nodes_n = node_feats.size(0)
        if type_str in (ActionType.ADD_BOND.value, ActionType.CHANGE_BOND.value, ActionType.DEL_BOND.value, ActionType.H_GAIN.value, ActionType.H_LOSS.value, ActionType.CHANGE_ATOM.value):
            i_idx = int(torch.argmax(out["i"]).item())
            args["i"] = int(node_feats[i_idx, 9].item())  # map_num
            if type_str in (ActionType.ADD_BOND.value, ActionType.CHANGE_BOND.value, ActionType.DEL_BOND.value):
                # compute j conditioned on i
                h, _ = model.encode(node_feats.to(device))
                j_logits = model.policy.j_logits(h, i_idx)
                j_idx = int(torch.argmax(j_logits).item())
                args["j"] = int(node_feats[j_idx, 9].item())
                if type_str == ActionType.ADD_BOND.value:
                    order_idx = 0
                    stereo_idx = 0
                    order_map = [BondOrder.SINGLE.value, BondOrder.DOUBLE.value, BondOrder.TRIPLE.value, BondOrder.AROMATIC.value]
                    stereo_map = [BondStereo.NONE.value, BondStereo.E.value, BondStereo.Z.value]
                    ologits, slogits = model.policy.order_stereo_logits(h, i_idx, j_idx)
                    order_idx = int(torch.argmax(ologits).item())
                    stereo_idx = int(torch.argmax(slogits).item())
                    args["order"] = order_map[order_idx]
                    args["stereo"] = stereo_map[stereo_idx]
                if type_str == ActionType.CHANGE_BOND.value:
                    order_map = [BondOrder.SINGLE.value, BondOrder.DOUBLE.value, BondOrder.TRIPLE.value, BondOrder.AROMATIC.value]
                    stereo_map = [BondStereo.NONE.value, BondStereo.E.value, BondStereo.Z.value]
                    h2, _ = model.encode(node_feats.to(device))
                    ologits, slogits = model.policy.order_stereo_logits(h2, i_idx, j_idx)
                    args["new_order"] = order_map[int(torch.argmax(ologits).item())]
                    args["new_stereo"] = stereo_map[int(torch.argmax(slogits).item())]
        return {"type": type_str, "args": args}


@click.command()
@click.option("--corpus-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False))
@click.option("--epochs", default=1, show_default=True, type=int)
@click.option("--batch-size", default=32, show_default=True, type=int)
@click.option("--lr", default=1e-3, show_default=True, type=float)
@click.option("--log-every", default=100, show_default=True, type=int)
@click.option("--val-every", default=500, show_default=True, type=int)
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", show_default=True)
@click.option("--dim", default=256, show_default=True, type=int)
@click.option("--depth", default=4, show_default=True, type=int)
@click.option("--heads", default=4, show_default=True, type=int)
def main(corpus_dir: str, out_dir: str, epochs: int, batch_size: int, lr: float, log_every: int, val_every: int, device: str, dim: int, depth: int, heads: int):
    os.makedirs(out_dir, exist_ok=True)
    action_vocab_path = os.path.join(corpus_dir, "vocabs", "actions.json")
    action_to_id = json.load(open(action_vocab_path, "r"))
    num_action_types = len(action_to_id)
    id_to_action = {v: k for k, v in action_to_id.items()}

    model = EditModel(num_action_types=num_action_types, dim=dim, depth=depth, heads=heads)
    print(f"Model Params: {sum(p.numel() for p in model.parameters())}")
    device_t = torch.device(device)
    model.to(device_t)

    train_ds = EditStepDataset(corpus_dir, split="train", action_vocab_path=action_vocab_path)
    val_seq = ReactionSequenceDataset(corpus_dir, split="val")

    opt = optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loader_idx = 0
        for i in tqdm(range(0, len(train_ds), batch_size), desc=f"epoch {epoch}"):
            batch = [train_ds[j] for j in range(i, min(i + batch_size, len(train_ds)))]
            # Compute batched loss by summing per-graph losses (teacher forcing)
            total_loss = 0.0
            correct_type = 0
            total_type = 0
            for ex in batch:
                node_feats = ex["node_feats"].to(device_t)
                target = ex["target"]
                out = model(node_feats)
                tgt_type_id = action_to_id[target["type"]]
                loss = ce(out["type"].unsqueeze(0), torch.tensor([tgt_type_id], device=device_t))
                # Simple type accuracy
                pred_type = int(torch.argmax(out["type"]).item())
                correct_type += int(pred_type == tgt_type_id)
                total_type += 1
                total_loss = total_loss + loss
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            step += 1
            if step % log_every == 0:
                avg_type_acc = correct_type / max(1, total_type)
                print(f"step {step} | epoch {epoch} | loss {float(total_loss.item()):.4f} | type_acc {avg_type_acc:.3f}")

            if step % val_every == 0 and len(val_seq) > 0:
                # Teacher-forced eval on a small subset
                model.eval()
                with torch.no_grad():
                    tf_correct = 0
                    tf_total = 0
                    N_eval = min(200, len(train_ds))
                    for k in range(N_eval):
                        ex = train_ds[k]
                        out = model(ex["node_feats"].to(device_t))
                        pred_type = int(torch.argmax(out["type"]).item())
                        tf_total += 1
                        tf_correct += int(pred_type == action_to_id[ex["target"]["type"]])
                print(f"TF Acc (type only) on {N_eval} examples: {tf_correct / max(1, tf_total):.3f}")

                # Autoregressive eval: sequence-by-sequence
                n_eval_seq = min(50, len(val_seq))
                seq_match = 0
                seq_total = 0
                for sidx in range(n_eval_seq):
                    rxn = val_seq[sidx]
                    mapped_rxn = rxn.get("mapped_rxn", "")
                    rw = rw_mol_from_mapped_reactants(mapped_rxn)
                    exact_all = True
                    for rec in rxn.get("records", []):
                        # Build current graph from rw
                        graph = serialize_graph(rw)
                        nodes = graph["nodes"]
                        node_feats = torch.tensor([
                            [n.get("z", 0), n.get("degree", 0), n.get("total_valence", 0), n.get("formal_charge", 0), n.get("hybridization", 0), n.get("aromatic", 0), n.get("implicit_h", 0), n.get("explicit_h", 0), n.get("chiral_tag", 0), n.get("map_num", 0)]
                            for n in nodes
                        ], dtype=torch.long)
                        pred = build_pred_action(model, node_feats, device_t)
                        tgt = rec["target_action"]
                        exact = action_exact_match(pred, tgt)
                        exact_all = exact_all and exact
                        # Apply predicted action
                        act = Action(type=ActionType(pred["type"]), args=pred.get("args", {}))
                        rw, failed = apply_single_action(rw, act)
                        if failed:
                            break
                        if pred["type"] == ActionType.STOP.value:
                            break
                    seq_total += 1
                    seq_match += int(exact_all)
                print(f"AR Seq Exact (type+args subset): {seq_match / max(1, seq_total):.3f}")

        # save checkpoint each epoch
        ckpt_path = os.path.join(out_dir, f"model-epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "config": {"dim": dim, "depth": depth, "heads": heads, "num_actions": num_action_types}}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

