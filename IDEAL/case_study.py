#

import argparse
import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from data_utils import load_features, build_gcn_adj
from gcn_model import GCNWithMLP, MLPDecoder


DATASET_CFG: Dict[str, Dict[str, torch.Tensor]] = {
    "MDAD": {
        "base_path": "./MDAD/",
        "microbe_offset": 1373,
    },
    "DrugVirus": {
        "base_path": "./DrugVirus/",
        "microbe_offset": 175,
    },
    "aBiofilm": {
        "base_path": "./aBiofilm/",
        "microbe_offset": 1720,
    },
}


def load_name_mapping(excel_path: str, id_col: str, name_col: str) -> Optional[Dict[int, str]]:
    if not os.path.exists(excel_path):
        return None
    df = pd.read_excel(excel_path)
    if id_col not in df.columns or name_col not in df.columns:
        return None
    return pd.Series(df[name_col].values, index=df[id_col]).to_dict()


def load_graph_tensors(base_path: str, num_drug: int, num_microbe: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    Sd = np.loadtxt(os.path.join(base_path, "drugsimilarity.txt"))
    Sm = np.loadtxt(os.path.join(base_path, "microbesimilarity.txt"))

    adj_path = os.path.join(base_path, "adj_out.txt")
    I = np.zeros((num_drug, num_microbe), dtype=float)
    adj_df = pd.read_csv(adj_path, sep=r"\s+", header=None)
    for _, row in adj_df.iterrows():
        drug_idx, microbe_idx, val = map(int, row[:3])
        I[drug_idx, microbe_idx] = val

    A = build_gcn_adj(Sd, Sm, I)
    row, col = np.where(A != 0)
    edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(A[row, col], dtype=torch.float32, device=device)
    return edge_index, edge_weight


def load_feature_tensors(base_path: str, device: torch.device) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, int]]:
    drug_features, drug_bert, drug_fg, microbe_features, microbe_bert, microbe_path = load_features(
        os.path.join(base_path, "drugfeatures.txt"),
        os.path.join(base_path, "drug_bert.xlsx"),
        os.path.join(base_path, "fingerprint.xlsx"),
        os.path.join(base_path, "microbefeatures.txt"),
        os.path.join(base_path, "microbe_bert.xlsx"),
        os.path.join(base_path, "microbe_path.xlsx"),
    )

    shapes = {
        "num_drug": drug_fg.shape[0],
        "num_microbe": microbe_features.shape[0],
        "drug_feat_dim": drug_fg.shape[1],
        "microbe_feat_dim": microbe_features.shape[1],
    }

    tensors = tuple(
        torch.tensor(arr, dtype=torch.float32, device=device)
        for arr in (drug_fg, drug_features, drug_bert, microbe_features, microbe_bert, microbe_path)
    )
    return tensors, shapes


def build_model(
    dataset: str,
    shapes: Dict[str, int],
    hidden_dim: int,
    dropout: float,
    f_flag: int,
    device: torch.device,
) -> Tuple[GCNWithMLP, MLPDecoder]:
    model = GCNWithMLP(
        drug_in_dim=shapes["drug_feat_dim"],
        drug_out_dim=shapes["num_drug"],
        microbe_dim=shapes["microbe_feat_dim"],
        microbe_out_dim=shapes["microbe_feat_dim"],
        gcn_hidden=hidden_dim,
        dropout=dropout,
        use_microbe_mlp=False,
        dataset_name=dataset,
        f=f_flag,
    ).to(device)

    decoder = MLPDecoder(input_dim=hidden_dim).to(device)
    return model, decoder


def predict_single_drug(
    model: GCNWithMLP,
    decoder: MLPDecoder,
    drug_id: int,
    feature_tensors: Tuple[torch.Tensor, ...],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    microbe_offset: int,
    num_microbe: int,
) -> np.ndarray:
    model.eval()
    decoder.eval()
    with torch.no_grad():
        embeddings, _ = model(feature_tensors, edge_index, edge_weight)
        drug_embed = embeddings[drug_id].unsqueeze(0).expand(num_microbe, -1)
        microbe_nodes = torch.arange(num_microbe, device=embeddings.device) + microbe_offset
        microbe_embeddings = embeddings[microbe_nodes]
        logits = decoder(drug_embed, microbe_embeddings)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def predict_single_microbe(
    model: GCNWithMLP,
    decoder: MLPDecoder,
    microbe_id: int,
    feature_tensors: Tuple[torch.Tensor, ...],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    microbe_offset: int,
    num_drug: int,
) -> np.ndarray:
    model.eval()
    decoder.eval()
    with torch.no_grad():
        embeddings, _ = model(feature_tensors, edge_index, edge_weight)
        microbe_node = microbe_offset + microbe_id
        microbe_embed = embeddings[microbe_node].unsqueeze(0).expand(num_drug, -1)
        drug_embeddings = embeddings[:num_drug]
        logits = decoder(drug_embeddings, microbe_embed)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def main(args: argparse.Namespace) -> None:
    if args.dataset not in DATASET_CFG:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    cfg = DATASET_CFG[args.dataset]
    base_path = cfg["base_path"]
    microbe_offset = cfg["microbe_offset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    feature_tensors, shape_info = load_feature_tensors(base_path, device)
    edge_index, edge_weight = load_graph_tensors(
        base_path, shape_info["num_drug"], shape_info["num_microbe"], device
    )

    model, decoder = build_model(
        dataset=args.dataset,
        shapes=shape_info,
        hidden_dim=args.hidden_dim,
        dropout=0.4,
        f_flag=args.f,
        device=device,
    )

    fold_subdir = os.path.join(base_path, f"fold{args.fold}")
   # model_path = os.path.join(fold_subdir, f"{args.dataset}_gcn_model_fused.pth")
    #decoder_path = os.path.join(fold_subdir, f"{args.dataset}_decoder_fused.pth")
    model_path = os.path.join(fold_subdir, f"{args.dataset}_hd{args.hidden_dim}_gcn_model_fused.pth")
    decoder_path = os.path.join(fold_subdir, f"{args.dataset}_hd{args.hidden_dim}_decoder_fused.pth")

    if not os.path.exists(model_path) or not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Missing model or decoder file in {fold_subdir}")

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    drug_names = load_name_mapping(os.path.join(base_path, "drugs.xlsx"), "Drug_ID", "Drug_name")
    microbe_names = load_name_mapping(os.path.join(base_path, "microbes.xlsx"), "Microbe_ID", "Microbe_name")

    output_dir = os.path.join(base_path, "case_study_results")
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "drug_to_microbe":
        if args.drug_id is None:
            raise ValueError("drug_to_microbe 模式需要 --drug_id")
        drug_idx = args.drug_id - 1
        scores = predict_single_drug(
            model,
            decoder,
            drug_idx,
            feature_tensors,
            edge_index,
            edge_weight,
            microbe_offset,
            shape_info["num_microbe"],
        )
        microbe_ids = np.arange(shape_info["num_microbe"])+1
        result_df = pd.DataFrame({
            "Drug_ID": args.drug_id,
            "Microbe_ID": microbe_ids,
            "Predicted_Score": scores,
        })

        if drug_names:
            result_df["Drug_name"] = drug_names.get(args.drug_id, "N/A")
        if microbe_names:
            result_df["Microbe_name"] = result_df["Microbe_ID"].map(lambda idx: microbe_names.get(idx, "N/A"))
            result_df = result_df[["Drug_ID", "Drug_name", "Microbe_ID", "Microbe_name", "Predicted_Score"]]
        else:
            result_df = result_df[["Drug_ID", "Microbe_ID", "Predicted_Score"]]

        output_path = os.path.join(output_dir, f"case_study_drug_{args.drug_id}_fold_{args.fold}.xlsx")

    else:
        if args.microbe_id is None:
            raise ValueError("microbe_to_drug 模式需要 --microbe_id")
        microbe_idx = args.microbe_id - 1
        scores = predict_single_microbe(
            model,
            decoder,
            microbe_idx,
            feature_tensors,
            edge_index,
            edge_weight,
            microbe_offset,
            shape_info["num_drug"],
        )
        drug_ids = np.arange(shape_info["num_drug"])
        if args.drug_ids:
            filtered_ids = np.array(args.drug_ids) - 1
            mask = np.isin(drug_ids, filtered_ids)
            drug_ids = drug_ids[mask]
            scores = scores[mask]

        result_df = pd.DataFrame({
            "Microbe_ID": args.microbe_id,
            "Drug_ID": drug_ids+1,
            "Predicted_Score": scores,
        })

        if microbe_names:
            result_df["Microbe_name"] = microbe_names.get(args.microbe_id, "N/A")
        if drug_names:
            result_df["Drug_name"] = result_df["Drug_ID"].map(lambda idx: drug_names.get(idx, "N/A"))

        columns = ["Microbe_ID", "Microbe_name", "Drug_ID", "Drug_name", "Predicted_Score"]
        present_columns = [col for col in columns if col in result_df.columns]
        result_df = result_df[present_columns]

        output_path = os.path.join(output_dir, f"case_study_microbe_{args.microbe_id}_fold_{args.fold}.xlsx")

    result_df.sort_values(by="Predicted_Score", ascending=False, inplace=True, ignore_index=True)
    result_df.to_excel(output_path, index=False)
    print(f"Saved results to {output_path}")

    top_k = result_df.head(10)
    print("\nTop 10 predictions:")
    print(top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Case study prediction for drug-microbe interactions.")
    parser.add_argument("--dataset", type=str, default="DrugVirus")#选数据集
    parser.add_argument("--fold", type=int, default=1)#选折
    parser.add_argument("--mode", choices=["drug_to_microbe", "microbe_to_drug"], default="microbe_to_drug")#选药物或者微生物
    parser.add_argument("--drug_id", type=int, default=11)#选药物id（从1开始）
    parser.add_argument("--microbe_id", type=int, default=9)#选微生物id（从1开始）
    parser.add_argument("--drug_ids", type=int, nargs="*", default=None, help="microbe_to_drug 模式下可选，限制输出药物列表")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout_retrain", type=float, default=0.4)
    parser.add_argument("--f", type=int, default=2)
    main(parser.parse_args())
