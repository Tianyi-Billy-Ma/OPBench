"""Process PDMP raw data into heterogeneous graph format with metapaths.

Usage:
    PYTHONDONTWRITEBYTECODE=1 python -m src.scripts.data_processing.process_pdmp \\
        --input_dir datasets/raw/pdmp \\
        --output_dir datasets/hetgraphs/pdmp_opioid_detect/raw \\
        --compute_metapaths

Metapath computation uses PyG's AddMetaPaths transform. Use --max_sample to limit
neighbors per step for scalability (None=full computation).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SEX_MAP = {
    1: "male",
    2: "female",
    "1": "male",
    "2": "female",
    "M": "male",
    "F": "female",
    "m": "male",
    "f": "female",
}

PAYMENT_TYPE_MAP = {
    1: "Other",
    2: "Cash/Self-Pay",
    3: "Medicare/Medicaid",
    4: "Private Insurance",
    "1": "Other",
    "2": "Cash/Self-Pay",
    "3": "Medicare/Medicaid",
    "4": "Private Insurance",
}


def load_pdmp_data(input_dir: Path) -> pd.DataFrame:
    overdose_path = input_dir / "overdose.csv"
    nonoverdose_path = input_dir / "nonoverdose.csv"

    logger.info("Loading overdose data from %s", overdose_path)
    overdose_df = pd.read_csv(overdose_path)
    overdose_df["label"] = 1
    logger.info("Loaded %d overdose records", len(overdose_df))

    logger.info("Loading non-overdose data from %s", nonoverdose_path)
    nonoverdose_df = pd.read_csv(nonoverdose_path)
    nonoverdose_df["label"] = 0
    logger.info("Loaded %d non-overdose records", len(nonoverdose_df))

    df = pd.concat([overdose_df, nonoverdose_df], ignore_index=True)
    logger.info("Total records: %d", len(df))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    initial_count = len(df)

    string_cols = df.select_dtypes(include=["object"]).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    df = df.replace(["", "nan", "None", "NaN"], np.nan)

    for col in [
        "PrescriberSpecialty",
        "PharmacyZip",
        "PrescriberZip",
        "PatientZip",
        "PrescriberBACCode",
        "PrescriberBACSubCode",
        "PharmacyBACCode",
        "PharmacyBACSubCode",
        "NDC",
        "Drug",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    df = df[df["Quantity"] > 0]
    df = df[df["DaysSupply"] > 0]
    df["DateFilled"] = pd.to_datetime(df["DateFilled"], errors="coerce")

    required_cols = [
        "DateFilled",
        "PatientGroupIDHash",
        "PrescriberHash",
        "PharmacyHash",
        "TherClassCode",
    ]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    logger.info("Cleaned: %d -> %d records", initial_count, len(df))
    return df


def balance_dataset(
    df: pd.DataFrame, ratio: float = 1.0, seed: int = 42
) -> pd.DataFrame:
    """Balance dataset by undersampling majority class (non-overdose) at patient level.

    Args:
        df: DataFrame with prescription records containing PatientGroupIDHash and label.
        ratio: Target ratio of majority:minority. 1.0 = balanced (1:1), 2.0 = 1:2.
        seed: Random seed for reproducibility.

    Returns:
        Filtered DataFrame with balanced patient distribution.
    """
    patient_labels = df.groupby("PatientGroupIDHash")["label"].first()
    overdose_patients = patient_labels[patient_labels == 1].index.tolist()
    non_overdose_patients = patient_labels[patient_labels == 0].index.tolist()

    n_minority = len(overdose_patients)
    n_majority = len(non_overdose_patients)
    target_majority = int(n_minority * ratio)

    logger.info("Balancing dataset (ratio=%.1f, seed=%d)...", ratio, seed)
    logger.info("  Before: %d overdose, %d non-overdose", n_minority, n_majority)

    if target_majority >= n_majority:
        logger.info(
            "  Target (%d) >= actual majority (%d), skipping sampling",
            target_majority,
            n_majority,
        )
        return df

    rng = np.random.default_rng(seed)
    sampled_non_overdose = rng.choice(
        non_overdose_patients, size=target_majority, replace=False
    ).tolist()

    keep_patients = set(overdose_patients + sampled_non_overdose)
    df_balanced = df[df["PatientGroupIDHash"].isin(keep_patients)].copy()

    logger.info(
        "  After: %d overdose, %d non-overdose", n_minority, len(sampled_non_overdose)
    )
    logger.info("  Records: %d -> %d", len(df), len(df_balanced))

    return df_balanced


def create_entity_mappings(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    logger.info("Creating entity ID mappings...")
    mappings = {
        "patient": {h: i for i, h in enumerate(df["PatientGroupIDHash"].unique())},
        "prescriber": {h: i for i, h in enumerate(df["PrescriberHash"].unique())},
        "pharmacy": {h: i for i, h in enumerate(df["PharmacyHash"].unique())},
        "drug": {str(h): i for i, h in enumerate(df["TherClassCode"].unique())},
    }
    for k, v in mappings.items():
        logger.info("  - %d %s", len(v), k)
    return mappings


def compute_patient_stats(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    logger.info("Computing patient statistics...")
    patient_groups = df.groupby("PatientGroupIDHash")
    stats = patient_groups.agg(
        {
            "PatientAge": "first",
            "PatientSex": "first",
            "PatientZip": "first",
            "label": "first",
        }
    ).reset_index()
    stats["patient_id"] = stats["PatientGroupIDHash"].map(mappings["patient"])
    return stats.sort_values("patient_id").reset_index(drop=True)


def compute_prescriber_stats(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    logger.info("Computing prescriber statistics...")
    groups = df.groupby("PrescriberHash")
    stats = groups.agg(
        {
            "PrescriberSpecialty": "first",
            "PrescriberZip": "first",
            "PrescriberBACCode": "first",
            "PrescriberBACSubCode": "first",
        }
    ).reset_index()
    stats.columns = ["PrescriberHash", "specialty", "zip", "bac_code", "bac_sub_code"]
    stats["prescriber_id"] = stats["PrescriberHash"].map(mappings["prescriber"])
    return stats.sort_values("prescriber_id").reset_index(drop=True)


def compute_pharmacy_stats(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    logger.info("Computing pharmacy statistics...")
    groups = df.groupby("PharmacyHash")
    stats = groups.agg(
        {
            "PharmacyZip": "first",
            "PharmacyBACCode": "first",
            "PharmacyBACSubCode": "first",
        }
    ).reset_index()
    stats.columns = ["PharmacyHash", "zip", "bac_code", "bac_sub_code"]
    stats["pharmacy_id"] = stats["PharmacyHash"].map(mappings["pharmacy"])
    return stats.sort_values("pharmacy_id").reset_index(drop=True)


def compute_drug_stats(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    logger.info("Computing drug statistics...")
    groups = df.groupby("TherClassCode")
    stats = groups.agg(
        {
            "TherClassDesc": "first",
            "NDC": "first",
            "Drug": "first",
        }
    ).reset_index()
    stats.columns = ["TherClassCode", "ther_class_desc", "ndc", "drug_name"]
    stats["drug_id"] = stats["TherClassCode"].astype(str).map(mappings["drug"])
    return stats.sort_values("drug_id").reset_index(drop=True)


def generate_patient_texts(patient_stats: pd.DataFrame) -> list[str]:
    logger.info("Generating patient text descriptions...")
    texts = []
    for _, row in tqdm(
        patient_stats.iterrows(), total=len(patient_stats), desc="Patients"
    ):
        sex = SEX_MAP.get(row["PatientSex"], "unknown")
        text = f"ID: {row['PatientGroupIDHash']}, Age: {int(row['PatientAge'])}, Sex: {sex}, Zip: {row['PatientZip']}"
        texts.append(text)
    return texts


def generate_prescriber_texts(prescriber_stats: pd.DataFrame) -> list[str]:
    logger.info("Generating prescriber text descriptions...")
    texts = []
    for _, row in tqdm(
        prescriber_stats.iterrows(), total=len(prescriber_stats), desc="Prescribers"
    ):
        text = f"ID: {row['PrescriberHash']}, Zip: {row['zip']}, Specialty: {row['specialty']}, BAC: {row['bac_code']}-{row['bac_sub_code']}"
        texts.append(text)
    return texts


def generate_pharmacy_texts(pharmacy_stats: pd.DataFrame) -> list[str]:
    logger.info("Generating pharmacy text descriptions...")
    texts = []
    for _, row in tqdm(
        pharmacy_stats.iterrows(), total=len(pharmacy_stats), desc="Pharmacies"
    ):
        text = f"ID: {row['PharmacyHash']}, Zip: {row['zip']}, BAC: {row['bac_code']}-{row['bac_sub_code']}"
        texts.append(text)
    return texts


def generate_drug_texts(drug_stats: pd.DataFrame) -> list[str]:
    logger.info("Generating drug text descriptions...")
    texts = []
    for _, row in tqdm(drug_stats.iterrows(), total=len(drug_stats), desc="Drugs"):
        text = f"NDC: {row['ndc']}, Drug: {row['drug_name']}, Class: {row['ther_class_desc']} ({row['TherClassCode']})"
        texts.append(text)
    return texts


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required: pip install sentence-transformers"
        )

    logger.info("Loading SentenceBERT model (all-mpnet-base-v2)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    logger.info("Encoding %d texts...", len(texts))
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    logger.info("Encoded embeddings shape: %s", embeddings.shape)
    return embeddings


def extract_edges(df: pd.DataFrame, mappings: dict) -> dict[str, np.ndarray]:
    logger.info("Extracting edge lists...")
    patient_ids = df["PatientGroupIDHash"].map(mappings["patient"]).values
    prescriber_ids = df["PrescriberHash"].map(mappings["prescriber"]).values
    pharmacy_ids = df["PharmacyHash"].map(mappings["pharmacy"]).values
    drug_ids = df["TherClassCode"].astype(str).map(mappings["drug"]).values

    def unique_edges(src, dst):
        pairs = np.stack([src, dst], axis=1)
        unique = np.unique(pairs, axis=0)
        return unique.T.astype(np.int64)

    edges = {
        "patient_take_drug": unique_edges(patient_ids, drug_ids),
        "patient_pickup_at_pharmacy": unique_edges(patient_ids, pharmacy_ids),
        "patient_visit_prescriber": unique_edges(patient_ids, prescriber_ids),
        "prescriber_prescribe_drug": unique_edges(prescriber_ids, drug_ids),
        "pharmacy_dispense_drug": unique_edges(pharmacy_ids, drug_ids),
    }

    for name, e in edges.items():
        logger.info("  - %s: %d edges", name, e.shape[1])
    return edges


def save_outputs(
    output_dir: Path,
    patient_features: np.ndarray,
    prescriber_features: np.ndarray,
    pharmacy_features: np.ndarray,
    drug_features: np.ndarray,
    patient_labels: np.ndarray,
    edges: dict[str, np.ndarray],
    mappings: dict,
    metadata: dict,
) -> None:
    logger.info("Saving outputs to %s", output_dir)
    output_dir = Path(output_dir)
    edges_dir = output_dir / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)

    num_patients = patient_features.shape[0]
    chunk_size = (num_patients + 3) // 4
    for i in range(4):
        start, end = i * chunk_size, min((i + 1) * chunk_size, num_patients)
        np.save(
            output_dir / f"patient_features_part{i}.npy", patient_features[start:end]
        )
    logger.info("  - Saved patient features in 4 parts")

    num_prescribers = prescriber_features.shape[0]
    chunk_size = (num_prescribers + 1) // 2
    for i in range(2):
        start, end = i * chunk_size, min((i + 1) * chunk_size, num_prescribers)
        np.save(
            output_dir / f"prescriber_features_part{i}.npy",
            prescriber_features[start:end],
        )
    logger.info("  - Saved prescriber features in 2 parts")

    np.save(output_dir / "pharmacy_features.npy", pharmacy_features)
    np.save(output_dir / "drug_features.npy", drug_features)
    np.save(output_dir / "patient_labels.npy", patient_labels)

    for name, e in edges.items():
        np.save(edges_dir / f"{name}.npy", e)
    logger.info("  - Saved edge files")

    mappings_serializable = {
        k: {str(kk): vv for kk, vv in v.items()} for k, v in mappings.items()
    }
    with open(output_dir / "mappings.json", "w") as f:
        json.dump(mappings_serializable, f)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  - Saved metadata.json")


def compute_and_save_metapaths(output_dir: Path, max_sample: int | None = None) -> None:
    import torch_geometric.transforms as T

    from src.data.loader import pdmp_builder
    from src.data.metapaths import PDMP_METAPATHS, save_metapaths

    logger.info("Loading processed graph for metapath computation...")
    data = pdmp_builder(output_dir)

    for edge_type in data.edge_types:
        data[edge_type].edge_index = data[edge_type].edge_index.contiguous()

    metapath_names = list(PDMP_METAPATHS.keys())
    pyg_metapaths = list(PDMP_METAPATHS.values())

    logger.info("Computing %d metapaths using PyG AddMetaPaths...", len(pyg_metapaths))
    if max_sample is not None:
        logger.info("  max_sample=%d", max_sample)

    transform = T.AddMetaPaths(
        metapaths=pyg_metapaths,
        drop_orig_edge_types=False,
        drop_unconnected_node_types=False,
        max_sample=max_sample,
    )
    data = transform(data)

    computed_metapaths: dict[str, np.ndarray] = {}
    for idx, name in enumerate(metapath_names):
        edge_type = ("patient", f"metapath_{idx}", "patient")
        if edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index.cpu().numpy()
            computed_metapaths[name] = edge_index
            logger.info("  - %s: %d edges", name, edge_index.shape[1])
        else:
            logger.warning("  - %s: not found in computed edge types", name)

    if not computed_metapaths:
        raise RuntimeError("No metapaths were computed - check input data connectivity")

    logger.info("Saving metapaths...")
    save_metapaths(computed_metapaths, output_dir)


def main(
    input_dir: str,
    output_dir: str,
    batch_size: int = 64,
    skip_encoding: bool = False,
    balance_ratio: float | None = None,
    seed: int = 42,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    logger.info("=" * 60)
    logger.info("PDMP Data Processing Pipeline")
    logger.info("=" * 60)

    df = load_pdmp_data(input_dir)
    df = clean_data(df)

    if balance_ratio is not None:
        df = balance_dataset(df, ratio=balance_ratio, seed=seed)

    mappings = create_entity_mappings(df)

    patient_stats = compute_patient_stats(df, mappings)
    prescriber_stats = compute_prescriber_stats(df, mappings)
    pharmacy_stats = compute_pharmacy_stats(df, mappings)
    drug_stats = compute_drug_stats(df, mappings)

    patient_texts = generate_patient_texts(patient_stats)
    prescriber_texts = generate_prescriber_texts(prescriber_stats)
    pharmacy_texts = generate_pharmacy_texts(pharmacy_stats)
    drug_texts = generate_drug_texts(drug_stats)

    if skip_encoding:
        logger.info("Skipping SentenceBERT encoding (--skip_encoding)")
        patient_features = np.random.randn(len(patient_texts), 768).astype(np.float32)
        prescriber_features = np.random.randn(len(prescriber_texts), 768).astype(
            np.float32
        )
        pharmacy_features = np.random.randn(len(pharmacy_texts), 768).astype(np.float32)
        drug_features = np.random.randn(len(drug_texts), 768).astype(np.float32)
    else:
        patient_features = encode_texts(patient_texts, batch_size)
        prescriber_features = encode_texts(prescriber_texts, batch_size)
        pharmacy_features = encode_texts(pharmacy_texts, batch_size)
        drug_features = encode_texts(drug_texts, batch_size)

    edges = extract_edges(df, mappings)
    patient_labels = patient_stats["label"].values.astype(np.int64)

    metadata: dict[str, Any] = {
        "num_patients": len(mappings["patient"]),
        "num_prescribers": len(mappings["prescriber"]),
        "num_pharmacies": len(mappings["pharmacy"]),
        "num_drugs": len(mappings["drug"]),
        "num_edges": {k: int(v.shape[1]) for k, v in edges.items()},
        "label_distribution": {
            "overdose": int(patient_labels.sum()),
            "non_overdose": int(len(patient_labels) - patient_labels.sum()),
        },
        "feature_dim": 768,
        "encoding_model": "all-mpnet-base-v2",
        "sampling": {
            "balance_ratio": balance_ratio,
            "seed": seed if balance_ratio is not None else None,
        },
    }

    save_outputs(
        output_dir,
        patient_features.astype(np.float32),
        prescriber_features.astype(np.float32),
        pharmacy_features.astype(np.float32),
        drug_features.astype(np.float32),
        patient_labels,
        edges,
        mappings,
        metadata,
    )

    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PDMP raw data into heterogeneous graph format."
    )
    parser.add_argument("--input_dir", type=str, default="datasets/raw/pdmp")
    parser.add_argument(
        "--output_dir", type=str, default="datasets/hetgraphs/pdmp_opioid_detect/raw"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_encoding", action="store_true")
    parser.add_argument("--compute_metapaths", action="store_true")
    parser.add_argument(
        "--max_sample",
        type=int,
        default=None,
        help="Max neighbors per metapath step (None=full, set for scalability)",
    )
    parser.add_argument(
        "--balance_ratio",
        type=float,
        default=None,
        help="Ratio of majority:minority for undersampling (1.0=balanced, 2.0=1:2, None=no sampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )

    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_dir,
        args.batch_size,
        args.skip_encoding,
        args.balance_ratio,
        args.seed,
    )

    if args.compute_metapaths:
        compute_and_save_metapaths(Path(args.output_dir), args.max_sample)
