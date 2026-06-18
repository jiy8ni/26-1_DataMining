from dataclasses import dataclass, field
from typing import List, Optional

# Columns that are right-skewed counts/prices — log1p-transformed in v3+
LOG_TRANSFORM_COLS: List[str] = [
    "text_length", "image_count", "table_count", "list_item_count",
    "paragraph_count", "section_count", "jsonld_field_count",
    "explicit_number_count", "ambiguous_term_count",
    "price_krw", "skin_type_targets_count",
    "active_ingredient_count", "claim_keyword_count", "texture_keyword_count",
    "no_list_count", "cosmetic_cert_count",
    "Q4_social_proof_count", "Q9_external_authority_count",
]

# v1: all 24 numeric features (includes high-missingness columns)
FEATURE_COLS_V1: List[str] = [
    "text_length", "image_count", "table_count", "list_item_count",
    "paragraph_count", "section_count", "jsonld_field_count",
    "explicit_number_count", "ambiguous_term_count", "numeric_specificity_ratio",
    "price_krw", "skin_type_targets_count", "ph_value",
    "active_ingredient_count", "claim_keyword_count", "texture_keyword_count",
    "no_list_count", "cosmetic_cert_count", "volume_ml",
    "aggregate_rating_value", "aggregate_rating_count",
    "T7_eat_score", "Q4_social_proof_count", "Q9_external_authority_count",
]

# v2: removed columns with >20% missingness
#   ph_value              98.2% missing
#   aggregate_rating_value 38.8% missing
#   aggregate_rating_count 38.5% missing
#   volume_ml             19.9% missing
FEATURE_COLS_V2: List[str] = [
    "text_length", "image_count", "table_count", "list_item_count",
    "paragraph_count", "section_count", "jsonld_field_count",
    "explicit_number_count", "ambiguous_term_count", "numeric_specificity_ratio",
    "price_krw", "skin_type_targets_count",
    "active_ingredient_count", "claim_keyword_count", "texture_keyword_count",
    "no_list_count", "cosmetic_cert_count",
    "T7_eat_score", "Q4_social_proof_count", "Q9_external_authority_count",
]

FEATURE_COLS_V3: List[str] = [
    "text_length", "image_count", "table_count", "list_item_count",
    "paragraph_count", "section_count", "jsonld_field_count",
    "explicit_number_count", "ambiguous_term_count", "numeric_specificity_ratio",
    "price_krw", "skin_type_targets_count",
    "active_ingredient_count", "claim_keyword_count", "texture_keyword_count",
    "no_list_count", "cosmetic_cert_count",
    "T7_eat_score", "Q4_social_proof_count", "Q9_external_authority_count",
]


@dataclass
class Config:
    # Paths (relative to repo root)
    data_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    ckpt_dir: str = "checkpoints"

    # Evaluation protocol
    # "step1" = seen-item (trial-level split)
    # "step2" = unseen-item (brand-level holdout)
    protocol: str = "step2"

    # Model version tag — used in wandb run name and checkpoint filename
    version: str = "v3"

    # K-fold cross-validation (brand-level holdout)
    n_folds: int = 5

    # Engine filter: "openai" | "anthropic" | None (use both — not recommended)
    engine_filter: Optional[str] = "openai"

    # Columns that identify a unique trial
    trial_keys: List[str] = field(default_factory=lambda: ["set_id", "engine", "round"])

    # Feature columns — switch to FEATURE_COLS_V1 to reproduce baseline.
    # V3 = structural features; text/image embeddings are appended automatically
    # at runtime (PCA-reduced txt_pca_*/img_pca_*) when use_semantic_features=True.
    feature_cols: List[str] = field(default_factory=lambda: FEATURE_COLS_V3)

    # Columns to apply log1p before StandardScaler (v3+); empty list = no transform
    log_transform_cols: List[str] = field(default_factory=lambda: LOG_TRANSFORM_COLS)

    # Whether to append normalized sku_pos (presentation order) as an extra feature
    use_position_feature: bool = True

    # --- Semantic (text/image embedding) features ---------------------------
    # When enabled, raw per-URL embeddings are merged by resolved_url, reduced
    # with PCA fitted on the training fold only (no leakage), and the reduced
    # dims (txt_pca_*, img_pca_*) are appended to the feature matrix and scaled
    # alongside the structural features. Set False to reproduce the structural
    # baseline.
    use_semantic_features: bool = True
    text_emb_path: str = "data/processed/item_text_emb.parquet"
    image_emb_path: str = "data/processed/item_image_emb.parquet"
    text_pca_dim: int = 16    # reduce 1024-d BGE-M3 text embedding -> this many dims
    image_pca_dim: int = 8    # reduce 512-d CLIP image embedding  -> this many dims

    # Model architecture
    # NOTE: with only ~259 unique training items, the original [128,64,32] over-fits.
    # Defaults below reflect the brand-CV-selected, more-regularized MLP; override
    # via artifacts/tuning/mlp_best_params.json (loaded by mlp/train.py when present).
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3
    use_batch_norm: bool = True

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64      # number of trials per batch
    n_epochs: int = 100
    patience: int = 15        # early stopping patience (val loss)
    seed: int = 42

    # Seed ensembling for the vanilla single-split trainers: each trainer fits
    # n_seeds models on the SAME split (seed = cfg.seed + i) and averages raw
    # scores (+ averages the per-seed calibration temperature). Set 1 to disable.
    n_seeds: int = 5

    # Brand-CV-selected hyperparameters are written here by src/tune/*.py and
    # loaded by the vanilla trainers when the file exists.
    tuning_dir: str = "artifacts/tuning"
    # Per-model raw scores (val/test) are dumped here for src/blend.py.
    preds_dir: str = "artifacts/preds"

    # PL-fitted labels for hybrid loss
    pl_labels_path: str = "data/processed/pl_labels_step2_openai.csv"
    lambda_mse: float = 0.5   # weight on MSE(score, pl_theta)

    # Temperature calibration grid
    temp_candidates: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0
    ])
