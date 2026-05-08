from dataclasses import dataclass, field
from typing import List

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
    version: str = "v2"

    # Columns that identify a unique trial
    trial_keys: List[str] = field(default_factory=lambda: ["set_id", "engine", "round"])

    # Feature columns — switch to FEATURE_COLS_V1 to reproduce baseline
    feature_cols: List[str] = field(default_factory=lambda: FEATURE_COLS_V2)

    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.1
    use_batch_norm: bool = True

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64      # number of trials per batch
    n_epochs: int = 100
    patience: int = 15        # early stopping patience (val loss)
    seed: int = 42

    # Temperature calibration grid
    temp_candidates: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0
    ])
