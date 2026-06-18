"""Path isolation for the handoff folder.

The shared files copied from the original ``src/`` (config.py, data.py, ...) are
kept BYTE-FOR-BYTE UNMODIFIED so they stay in sync with the original project.
Their default paths in ``Config`` are repo-root-relative ("data/processed",
"artifacts/preds", ...). This helper rewrites those defaults so that:

  * input DATA is read (read-only) from the ORIGINAL project's data/ folder, and
  * all OUTPUTS (tuning JSON, prediction .npz) are written INSIDE this handoff
    folder — never touching the original artifacts/.

Every new tuner / trainer / blend script calls ``configure_paths(cfg)`` right
after ``cfg = Config()`` so it can be launched from any working directory.
"""
import os

HANDOFF_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HANDOFF_ROOT)


def configure_paths(cfg):
    """Point data reads at the original project; outputs at this handoff folder."""
    data_proc = os.path.join(PROJECT_ROOT, "data", "processed")

    # --- inputs: read the original project's data (read-only) ---------------
    cfg.data_dir        = data_proc
    cfg.splits_dir      = os.path.join(PROJECT_ROOT, "data", "splits")
    cfg.text_emb_path   = os.path.join(data_proc, "item_text_emb.parquet")
    cfg.image_emb_path  = os.path.join(data_proc, "item_image_emb.parquet")
    cfg.pl_labels_path  = os.path.join(data_proc, "pl_labels_step2_openai.csv")

    # --- outputs: write only inside the handoff folder ----------------------
    cfg.tuning_dir = os.path.join(HANDOFF_ROOT, "artifacts", "tuning")
    cfg.preds_dir  = os.path.join(HANDOFF_ROOT, "artifacts", "preds")
    return cfg
