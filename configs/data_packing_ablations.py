from dataclasses import dataclass
from configs.llm_config import Tiny1M3MConfig


# ============================================================================
# Data / sequence-packing ablations — see docs/research/data_packing/plan.md
#
# D1 is a loader-level lever. The flag lives on DataConfig (where the loader
# reads it); the LLMConfig subclass below is a thin doc/marker class so
# `python train_llm.py --config Tiny1M3MDocPackConfig` selects the lever.
# The data-pipeline wiring (LLMConfig.use_doc_pack -> DataConfig.use_doc_pack)
# is a follow-up; the lever code path is exercised by the DataConfig flag.
# ============================================================================


@dataclass
class Tiny1M3MDocPackConfig(Tiny1M3MConfig):
    """D1 — DocPack: emit a `doc_id` column marking which document each token came from.

    The baseline `data/loader.py:group_texts` already concatenates documents
    and chunks at `seq_length` (implicit doc pack). This lever exposes the
    cross-doc boundary info as a per-token `doc_id` array. Step-0 == base
    (loader-level change; the column is only produced when the flag is on).
    """
    use_doc_pack: bool = True
