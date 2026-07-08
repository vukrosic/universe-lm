"""lm-eval-harness adapter for MinimalLLM lab checkpoints.

Wraps a lab checkpoint (model.pt / checkpoint_step_*.pt) as an
EleutherAI lm-evaluation-harness model so every standard benchmark
(HellaSwag, ARC, PIQA, MMLU, GSM8K, HumanEval, ...) runs with the same
scoring code the SmolLM2 numbers were published with. This supersedes
the hand-rolled scripts in benchmarks/ for anything lm-eval covers.

Suites (two tiers — see evals/README.md):
  core   0-shot loglikelihood tasks where small models score above random.
         This is the tier the flagship "beat SmolLM2-135M" verdict uses.
  bio    MMLU biology subsets. ~25% (random) until models are much bigger;
         wired now so the curve exists from day one.
  math   GSM8K 5-shot generative. ~0 at lab scales; aspirational tier.
  code   HumanEval pass@1. Executes generated code — box-only, never on
         a machine you care about. ~0 at lab scales; aspirational tier.

Usage (on the GPU box, inside /venv/main):
  python -m evals.lm_eval_universe --checkpoint lab_runs/B-S0/model.pt \
      --suite core --out results/evals
  # smoke test: --suite core --limit 20
Results land in <out>/<run_name>/<suite>.json plus a printed table.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_eval import simple_evaluate
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

SUITES = {
    # SmolLM2-comparable tier: 0-shot, loglikelihood-scored.
    "core": dict(
        tasks=["hellaswag", "arc_easy", "arc_challenge", "piqa",
               "winogrande", "openbookqa", "boolq", "commonsense_qa"],
        num_fewshot=0,
    ),
    # Domain knowledge (biology focus). Random floor is 25%.
    "bio": dict(
        tasks=["mmlu_college_biology", "mmlu_high_school_biology",
               "mmlu_anatomy"],
        num_fewshot=0,
    ),
    # Aspirational tiers — expect ~0 below ~1B params; tracked anyway.
    "math": dict(tasks=["gsm8k"], num_fewshot=5),
    "code": dict(tasks=["humaneval"], num_fewshot=0, unsafe=True),
}


class UniverseLM(TemplateLM):
    """Minimal loglikelihood + greedy-generation LM over MinimalLLM."""

    def __init__(self, checkpoint, device=None, batch_size=8,
                 tokenizer_name="HuggingFaceTB/SmolLM2-135M"):
        super().__init__()
        from benchmarks.common import load_model_from_checkpoint

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model, self.config, _ = load_model_from_checkpoint(
            checkpoint, device=self.device, dtype=torch.bfloat16)
        # Training tokenizer is SmolLM2 (configs/dataset_config.py), not the
        # SmolLM v1 one benchmarks.common loads — load the right one here.
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = int(getattr(self.config, "max_seq_len", 2048))
        self.max_gen_toks = 256

    # --- TemplateLM required surface -------------------------------------
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def _model_logits(self, batch):
        """batch: LongTensor [B, T] -> float32 logits [B, T, V]."""
        with torch.no_grad(), torch.autocast(
                device_type=self.device.split(":")[0], dtype=torch.bfloat16,
                enabled=self.device.startswith("cuda")):
            out = self.model(batch)
        logits = out[0] if isinstance(out, tuple) else out
        return logits.float()

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        res = []
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start:start + self.batch_size]
            inps, cont_lens, targets = [], [], []
            for _, ctx_enc, cont_enc in chunk:
                whole = (ctx_enc + cont_enc)[-(self.max_length + 1):]
                # keep the continuation intact; ctx is what gets clipped
                cont = cont_enc[-self.max_length:]
                inps.append(torch.tensor(whole[:-1], dtype=torch.long))
                cont_lens.append(len(cont))
                targets.append(torch.tensor(whole[1:], dtype=torch.long))
            maxlen = max(t.numel() for t in inps)
            batch = torch.full((len(inps), maxlen), self.eot_token_id or 0,
                               dtype=torch.long)
            for i, t in enumerate(inps):
                batch[i, :t.numel()] = t
            logits = self._model_logits(batch.to(self.device))
            for i, (t, cl, tgt) in enumerate(zip(inps, cont_lens, targets)):
                L = t.numel()
                lg = logits[i, L - cl:L]                    # [cl, V]
                tg = tgt[-cl:].to(self.device)              # [cl]
                lp = F.log_softmax(lg, dim=-1)
                ll = lp.gather(1, tg.unsqueeze(1)).sum().item()
                greedy = bool((lg.argmax(dim=-1) == tg).all().item())
                res.append((ll, greedy))
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            (string,) = req.args
            windows = list(map(make_disjoint_window, get_rolling_token_windows(
                token_list=self.tok_encode(string),
                prefix_token=self.prefix_token_id,
                max_seq_len=self.max_length, context_len=1)))
            wins = [(None, ctx, cont) for ctx, cont in windows]
            out.append(sum(ll for ll, _ in self._loglikelihood_tokens(wins)))
        return out

    def generate_until(self, requests, disable_tqdm=False):
        outs = []
        for req in requests:
            ctx, gen_kwargs = req.args
            until = list(gen_kwargs.get("until") or [])
            max_toks = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            ids = self.tok_encode(ctx)[-(self.max_length - max_toks):]
            ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            new = []
            for _ in range(max_toks):
                logits = self._model_logits(ids[:, -self.max_length:])
                nxt = int(logits[0, -1].argmax().item())
                if nxt == self.eot_token_id:
                    break
                new.append(nxt)
                ids = torch.cat(
                    [ids, torch.tensor([[nxt]], device=self.device)], dim=1)
                text = self.tokenizer.decode(new)
                if any(s in text for s in until):
                    break
            text = self.tokenizer.decode(new)
            for s in until:  # trim at the first stop sequence
                if s in text:
                    text = text.split(s)[0]
            outs.append(text)
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="core", choices=sorted(SUITES))
    ap.add_argument("--limit", type=int, default=None,
                    help="samples per task (smoke tests)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/evals")
    args = ap.parse_args()

    suite = SUITES[args.suite]
    lm = UniverseLM(args.checkpoint, device=args.device,
                    batch_size=args.batch_size)
    t0 = time.time()
    results = simple_evaluate(
        model=lm, tasks=suite["tasks"], num_fewshot=suite["num_fewshot"],
        limit=args.limit, confirm_run_unsafe_code=suite.get("unsafe", False))

    run_name = Path(args.checkpoint).parent.name or "run"
    outdir = Path(args.out) / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint), "run_name": run_name,
        "suite": args.suite, "limit": args.limit,
        "num_fewshot": suite["num_fewshot"],
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "results": results["results"],
        "versions": results.get("versions", {}),
    }
    out_path = outdir / f"{args.suite}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n== {run_name} / {args.suite} "
          f"(limit={args.limit}, {payload['elapsed_min']} min) ==")
    for task, m in results["results"].items():
        keys = [k for k in ("acc_norm,none", "acc,none", "exact_match,strict-match",
                            "pass@1,create_test") if k in m]
        vals = "  ".join(f"{k.split(',')[0]}={m[k]:.4f}" for k in keys)
        print(f"  {task:28s} {vals}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
