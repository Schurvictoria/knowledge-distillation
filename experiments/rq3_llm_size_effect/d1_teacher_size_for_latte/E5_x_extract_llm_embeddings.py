#!/usr/bin/env python3
"""
Generic LLM embedding extractor for LATTE teachers.

For each client in train+test:
  1. Build transaction text (using existing serialize from load_dataset)
  2. Forward pass through LLM
  3. Mean-pool last hidden state → embedding vector
  4. Save as results/{dataset}_{teacher}_llm_embeddings.npz

Usage:
  python extract_llm_embeddings.py --model google/gemma-3-4b-it --teacher gemma4b --datasets gender rosbank age
"""
import os, json, time, argparse, warnings, gc
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from run_openrouter_experiments import load_dataset

# ---- Reproducibility (seed=42) ----
import random as _random, os as _os
_SEED = 42
_random.seed(_SEED); np.random.seed(_SEED)
torch.manual_seed(_SEED); torch.cuda.manual_seed_all(_SEED)
import pytorch_lightning as _pl
_pl.seed_everything(_SEED, workers=True)
_os.environ["PYTHONHASHSEED"] = str(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)


def extract_one_dataset(model_id, teacher_short, dataset_name, use_4bit=True):
    out_dir = Path(f"results/{dataset_name}_{teacher_short}_llm_embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "llm_embeddings.npz"
    if out_file.exists():
        print(f"  Cached: {out_file}", flush=True)
        return

    print(f"\n=== {teacher_short} embeddings on {dataset_name} ===", flush=True)
    data = load_dataset(dataset_name)
    cids_train = np.load(f"embeddings/{dataset_name}/cids_train_seed42.npy")
    cids_test = np.load(f"embeddings/{dataset_name}/cids_test_seed42.npy")
    all_cids = np.concatenate([cids_train, cids_test])
    print(f"  Total clients: {len(all_cids)} (train={len(cids_train)}, test={len(cids_test)})", flush=True)

    # Load model
    print(f"  Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModel.from_pretrained(model_id, quantization_config=bnb,
                                           device_map="auto", trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                           device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"  Loaded. hidden_size={model.config.hidden_size}", flush=True)

    # Extract embeddings
    embeddings = np.zeros((len(all_cids), model.config.hidden_size), dtype=np.float32)
    cid_order = np.array(all_cids)

    t0 = time.time()
    with torch.no_grad():
        for i, cid in enumerate(all_cids):
            text = data["serialize"](int(cid))
            if not text or text == "No txns.":
                continue
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            # Forward pass — match LLM4ES convention (E2_2): mean-pool over LAST 8 LAYERS,
            # then masked mean over sequence length.
            out = model(**inp, output_hidden_states=True)
            hidden = torch.stack(out.hidden_states[-8:]).mean(0)[0]  # [seq, hidden]
            mask = inp["attention_mask"][0].float().unsqueeze(-1)
            pooled = (hidden * mask).sum(0) / mask.sum(0).clamp(min=1)
            embeddings[i] = pooled.float().cpu().numpy()

            if (i+1) % 500 == 0:
                rate = (i+1) / (time.time() - t0)
                eta_s = (len(all_cids) - i - 1) / max(rate, 0.1)
                print(f"    {i+1}/{len(all_cids)} ({rate:.1f}/s, ETA {eta_s/60:.1f}min)", flush=True)

    np.savez(out_file, embeddings=embeddings, cid_order=cid_order)
    print(f"  Saved: {out_file} {embeddings.shape}", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model ID")
    ap.add_argument("--teacher", required=True, help="Short name for output file")
    ap.add_argument("--datasets", nargs="*", default=["gender", "rosbank", "age"])
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = ap.parse_args()

    for ds in args.datasets:
        extract_one_dataset(args.model, args.teacher, ds, use_4bit=not args.no_4bit)
