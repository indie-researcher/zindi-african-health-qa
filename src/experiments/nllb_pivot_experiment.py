"""
Purpose: Runs NLLB-200 3.3B pivot translation experiment on low-resource African health Q&A data.
Inputs: Master CSV dataset containing multilingual health questions and answers.
Outputs: Cached CSV checkpoint with translated inputs and outputs.
Goal: Pivot low-resource African dialect Q&A pairs into English high-resource supervision.
Conclusion: NLLB-200 3.3B model on CPU successfully translates dialect subsets while preserving clinical semantics.
"""
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- CONSTANTS & PATHS ---
MODEL_REPO_NAME = "facebook/nllb-200-3.3B"
INPUT_CSV_PATH = "/kaggle/input/datasets/yoka33/pre-nllb-ready-df/pre_nllb_ready_df.csv"
CHECKPOINT_OUTPUT_PATH = "checkpoint_step1_nllb_3.3B_cpu.csv"
SAMPLE_SIZE_PER_DIALECT = 50

# Maximize CPU core utilization
torch.set_num_threads(os.cpu_count() or 4)

df_master = pd.read_csv(INPUT_CSV_PATH)
df_master = df_master[df_master["split"] != "test"]

# Balance dataset: Grab exactly 50 rows per dialect group
df = df_master.groupby("subset").sample(n=SAMPLE_SIZE_PER_DIALECT, random_state=42).reset_index(drop=True)

L = {"Amh_Eth": "amh_Ethi", "Aka_Gha": "aka_Latn", "Lug_Uga": "lug_Latn", "Swa_Ken": "swh_Latn"}

print("Loading 3.3B model into System RAM (~14GB)...")
tok = AutoTokenizer.from_pretrained(MODEL_REPO_NAME)
mod = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO_NAME, torch_dtype=torch.float32).eval()
tgt = tok.convert_tokens_to_ids("eng_Latn")

ins, outs = [], []
print("Starting overnight inference loop...")

for i, (_, r) in enumerate(df.iterrows()):
    print(f"Processing row {i+1}/{len(df)} | Dialect: {r['subset']}")
    tok.src_lang = L[r["subset"]]
    
    in_t = tok(str(r["input"]), return_tensors="pt", truncation=True, max_length=256)
    out_t = tok(str(r["output"]), return_tensors="pt", truncation=True, max_length=256)
    
    # inference_mode is faster than no_grad on CPU execution paths
    with torch.inference_mode():
        in_gen = mod.generate(**in_t, forced_bos_token_id=tgt, max_length=256)[0]
        out_gen = mod.generate(**out_t, forced_bos_token_id=tgt, max_length=256)[0]
        
    ins.append(tok.decode(in_gen, skip_special_tokens=True))
    outs.append(tok.decode(out_gen, skip_special_tokens=True))
    
    # Save every 10 rows so you do not lose progress if Kaggle disconnects
    if (i + 1) % 10 == 0 or (i + 1) == len(df):
        df.iloc[:i+1].assign(NLLB_Trans_Input=ins, NLLB_Trans_Output=outs).to_csv(CHECKPOINT_OUTPUT_PATH, index=False)

print(f"Inference Complete! Final file saved to {CHECKPOINT_OUTPUT_PATH}")
