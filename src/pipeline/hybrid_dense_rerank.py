"""Hybrid retrieval: BM25 + dense embeddings fused with reciprocal-rank fusion
(per-subset k1/b/alpha), top-12 candidates reranked by a cross-encoder
(BAAI/bge-reranker-v2-m3); the best-reranked training answer is returned.
No generation. Kaggle paths; needs the precomputed embeddings dataset.
Note: the validation holdout is added to the retrieval pool, so this script
is not a clean validation setup. Script dated 2026-06-21."""
import os, re, warnings, time
import numpy as np
import pandas as pd
import torch

print("⚙️ Booting high-performance final runtime...")
os.system("pip install sentence_transformers rouge_score rank_bm25 --quiet")

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

warnings.filterwarnings('ignore')
start_time = time.time()

TOP_K_RECALL = 12  # Set to optimal speed-accuracy threshold
EMB_DIR = '/kaggle/input/embeddings-overnight-qa-afr/embeddings_overnight'
BASE = '/kaggle/input/datasets/yoka33/qna-african-languages-challenge'
WORK_DIR = '/kaggle/working'

raw_train = pd.read_csv(f'{BASE}/optimized_train.csv')
raw_val   = pd.read_csv(f'{BASE}/optimized_val.csv')
raw_test  = pd.read_csv(f'{BASE}/Test.csv')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device)

# --- Lexical Heuristics and Tuning configurations ---
AMHARIC_DICT = {
    "ኤችአይቪ": ("ኤችአይቪ", "HIV"), "በኤችአይቪ": ("ኤችአይቪ", "HIV"), "ለኤችአይቪ": ("ኤችአይቪ", "HIV"),
    "ቂጥኝ": ("ቂጥኝ", "Syphilis"), "ትሪኮሞኒያስስ": ("ትሪኮሞኒያስስ", "Trichomoniasis"), 
    "ክላሚዲያ": ("ክላሚዲያ", "Chlamydia"), "ሄርፒስ": ("ሄርፒስ", "Herpes"), "ጨብጥ": ("ጨብጥ", "Gonorrhea"), 
    "ካንሰር": ("ካንሰር", "Cancer"), "በሽታ": ("በሽታ", "Disease"), "ኢንፌክሽን": ("ኢንፌክሽን", "Infection")
}

CHAMPION_TUNING = {
    "Aka_Gha": {"model": "BGE-M3", "k1": 1.6, "b": 0.75, "alpha": 1.0, "side": "corpus_questions"},
    "Amh_Eth": {"model": "BGE-M3", "k1": 1.2, "b": 0.30, "alpha": 0.02,  "side": "corpus_questions"},
    "Eng_Eth": {"model": "BGE-M3", "k1": 1.2, "b": 0.75, "alpha": 10.0, "side": "corpus_questions"},
    "Eng_Gha": {"model": "Multilingual-E5-Base", "k1": 1.6, "b": 0.75, "alpha": 5.0, "side": "corpus_questions"},
    "Eng_Ken": {"model": "Multilingual-E5-Base", "k1": 1.2, "b": 0.75, "alpha": 10.0, "side": "corpus_questions"},
    "Eng_Uga": {"model": "Multilingual-E5-Base", "k1": 1.2, "b": 0.55, "alpha": 10.0, "side": "corpus_questions"},
    "Lug_Uga": {"model": "Multilingual-E5-Base", "k1": 2.0, "b": 0.75, "alpha": 12.0, "side": "corpus_questions"},
    "Swa_Ken": {"model": "BGE-M3", "k1": 1.2, "b": 0.30, "alpha": 10.0, "side": "corpus_questions"}
}

def clean_tok_universal(text, subset):
    t = str(text).lower()
    if subset == "Amh_Eth":
        return [AMHARIC_DICT[w][0] if w in AMHARIC_DICT else w for w in re.sub(r'[፡።፣፤፥፦፤፧፨]+', ' ', t).split() if w]
    return [w for w in re.sub(r'[?!.,;:\'"\-]+', ' ', t).split() if len(w) > 2]

def load_emb_matrix(model_name, dataset_name, subset):
    e, i = f"{EMB_DIR}/{model_name}__{dataset_name}__{subset}.npy", f"{EMB_DIR}/{model_name}__{dataset_name}__{subset}_ids.csv"
    if not (os.path.exists(e) and os.path.exists(i)): return None, None
    return np.load(e), pd.read_csv(i)

submission_frames = []

for subset, cfg in CHAMPION_TUNING.items():
    sub_pool = raw_train[raw_train['subset'] == subset].reset_index(drop=True)
    sub_test = raw_test[raw_test['subset'] == subset].copy().reset_index(drop=True)
    if len(sub_test) == 0: continue
    
    # Pool expansion: the validation holdout is appended to the retrieval pool
    val_holdout = raw_val[raw_val['subset'] == subset]
    extra_rows = val_holdout[['input', 'output']]
    
    lexical_pool = pd.concat([sub_pool, extra_rows], ignore_index=True).drop_duplicates(subset=['input']).reset_index(drop=True)
    
    t_emb, t_ids = load_emb_matrix(cfg['model'], 'test_queries', subset)
    p_emb, p_ids = load_emb_matrix(cfg['model'], cfg['side'], subset)
    if t_emb is None or p_emb is None:
        sub_test['prediction'] = "Please visit a medical professional."
        submission_frames.append(sub_test); continue

    valid_t_idx = t_ids[t_ids['ID'].isin(sub_test['ID'])].index.tolist()
    t_emb = t_emb[valid_t_idx]
    sub_train_dense = sub_pool.set_index('ID').loc[p_ids['ID']].reset_index()

    bm25_opt = BM25Okapi([clean_tok_universal(x, subset) for x in lexical_pool['input']], k1=cfg['k1'], b=cfg['b'])
    dense_ranks = np.argsort(np.argsort(-(t_emb @ p_emb.T), axis=1), axis=1)
    
    sparse_scores = np.zeros((len(sub_test), len(lexical_pool)))
    for idx, row in sub_test.iterrows():
        q_toks = clean_tok_universal(row['input'], subset)
        if q_toks: sparse_scores[idx] = bm25_opt.get_scores(q_toks)
            
    padded_dense_ranks = np.full((len(sub_test), len(lexical_pool)), len(lexical_pool))
    padded_dense_ranks[:, :len(sub_train_dense)] = dense_ranks
    sparse_ranks = np.argsort(np.argsort(-sparse_scores, axis=1), axis=1)
    
    rrf_matrix = (1.0 / (60.0 + sparse_ranks)) + (cfg['alpha'] * (1.0 / (60.0 + padded_dense_ranks)))
    topK_indices_per_query = np.argsort(-rrf_matrix, axis=1)[:, :TOP_K_RECALL]
    
    cross_pairs = []
    for q_idx, test_row in sub_test.iterrows():
        for local_rank_idx in topK_indices_per_query[q_idx]:
            cross_pairs.append([test_row['input'], lexical_pool.iloc[local_rank_idx]['input']])
            
    cross_scores = reranker.predict(cross_pairs, batch_size=256, show_progress_bar=False)
    reshaped_scores = cross_scores.reshape(len(sub_test), TOP_K_RECALL)
    final_best_local_indices = np.argmax(reshaped_scores, axis=1)
    
    test_preds = []
    for q_idx in range(len(sub_test)):
        chosen_pool_idx = topK_indices_per_query[q_idx][final_best_local_indices[q_idx]]
        test_preds.append(lexical_pool.iloc[chosen_pool_idx]['output'])
        
    sub_test['prediction'] = test_preds
    submission_frames.append(sub_test)

# ── 3. EXPORT FINAL SUBMISSION WITH TARGET OVERRIDES ────────────────────────
master_submission = pd.concat(submission_frames)
final_frame = raw_test[['ID']].merge(master_submission[['ID', 'prediction']], on='ID', how='left')

submission = pd.DataFrame({
    "ID": final_frame["ID"],
    "TargetRLF1": final_frame["prediction"].fillna("Please visit a health facility."),
    "TargetR1F1": final_frame["prediction"].fillna("Please visit a health facility."),
    "TargetLLM":  final_frame["prediction"].fillna("Please visit a health facility.")
})

submission.to_csv(f"{WORK_DIR}/submission.csv", index=False)
print(f"🏁 Finished! Submission saved to: {WORK_DIR}/submission.csv")
