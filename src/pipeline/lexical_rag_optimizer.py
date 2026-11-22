"""
Purpose: Optimizes RAG retrieval thresholds, clinical dictionaries, and lexical alignment via Bayesian search.
Inputs: Training, validation, and test CSV datasets.
Outputs: Optimized RAG submission CSV, validation logs, and best threshold JSON files.
"""
import re
import time
import json
import random
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rouge_score import rouge_scorer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Per-subset expected answer length (word count) — used for template selection
LENGTH_TARGETS = {
    "Amh_Eth": 20,
    "Eng_Eth": 25,
    "Eng_Gha": 72,
    "Eng_Ken": 65,
    "Eng_Uga": 80,
    "Lug_Uga": 68,
    "Swa_Ken": 68,
    "Aka_Gha": 100,
}

# Clinical dictionary — expanded with dialect-specific terms
# Used for: (a) cluster routing, (b) answer template selection,
#           (c) TF-IDF vocabulary boosting
CLINICAL_DICTS = {
    "HIV GENERAL": [
        "hiv", "aids", "arv", "art", "antiretroviral", "viral load", "cd4",
        "undetectable", "positive", "negative", "test", "treatment",
        # Kiswahili
        "ukimwi", "vvu", "damu", "virusi",
        # Luganda
        "siriimu", "akawuka", "obulwadde", "omusaayi",
        # Amharic
        "ኤችአይቪ", "ኤድስ", "ደም", "ህክምና",
        # Akan
        "esikyibea", "mogya", "yadee",
    ],
    "HIV TRANSMISSION": [
        "transmit", "spread", "mosquito", "saliva", "kissing", "touch",
        "washing", "casual", "contact", "blood", "needle", "mother",
        "ukimwi", "maambukizi", "kutambaza",
        "siriimu", "okwambukiza",
        "ኤችአይቪ", "መተላለፍ",
    ],
    "CULTURAL MYTHS": [
        "prayer", "pray", "faith", "holy water", "fasting", "fast",
        "pastor", "church", "mosque", "traditional", "healer", "herbs",
        "garlic", "lemon", "witchcraft", "cure", "heal",
        "dawa ya kienyeji", "miti shamba",
        "ሎሚ", "ነጭ ሽንኩርት", "ጸሎት",
        "odunsini", "nnuaba",
    ],
    "ADOLESCENT SRH": [
        "puberty", "adolescent", "teen", "young", "peer", "pressure",
        "sexual", "reproductive", "education", "access", "consent",
        "menstrual", "period", "pregnancy", "contraception",
        "mimba", "hedhi", "ujana",
        "obuvubuka", "omukazi", "omwana",
        "ቅድሚያ", "ወጣት", "ጉርምስና",
        "mmabun", "abofra",
    ],
    "SYPHILIS": [
        "syphilis", "chancre", "rash", "stages", "primary", "secondary",
        "latent", "tertiary", "penicillin", "vdrl",
        "kaswende", "ቂጥኝ", "kabotongo",
    ],
    "CHLAMYDIA": [
        "chlamydia", "discharge", "burning", "asymptomatic", "pelvic",
        "infertility", "antibiotic", "silent",
        "ክላሚዲያ", "kilamida",
    ],
    "GONORRHEA": [
        "gonorrhea", "gonorrhoea", "clap", "discharge", "burning",
        "neisseria", "antibiotic", "resistance",
        "kisonono", "nzege", "ጨብጥ", "babaso",
    ],
    "HERPES": [
        "herpes", "hsv", "blister", "sore", "outbreak", "acyclovir",
        "lifelong", "antiviral", "genital",
        "ሄርፒስ",
    ],
    "HPV": [
        "hpv", "wart", "cervical", "cancer", "vaccine", "papillomavirus",
        "screening", "pap",
        "ኤችፒቪ", "ኪንታሮት",
    ],
    "HEPATITIS": [
        "hepatitis", "hbv", "liver", "jaundice", "vaccine", "chronic",
        "cirrhosis", "antiviral",
        "homa ya ini", "ሄፓታይተስ",
    ],
    "TRICHOMONIASIS": [
        "trichomoniasis", "trich", "parasite", "discharge", "metronidazole",
        "ትሪኮሞኒያሲስ",
    ],
    "PEP PREP": [
        "pep", "prep", "post-exposure", "pre-exposure", "72 hours",
        "prophylaxis", "truvada", "emergency",
    ],
    "GENERAL STIS": [
        "sti", "std", "sexually transmitted", "infection", "venereal",
        "ndwara", "yareza",
    ],
}

# Myth signals for override detection
MYTH_SIGNALS = frozenset([
    "prayer", "pray", "holy water", "fasting", "traditional healer",
    "herbs", "herbal", "garlic", "lemon", "witchcraft", "faith healing",
    "pastor", "church cure", "mosque cure",
    "dawa ya kienyeji", "ሎሚ", "ጸሎት", "odunsini",
])

# Template answers per cluster — calibrated to subset length targets
CLUSTER_TEMPLATES = {
    "HIV GENERAL": (
        "HIV is a virus that attacks the immune system. "
        "With antiretroviral treatment (ART) taken daily, people living with HIV "
        "can live long and healthy lives. "
        "Get tested at your nearest health facility. "
        "An undetectable viral load means HIV cannot be transmitted to partners."
    ),
    "HIV TRANSMISSION": (
        "HIV is transmitted through blood, sexual fluids, and breast milk. "
        "It is NOT transmitted through mosquito bites, saliva, kissing, "
        "casual contact, or sharing food. "
        "Use condoms consistently and get tested regularly to protect yourself and your partner."
    ),
    "CULTURAL MYTHS": (
        "No. This condition requires medical treatment. "
        "While faith and spiritual practices can provide comfort and strength, "
        "they cannot cure a viral or bacterial infection. "
        "Please visit your nearest health facility as soon as possible for "
        "proper testing and treatment."
    ),
    "ADOLESCENT SRH": (
        "Young people have the right to access confidential reproductive health services. "
        "You can visit a youth-friendly health facility for information on "
        "contraception, STI prevention, puberty, and relationships. "
        "Services are confidential and non-judgmental."
    ),
    "SYPHILIS": (
        "Syphilis is a bacterial infection that progresses through stages: "
        "primary, secondary, latent, and tertiary. "
        "It is curable with antibiotics, especially in early stages. "
        "Untreated syphilis causes serious complications. "
        "Visit a health facility for testing, treatment, and partner notification."
    ),
    "CHLAMYDIA": (
        "Chlamydia is a common bacterial infection that often causes no symptoms. "
        "It is curable with antibiotics. Untreated chlamydia can lead to infertility. "
        "Get tested at a health facility and notify your partner for treatment."
    ),
    "GONORRHEA": (
        "Gonorrhea is a bacterial infection causing discharge and burning when urinating. "
        "It is treatable with antibiotics, though resistance is increasing. "
        "There is no vaccine for gonorrhea. "
        "Visit a health facility and notify your sexual partners."
    ),
    "HERPES": (
        "Herpes is a lifelong viral infection caused by HSV. "
        "It causes recurring blisters and sores. "
        "Antiviral medication reduces outbreak frequency and transmission risk. "
        "Herpes cannot be cured but can be managed effectively."
    ),
    "HPV": (
        "HPV is a common sexually transmitted infection. "
        "Some types cause genital warts and cervical cancer. "
        "An HPV vaccine is available and highly effective. "
        "Regular cervical screening detects problems early. "
        "Condoms reduce but do not eliminate HPV transmission."
    ),
    "HEPATITIS": (
        "Hepatitis B is a viral infection affecting the liver. "
        "It can be prevented by vaccination. "
        "Chronic hepatitis B can cause liver damage and cancer. "
        "Treatment with antiviral medication is available. "
        "Get tested and vaccinated at your nearest health facility."
    ),
    "TRICHOMONIASIS": (
        "Trichomoniasis is a curable parasitic infection. "
        "It increases HIV risk. "
        "Both partners must be treated simultaneously to prevent reinfection. "
        "Visit a health facility for testing and antibiotic treatment."
    ),
    "PEP PREP": (
        "PEP must be started within 72 hours of potential HIV exposure — the sooner the better. "
        "It is a 28-day course of medication. "
        "PrEP is taken before ongoing exposure to prevent HIV infection. "
        "Visit a health facility immediately for PEP."
    ),
    "GENERAL STIS": (
        "Sexually transmitted infections often cause no symptoms. "
        "Get tested at a health facility if you are sexually active. "
        "Use condoms to reduce risk. "
        "Inform your partner if you are diagnosed with an STI."
    ),
    "FALLBACK": (
        "Please visit your nearest health facility for advice and treatment "
        "regarding this health concern."
    ),
}


def get_tokenizer(subset: str):
    if subset == "Amh_Eth":
        return lambda text: [text[i:i+3] for i in range(len(text)-2)]
    elif subset == "Aka_Gha":
        def akan_tok(text):
            words = text.lower().split()
            chars = [text[i:i+2] for i in range(len(text)-1)]
            return words + chars
        return akan_tok
    elif subset in ("Lug_Uga", "Swa_Ken"):
        def bantu_tok(text):
            return [
                re.sub(r'[?!.,;:]+$', '', w).lower()
                for w in text.lower().split()
                if len(w) > 2
            ]
        return bantu_tok
    else:
        def eng_tok(text):
            words = text.lower().split()
            unigrams = words
            bigrams  = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            return unigrams + bigrams
        return eng_tok


class SubsetIndex:
    def __init__(self, subset: str, df: pd.DataFrame):
        self.subset    = subset
        self.tokenizer = get_tokenizer(subset)
        self.questions = df['input'].astype(str).tolist()
        self.answers   = df['output'].astype(str).tolist()
        self.ids       = df['ID'].tolist() if 'ID' in df.columns else list(range(len(df)))
        self.n         = len(df)

        q_tokens = [self.tokenizer(q) for q in self.questions]
        self.bm25_q = BM25Okapi(q_tokens)

        a_tokens = [self.tokenizer(a) for a in self.answers]
        self.bm25_a = BM25Okapi(a_tokens)

        analyzer = 'char_wb' if subset in ('Amh_Eth', 'Aka_Gha') else 'word'
        ngram    = (2, 4)     if subset in ('Amh_Eth', 'Aka_Gha') else (1, 2)
        self.tfidf = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram,
            max_features=8000,
            sublinear_tf=True,
        )
        self.X_answers = self.tfidf.fit_transform(self.answers)

    def query(self, question: str, top_k: int = 5):
        tokens = self.tokenizer(question)
        scores = self.bm25_q.get_scores(tokens)
        max_score = scores.max()
        if max_score == 0:
            return []
        norm_scores = scores / max_score
        top_idx = np.argsort(norm_scores)[-top_k:][::-1]
        return [
            {
                "answer":   self.answers[i],
                "question": self.questions[i],
                "score":    float(norm_scores[i]),
                "idx":      i,
            }
            for i in top_idx
            if norm_scores[i] > 0
        ]

    def query_by_answer_vocab(self, question: str, top_k: int = 3):
        q_vec = self.tfidf.transform([question])
        sims  = cosine_similarity(q_vec, self.X_answers).flatten()
        top_i = sims.argsort()[-top_k:][::-1]
        return [
            {"answer": self.answers[i], "score": float(sims[i])}
            for i in top_i if sims[i] > 0
        ]


def build_indexes(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    combined = pd.concat([train_df, val_df], ignore_index=True)
    indexes  = {}
    for subset in combined['subset'].unique():
        sub = combined[combined['subset'] == subset].reset_index(drop=True)
        print(f"  Building index: {subset} ({len(sub)} rows)...")
        indexes[subset] = SubsetIndex(subset, sub)
    return indexes


def classify_cluster(text: str) -> str:
    t = str(text).lower()
    if any(s in t for s in MYTH_SIGNALS):
        return "CULTURAL MYTHS"
    sorted_terms = sorted(
        [(term, cat)
         for cat, terms in CLINICAL_DICTS.items()
         for term in terms],
        key=lambda x: -len(x[0])
    )
    for term, cat in sorted_terms:
        if term in t:
            return cat
    return "FALLBACK"


def predict_single(
    question:       str,
    subset:         str,
    index:          SubsetIndex,
    thresholds:     dict,
    n_anchors:      int = 3,
) -> dict:
    top_results = index.query(question, top_k=max(n_anchors, 5))

    if not top_results:
        cluster  = classify_cluster(question)
        template = CLUSTER_TEMPLATES.get(cluster, CLUSTER_TEMPLATES["FALLBACK"])
        return {"prediction": template, "strategy": "TEMPLATE_NO_RESULTS",
                "score": 0.0, "cluster": cluster}

    top_score    = top_results[0]["score"]
    top_answer   = top_results[0]["answer"]
    cluster      = classify_cluster(question)

    copy_thresh   = thresholds.get(f"{subset}_copy",   0.85)
    anchor_thresh = thresholds.get(f"{subset}_anchor", 0.50)

    if top_score >= copy_thresh:
        return {
            "prediction": top_answer,
            "strategy":   "DIRECT_COPY",
            "score":      top_score,
            "cluster":    cluster,
        }

    if top_score >= anchor_thresh:
        anchors = [r["answer"] for r in top_results[:n_anchors]]
        vocab_matches = index.query_by_answer_vocab(question, top_k=2)
        vocab_answers = [m["answer"] for m in vocab_matches if m["score"] > 0.10]

        seen    = set()
        context = []
        for a in anchors + vocab_answers:
            key = a[:50]
            if key not in seen:
                seen.add(key)
                context.append(a)
            if len(context) >= n_anchors:
                break

        return {
            "prediction": top_answer,
            "strategy":   "ANCHOR_COPY",
            "score":      top_score,
            "cluster":    cluster,
            "anchors":    context,
        }

    template = CLUSTER_TEMPLATES.get(cluster, CLUSTER_TEMPLATES["FALLBACK"])
    return {
        "prediction": template,
        "strategy":   "TEMPLATE",
        "score":      top_score,
        "cluster":    cluster,
    }


def predict_batch(
    df:         pd.DataFrame,
    indexes:    dict,
    thresholds: dict,
    n_anchors:  int = 3,
) -> list:
    results = []
    for _, row in df.iterrows():
        subset = row['subset']
        idx    = indexes.get(subset)
        if idx is None:
            results.append({
                "prediction": CLUSTER_TEMPLATES["FALLBACK"],
                "strategy":   "NO_INDEX",
                "score":      0.0,
                "cluster":    "FALLBACK",
            })
            continue
        r = predict_single(
            question   = str(row['input']),
            subset     = subset,
            index      = idx,
            thresholds = thresholds,
            n_anchors  = n_anchors,
        )
        results.append(r)
    return results


_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

def evaluate_predictions(predictions: list, references: list, subsets: list) -> dict:
    results = defaultdict(list)
    for pred, ref, subset in zip(predictions, references, subsets):
        scores = _SCORER.score(str(ref), str(pred))
        results[subset].append({
            'rouge1': scores['rouge1'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
        })
        results['_all'].append({
            'rouge1': scores['rouge1'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
        })

    summary = {}
    for subset, rows in results.items():
        r1 = np.mean([r['rouge1'] for r in rows])
        rL = np.mean([r['rougeL'] for r in rows])
        summary[subset] = {
            'rouge1': round(r1, 4),
            'rougeL': round(rL, 4),
            'blended': round(0.5 * r1 + 0.5 * rL, 4),
            'n':       len(rows),
        }
    return summary


class BayesianThresholdOptimizer:
    def __init__(self, subsets: list, val_df: pd.DataFrame, indexes: dict):
        self.subsets  = subsets
        self.val_df   = val_df
        self.indexes  = indexes
        self.history  = []
        self.best_score      = -np.inf
        self.best_thresholds = self._default_thresholds()

    def _default_thresholds(self) -> dict:
        t = {"n_anchors": 3}
        for subset in self.subsets:
            t[f"{subset}_copy"]   = 0.85
            t[f"{subset}_anchor"] = 0.50
        return t

    def _sample_thresholds(self, iteration: int) -> dict:
        t = {}
        if iteration < 10 or not self.history:
            for subset in self.subsets:
                t[f"{subset}_copy"]   = random.uniform(0.65, 0.98)
                t[f"{subset}_anchor"] = random.uniform(0.20, 0.70)
            t["n_anchors"] = random.randint(1, 5)
        else:
            best = self.best_thresholds.copy()
            noise_scale = max(0.05, 0.20 * (1 - iteration / 200))
            for subset in self.subsets:
                copy_key   = f"{subset}_copy"
                anchor_key = f"{subset}_anchor"
                t[copy_key]   = float(np.clip(best[copy_key] + random.gauss(0, noise_scale), 0.60, 0.99))
                t[anchor_key] = float(np.clip(best[anchor_key] + random.gauss(0, noise_scale * 0.5), 0.15, 0.75))
            if random.random() < 0.20:
                t["n_anchors"] = random.randint(1, 5)
            else:
                t["n_anchors"] = best.get("n_anchors", 3)
        return t

    def _score_thresholds(self, thresholds: dict) -> tuple:
        results = predict_batch(self.val_df, self.indexes, thresholds, n_anchors=thresholds.get("n_anchors", 3))
        preds   = [r["prediction"] for r in results]
        refs    = self.val_df['output'].astype(str).tolist()
        subsets = self.val_df['subset'].tolist()
        summary = evaluate_predictions(preds, refs, subsets)
        return summary['_all']['blended'], summary

    def run(self, n_iterations: int = 200, log_path: str = "optimization_log.csv") -> dict:
        log_rows = []
        score, summary = self._score_thresholds(self.best_thresholds)
        self.best_score = score

        for i in range(1, n_iterations + 1):
            t0         = time.time()
            thresholds = self._sample_thresholds(i)
            score, summary = self._score_thresholds(thresholds)
            elapsed    = time.time() - t0

            self.history.append((thresholds.copy(), score))
            if score > self.best_score:
                self.best_score      = score
                self.best_thresholds = thresholds.copy()

            row = {"iteration": i, "blended": score, "elapsed_s": round(elapsed, 2), **thresholds}
            for subset, m in summary.items():
                if subset != '_all':
                    row[f"{subset}_r1"] = m['rouge1']
                    row[f"{subset}_rL"] = m['rougeL']
            log_rows.append(row)

            if i % 25 == 0:
                pd.DataFrame(log_rows).to_csv(log_path, index=False)

        pd.DataFrame(log_rows).to_csv(log_path, index=False)
        return self.best_thresholds


def extract_dialect_vocabulary(train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 500) -> dict:
    gaps = {}
    for subset in test_df['subset'].unique():
        test_sub  = test_df[test_df['subset'] == subset]
        train_sub = train_df[train_df['subset'] == subset]

        def clean_tokens(text):
            words = str(text).lower().split()
            return [re.sub(r'[?!.,;:\'\"]+', '', w) for w in words if len(w) > 3]

        test_counts  = Counter()
        train_counts = Counter()

        for text in test_sub['input']:
            for tok in clean_tokens(text):
                test_counts[tok] += 1

        for text in train_sub['output']:
            for tok in clean_tokens(text):
                train_counts[tok] += 1

        subset_gaps = {
            term: count for term, count in test_counts.items()
            if count >= 3 and train_counts[term] == 0
        }
        gaps[subset] = sorted(subset_gaps.items(), key=lambda x: -x[1])[:top_n]
    return gaps


def build_expanded_clinical_dict(base_dicts: dict, dialect_gaps: dict, train_df: pd.DataFrame) -> dict:
    expanded = {k: list(v) for k, v in base_dicts.items()}
    for subset, gap_terms in dialect_gaps.items():
        sub_df = train_df[train_df['subset'] == subset]
        all_answers = ' '.join(sub_df['output'].astype(str)).lower()
        for term, freq in gap_terms:
            best_cluster, best_cooccur = None, 0
            for cluster, seeds in base_dicts.items():
                cooccur = sum(1 for seed in seeds if seed in all_answers and term in all_answers)
                if cooccur > best_cooccur:
                    best_cooccur = cooccur
                    best_cluster = cluster
            if best_cluster and best_cooccur > 0:
                if term not in expanded[best_cluster]:
                    expanded[best_cluster].append(term)
    return expanded


def calibrate_prediction_length(pred: str, subset: str) -> str:
    target = LENGTH_TARGETS.get(subset, 100)
    words  = pred.split()
    if len(words) <= target * 1.3:
        return pred
    sentences = re.split(r'(?<=[.!?])\s+', pred)
    result = ""
    for sent in sentences:
        candidate = (result + " " + sent).strip()
        if len(candidate.split()) <= int(target * 1.3):
            result = candidate
        else:
            break
    return result if result else ' '.join(words[:target])


def main(args):
    print("="*60)
    print("  LEXICAL RAG OPTIMIZER")
    print("="*60)

    print("\nLoading data...")
    train_df = pd.read_csv(args.train)
    val_df   = pd.read_csv(args.val)
    test_df  = pd.read_csv(args.test)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    COMPOUNDS = [
        (r'\bHIV/AIDS\b', 'HIV'), (r'\bhiv/aids\b', 'HIV'),
        (r'\bukimwi/ukimwi\b', 'ukimwi'),
    ]
    for df_ in [train_df, val_df, test_df]:
        for col in ['input', 'output']:
            if col not in df_.columns:
                continue
            for pat, repl in COMPOUNDS:
                df_[col] = df_[col].astype(str).str.replace(pat, repl, regex=True, case=False)

    print("\nExtracting dialect vocabulary gaps...")
    dialect_gaps = extract_dialect_vocabulary(train_df, test_df)
    global CLINICAL_DICTS
    CLINICAL_DICTS = build_expanded_clinical_dict(CLINICAL_DICTS, dialect_gaps, train_df)

    print("\nBuilding per-subset BM25+TF-IDF indexes...")
    indexes = build_indexes(train_df, val_df)

    subsets   = list(indexes.keys())
    optimizer = BayesianThresholdOptimizer(subsets, val_df, indexes)
    best_thresholds = optimizer.run(n_iterations=args.iters, log_path=args.log)

    thresh_path = str(args.output).replace('.csv', '_thresholds.json')
    with open(thresh_path, 'w') as f:
        json.dump(best_thresholds, f, indent=2)

    print("\nGenerating test predictions...")
    test_results = predict_batch(test_df, indexes, best_thresholds, n_anchors=best_thresholds.get("n_anchors", 3))

    predictions = []
    for result, (_, row) in zip(test_results, test_df.iterrows()):
        pred = result["prediction"]
        pred = calibrate_prediction_length(pred, row['subset'])
        predictions.append(pred)

    submission = pd.DataFrame({
        "ID":         test_df["ID"],
        "TargetRLF1": predictions,
        "TargetR1F1": predictions,
        "TargetLLM":  predictions,
    })

    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved: {args.output} ({len(submission)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  required=True)
    parser.add_argument("--val",    required=True)
    parser.add_argument("--test",   required=True)
    parser.add_argument("--iters",  type=int, default=200)
    parser.add_argument("--output", default="submission_rag.csv")
    parser.add_argument("--log",    default="optimization_log.csv")
    args = parser.parse_args()
    main(args)
