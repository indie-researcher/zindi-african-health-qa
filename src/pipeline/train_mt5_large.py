"""
Purpose: Fine-tunes mT5-large model for multilingual health question answering across African dialects.
Inputs: Optimized training and validation CSVs, cluster dictionaries, and test data.
Outputs: Fine-tuned model checkpoints and final evaluation submission CSV.
"""
import os

# ── 1. FORCE COMPATIBILITY DEPENDENCY INSTALLATIONS FIRST ──────────────────
print("⚙️ Overriding environment with sm_60 compatible PyTorch binaries...")
os.system("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121 --quiet")

print("📦 Upgrading core model engineering framework packages...")
os.system("pip install sentencepiece transformers evaluate scikit-learn rouge_score rank_bm25 --quiet")

# ── 2. STANDARD IMPORTS ───────────────────────────────────────────────────────
import json, random, re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
import evaluate

from transformers import (
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from transformers import T5Tokenizer as MT5Tokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME        = "google/mt5-large"
OUTPUT_DIR        = "/kaggle/working/mt5_msrh_run1"
TENSORBOARD_DIR   = "/kaggle/working/logs"

# ENGINEER / REDUCE TIME: Read optimized files
TRAIN_PATH   = "optimized_train.csv"
VAL_PATH     = "optimized_val.csv"
TEST_PATH    = "/kaggle/input/qna-african-languages-challenge/Test.csv"

LENGTH_CONFIG = {
    "Amh_Eth": {"max": 60,  "min": 10},
    "Eng_Eth": {"max": 70,  "min": 15},
    "Eng_Gha": {"max": 200, "min": 20},
    "Eng_Ken": {"max": 220, "min": 20},
    "Eng_Uga": {"max": 260, "min": 20},
    "Lug_Uga": {"max": 220, "min": 20},
    "Swa_Ken": {"max": 230, "min": 20},
    "Aka_Gha": {"max": 270, "min": 30},
}

MAX_INPUT_LENGTH  = 128   
MAX_TARGET_LENGTH = 256   
BATCH_TRAIN       = 4     
BATCH_EVAL        = 1
EPOCHS            = 3
LR                = 1e-3  

print("Loading datasets...")
df_train = pd.read_csv(TRAIN_PATH)
df_val = pd.read_csv(VAL_PATH)
df_test = pd.read_csv(TEST_PATH)

# Clean up sticky trailing punctuation
def normalize_target_punctuation(text):
    if not isinstance(text, str): return text
    text = re.sub(r'([?.!,])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip()

df_train["output"] = df_train["output"].apply(normalize_target_punctuation)
df_val["output"] = df_val["output"].apply(normalize_target_punctuation)

# ── LOGIC: CLUSTER MAP & CLINICAL KEYWORD DICTIONARIES ──────────────────────
category_map = {
    'G01_ADOLESCENT_SRH': 'ADOLESCENT SRH',
    'G02_HIV_GENERAL': 'HIV GENERAL',
    'G03_HIV_TRANSMISSION': 'HIV TRANSMISSION',
    'G04_HERPES_HSV': 'HERPES',
    'G05_SYPHILIS': 'SYPHILIS',
    'G06_CHLAMYDA': 'CHLAMYDIA',
    'G06_CHLAMYDIA': 'CHLAMYDIA',
    'G07_GONORRHEA': 'GONORRHEA',
    'G08_HEPATITIS_HBV': 'HEPATITIS',
    'G09_TRICHOMONIASIS': 'TRICHOMONIASIS',
    'G10_HPV': 'HPV',
    'G11_PEP_PREP': 'PEP PREP',
    'G12_GENERAL_STIS': 'GENERAL STIS',
    'G13_CULTURAL_MYTHS': 'CULTURAL_MYTHS',
    'G14_FALLBACK': 'FALLBACK'
}

CLINICAL_DICTS = {
    'ADOLESCENT SRH': [
        'puberty', 'adolescent', 'teen', 'peer pressure', 'hormone', 
        'financial empowerment', 'pregnancy', 'contraceptive', 'condom', 'srh',
        'mimba', 'kondomu', 'lubuto', 'kapiira', 'nyinsɛn', 'kɔndɔm', 'እርግዝና', 'ኮንዶም'
    ],
    'HIV GENERAL': [
        'hiv', 'aids', 'arv', 'art', 'retroviral', 'viral load', 'cd4',
        'ukimwi', 'vvu', 'siriimu', 'akawuka', 'ኤችአይቪ', 'ኤድስ'
    ],
    'HIV TRANSMISSION': [
        'transmit', 'transmission', 'spread hiv', 'catch hiv', 'mosquito bite', 'saliva',
        'kutambaza', 'kwata'
    ],
    'HERPES': [
        'herpes', 'hsv', 'cold sore', 'blister', 'acyclovir', 'ሄርፒስ'
    ],
    'SYPHILIS': [
        'syphilis', 'chancre', 'vdrl', 'rpr', 'kaswende', 'kabotongo', 'ቂጥኝ'
    ],
    'CHLAMYDIA': [
        'chlamydia', 'ክላሚዲያ'
    ],
    'GONORRHEA': [
        'gonorrhea', 'clap', 'gonorrhoea', 'kisonono', 'nzege', 'babaso', 'ጨብጥ'
    ],
    'HEPATITIS': [
        'hepatitis', 'hbv', 'hepb', 'liver', 'homa ya ini', 'ሄፓታይተስ'
    ],
    'TRICHOMONIASIS': [
        'trichomoniasis', 'trich', 'ትሪኮሞኒያሲስ'
    ],
    'HPV': [
        'hpv', 'genital wart', 'cervical cancer'
    ],
    'PEP PREP': [
        'prep', 'pep', 'truvada', 'post-exposure'
    ],
    'GENERAL STIS': [
        'sti', 'std', 'venereal', 'infection', 'ndwara', 'yareza'
    ],
    'CULTURAL_MYTHS': [
        'cure', 'heal hiv', 'prayer', 'myth', 'witchcraft', 'lemon', 'garlic',
        'dawa ya kienyeji', 'mti', 'kitunguu swumu', 'ሎሚ', 'ነጭ ሽንኩርት'
    ]
}

def dictionary_classify(text):
    text_lower = str(text).lower()
    for category, keywords in CLINICAL_DICTS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return 'FALLBACK'

# Heuristic Mapping Integration
if 'target_group' not in df_train.columns:
    if 'ID' in df_train.columns and 'id_map' in globals():
        df_train['target_group'] = df_train['ID'].map(id_map).map(category_map)
    else:
        df_train['target_group'] = df_train['input'].apply(dictionary_classify)

GROUP_PREFIXES = {
    "CULTURAL_MYTHS":      "answer health myth: ",
    "ADOLESCENT SRH":      "answer adolescent health: ",
    "HIV GENERAL":         "answer HIV: ",
    "HIV TRANSMISSION":    "answer HIV myth: ",
    "HERPES":              "answer herpes: ",
    "SYPHILIS":            "answer syphilis: ",
    "CHLAMYDIA":           "answer chlamydia: ",
    "GONORRHEA":           "answer gonorrhea: ",
    "HEPATITIS":           "answer hepatitis: ",
    "TRICHOMONIASIS":      "answer trichomoniasis: ",
    "HPV":                 "answer HPV: ",
    "PEP PREP":            "answer PEP PrEP: ",
    "GENERAL STIS":        "answer STI: ",
    "FALLBACK":            "answer general health: "
}

# ── LOGIC: NEW BUILD INPUT WITH DICTIONARY ROUTING FALLBACK ──────────────────
def build_input(row):
    # Route via calculated target group heuristic instead of missing merged_group field
    assigned_group = row.get("target_group")
    if pd.isna(assigned_group) or assigned_group not in GROUP_PREFIXES:
        assigned_group = dictionary_classify(row['input'])
        
    prefix = GROUP_PREFIXES.get(assigned_group, "answer general health: ")
    base_input = str(row['input']).strip()
    
    if row.get("subset") == "Eng_Uga":
        base_input = f"please answer using simple medical terms in detail. {base_input}"
        
    return f"{prefix}{base_input}"

df_train["model_input"] = df_train.apply(build_input, axis=1)
df_val["model_input"] = df_val.apply(build_input, axis=1)

df_train = df_train[df_train["model_input"].str.len() > 5]
df_train = df_train[df_train["output"].astype(str).str.len() > 5]

# ── TOKENIZER & CACHED MAPPING ────────────────────────────────────────────────
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    model_inputs = tokenizer(batch["model_input"], max_length=128, truncation=True)
    labels = tokenizer(text_target=batch["output"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_hf = Dataset.from_pandas(df_train[["model_input", "output"]])
val_hf   = Dataset.from_pandas(df_val[["model_input", "output"]])

print("Tokenizing datasets...")
# Tokenization occurs here once outside training loop (Cached Tokenization)
train_tok = train_hf.map(tokenize, batched=True, batch_size=1000, remove_columns=train_hf.column_names)
val_tok   = val_hf.map(tokenize, batched=True, batch_size=1000, remove_columns=val_hf.column_names)

# ENGINEER / REDUCE TIME: Slice validation sets & specify feedback limit lengths
eval_dataset = val_tok.select(range(min(150, len(val_tok))))
generation_max_length = 128  

# ── MODEL CONFIGURATION & TRAINING ARGS ───────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, use_safetensors=True)

if hasattr(model.config, "tie_word_embeddings"):
    model.config.tie_word_embeddings = False

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(
        predictions=[p.strip() for p in decoded_preds],
        references=[l.strip() for l in decoded_labels],
        use_stemmer=False,
    )
    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_TRAIN,
    per_device_eval_batch_size  = BATCH_EVAL,
    predict_with_generate       = True,
    fp16                        = False,  
    eval_strategy               = "steps",
    eval_steps                  = 1000,
    save_steps                  = 1000,
    save_total_limit            = 1,
    load_best_model_at_end      = True,
    metric_for_best_model       = "rouge1",
    greater_is_better           = True,
    optim                       = "adafactor",
    learning_rate               = LR,
    lr_scheduler_type           = "constant",
    generation_max_length       = generation_max_length, # Dynamic evaluation limit cap applied
    generation_num_beams        = 4,
    logging_steps               = 200,
    report_to                   = "none",
    gradient_accumulation_steps = 2,
    group_by_length             = True                   # ENGINEER: Performance booster flag
)

trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tok,
    eval_dataset    = eval_dataset,                      # ENGINEER: Evaluates target slice length
    processing_class = tokenizer,  
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Starting training phase...")
trainer.train()

# ── INFERENCE ─────────────────────────────────────────────────────────────────
print("Running inference generation on test platform...")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
predictions = []

for _, row in df_test.iterrows():
    subset  = row["subset"]
    lc      = LENGTH_CONFIG.get(subset, {"max": 200, "min": 15})
    
    # Inference alignment with Dictionary matching
    assigned_group = dictionary_classify(row['input'])
    prefix  = GROUP_PREFIXES.get(assigned_group, "answer general health: ")
    
    base_inp = str(row['input']).strip()
    if subset == "Eng_Uga":
        base_inp = f"please answer using simple medical terms in detail. {base_inp}"
    inp = f"{prefix}{base_inp}"

    inputs  = tokenizer(inp, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens      = lc["max"],
            min_new_tokens      = lc["min"],
            num_beams           = 4,
            length_penalty      = 0.8,
            early_stopping      = True,
            no_repeat_ngram_size= 3,
        )

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred)

submission = pd.DataFrame({
    "ID":         df_test["ID"],
    "TargetRLF1": predictions,
    "TargetR1F1": predictions,
    "TargetLLM":  predictions,
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Processing complete. Submission saved successfully.")