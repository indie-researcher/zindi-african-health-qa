"""
Purpose: Harvests and extracts winning Kaggle solution writeups and competition metadata.
Inputs: Kaggle discussion CSVs, competition YAML manifests, and cached headers.
Outputs: Augmented JSON file containing structured competition tasks and solution writeups.
"""
import os
import re
import sys
import yaml
import json
import html
import argparse
import pandas as pd

# --- CLI ARGUMENTS ---
parser = argparse.ArgumentParser(description="Harvest Kaggle writeups and metadata.")
parser.add_argument("--dry-run", action="store_true", default=False, help="Enable dry-run test mode streaming limit.")
args = parser.parse_args()

# --- PATH & RUN CONFIGURATION ---
WORK_DIR = "/kaggle/working"
OUTPUT_PATH = os.path.join(WORK_DIR, "augmented_kaggle_research_tasks.json")

INPUT_DATASET_DIR = "/kaggle/input/kaggle-discussions"
MANIFEST_BACKUP_PATH = os.path.join(INPUT_DATASET_DIR, "checkpoint_raw_manifest.yaml")
FORUM_TOPICS_CSV = os.path.join(INPUT_DATASET_DIR, "ForumTopics.csv")
FORUM_MESSAGES_CSV = os.path.join(INPUT_DATASET_DIR, "ForumMessages.csv")
RAW_HEADERS_DIR = os.path.join(INPUT_DATASET_DIR, "raw_competition_headers")
DATASET_SLUG = "yoka33/kaggle-discussions"

print("⚙️ Launching Unified Metadata Extraction & Processing Pipeline...")

# ==========================================
# 🛡️ INPUT VALIDATION PRE-FLIGHTS
# ==========================================
for path in [MANIFEST_BACKUP_PATH, FORUM_TOPICS_CSV, FORUM_MESSAGES_CSV]:
    if not os.path.exists(path):
        print(f"❌ FATAL: Missing tracking dependency at {path}")
        sys.exit(1)

with open(MANIFEST_BACKUP_PATH, "r", encoding="utf-8") as f:
    raw_manifest = yaml.safe_load(f.read())

raw_items_list = raw_manifest.get('competitions', []) if isinstance(raw_manifest, dict) else raw_manifest
print(f"🚀 Loaded {len(raw_items_list)} competition metadata profiles.")

# ==========================================
# 📊 PHASE 1: TEXT SLUG EXTREMETIES HARVEST
# ==========================================
slug_to_meta_package = {}
target_writeup_slugs = set()

for idx, comp in enumerate(raw_items_list):
    if not isinstance(comp, dict): continue
    
    comp_link = comp.get("link", "")
    comp_slug = comp_link.split("/c/")[-1] if "/c/" in comp_link else f"unknown-slug-{idx}"
    comp_slug = comp_slug.strip().lower()

    # Match pre-downloaded training data attributes
    cached_header_file = os.path.join(RAW_HEADERS_DIR, f"{comp_slug}_train.csv")
    data_profile = {"status": "No train file available"}
    if os.path.exists(cached_header_file):
        try:
            df = pd.read_csv(cached_header_file)
            data_profile = {
                "status": "Success",
                "features_detected": len(df.columns),
                "column_types": df.dtypes.astype(str).to_dict()
            }
        except Exception as e:
            print(f"⚠️ Warning: Failed to read cached header {cached_header_file}: {e}")

    meta_package = {
        "competition_slug": comp_slug,
        "task_name": comp.get("title", comp_slug.replace("-", " ").title()),
        "category": comp.get("kind", "General"),
        "evaluation_metric": comp.get("metric", "Unknown Metric"),
        "data_distribution_profile": data_profile
    }

    solutions_list = comp.get("solutions", [])
    if isinstance(solutions_list, list):
        for sol in solutions_list:
            if isinstance(sol, dict) and "link" in sol:
                sol_url = str(sol["link"])
                writeup_slug = sol_url.split("/writeups/")[-1].strip().lower()
                if writeup_slug and writeup_slug != "none":
                    target_writeup_slugs.add(writeup_slug)
                    slug_to_meta_package[writeup_slug] = meta_package

print(f"🎯 Aggregated {len(target_writeup_slugs)} target writeup strings from manifest.")

# ==========================================
# ⚡ PHASE 2: INTERNAL FORUM ID INDEX MAPPING
# ==========================================
print("\n⚡ Scanning ForumTopics table to resolve relationship connections...")
topic_to_metadata = {}

def clean_to_url_slug(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"[\s\-]+", "-", text)
    return text.strip("-")

sample_df = pd.read_csv(FORUM_TOPICS_CSV, nrows=1)
id_col = next((c for c in sample_df.columns if c.lower() == 'id'), None)
title_col = next((c for c in sample_df.columns if 'title' in c.lower()), None)

chunk_size = 200_000
for chunk in pd.read_csv(FORUM_TOPICS_CSV, usecols=[id_col, title_col], chunksize=chunk_size):
    for _, row in chunk.iterrows():
        topic_slug = clean_to_url_slug(str(row[title_col]))
        if topic_slug in target_writeup_slugs:
            topic_id = int(row[id_col])
            # Explicitly capture topic_id inside metadata block for phase 3 safety lookup
            meta = slug_to_meta_package[topic_slug].copy()
            meta["topic_id"] = topic_id
            topic_to_metadata[topic_id] = meta

print(f"📋 Sanity Check: Mapped {len(topic_to_metadata)} distinct forum topic matching keys.")
assert len(topic_to_metadata) > 0, "Pipeline verification aborted: Zero topics bound!"

# ==========================================
# 🌊 PHASE 3: RELATIONAL TEXT EXTRACTION LOOP
# ==========================================
print("\n⚡ Dynamic columns inspection on ForumMessages database...")
augmented_dataset = []
target_set_filter = set(topic_to_metadata.keys())

# Inspect true column names case-insensitively
msg_sample = pd.read_csv(FORUM_MESSAGES_CSV, nrows=1)
actual_msg_cols = list(msg_sample.columns)
print(f"   ↳ ForumMessages headers found on disk: {actual_msg_cols}")

# Robust fallback matching sequence
msg_topic_col = next((c for c in actual_msg_cols if 'topic' in c.lower()), None)
message_col = next((c for c in actual_msg_cols if 'message' in c.lower() and 'id' not in c.lower()), None)
first_msg_col = next((c for c in actual_msg_cols if 'first' in c.lower() or 'isinitial' in c.lower()), None)

# Critical sanity fallback check
if not msg_topic_col or not message_col:
    print("❌ FATAL: Essential text columns missing from ForumMessages structure.")
    sys.exit(1)

# If IsFirstMessage isn't explicitly found, read dynamically without filtering it in the chunk layer
use_columns = [msg_topic_col, message_col]
if first_msg_col:
    use_columns.append(first_msg_col)

print(f"⚡ Streaming relational chunks via columns: {use_columns}")

DRY_RUN_TEST_MODE = args.dry_run
chunk_size = 200_000

for idx, chunk in enumerate(pd.read_csv(FORUM_MESSAGES_CSV, usecols=use_columns, chunksize=chunk_size)):
    if DRY_RUN_TEST_MODE and idx >= 3:
        print("🛑 Dry-run verification threshold boundary breached. Stopping streaming iteration loop.")
        break
        
    # Apply row filter dynamically based on available attributes
    if first_msg_col:
        matched_rows = chunk[(chunk[msg_topic_col].isin(target_set_filter)) & (chunk[first_msg_col] == True)]
    else:
        matched_rows = chunk[chunk[msg_topic_col].isin(target_set_filter)]
    
    for _, row in matched_rows.iterrows():
        t_id = int(row[msg_topic_col])
        meta = topic_to_metadata[t_id]
        
        clean_text = re.sub(r'<[^>]*>', ' ', str(row[message_col]))
        clean_text = html.unescape(clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) > 150:
            augmented_dataset.append({
                **meta,
                "solution_writeup_clean": clean_text
            })

# ==========================================
# 🛡️ FINAL METRIC CHECK AND OUTPUT
# ==========================================
print("\n🛡️ Running Core Pipeline Integration Validations...")
print(f"📊 Extraction Summary: Extracted {len(augmented_dataset)} complete comprehensive tasks!")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(augmented_dataset, f, indent=4, ensure_ascii=False)
print(f"💾 File written safely out to disk storage at: {OUTPUT_PATH}")

if len(augmented_dataset) > 0:
    print("\n🔍 SAMPLE EXTRACTION QA LOG:")
    print(json.dumps(augmented_dataset[0], indent=2)[:600] + "\n...")

print("\n🏁 Unified pipeline run execution completed successfully!")
