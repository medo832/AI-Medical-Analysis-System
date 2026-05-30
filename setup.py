"""
First-time Setup Script
========================
Run ONCE after `pip install -r requirements.txt`:

    py -3.11 setup.py

What it does:
1. Verifies all required files are present.
2. Rebuilds the CT Chroma DB at 384-dim (the bundled one was 1024-dim).
   The MRI Chroma DB is already at 384-dim and is left as-is.
3. Pre-loads the sentence-transformer once so the app's first run is fast.
"""
import gc
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Suppress TF info noise during setup
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config


def banner(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def check_files():
    banner("[1/4] Checking required files")
    required = [
        ("YOLO kidney weights", config.YOLO_KIDNEY),
        ("KMeans (kidney)", config.KMEANS_KIDNEY),
        ("Scaler (kidney)", config.SCALER_KIDNEY),
        ("CT Chroma DB", config.CT_CHROMA_DIR / "chroma.sqlite3"),
        ("MRI Chroma DB", config.MRI_CHROMA_DIR / "chroma.sqlite3"),
    ]
    missing = []
    for name, path in required:
        ok = path.exists()
        mark = "OK " if ok else "MISSING"
        print(f"  [{mark}] {name}: {path}")
        if not ok:
            missing.append(name)

    # U-Net model is optional (large file the user must drop in)
    unet_ok = config.UNET_BRAIN.exists()
    mark = "OK " if unet_ok else "NOT YET"
    print(f"  [{mark}] U-Net (MRI segmentation): {config.UNET_BRAIN}")
    if not unet_ok:
        print("\n  >> Copy `segmentation_model.h5` into the `models/` folder.")
        print("  >> The MRI tab won't work until that file is in place.")

    if missing:
        print("\nERROR: missing files above. Cannot continue.")
        sys.exit(1)
    return unet_ok


def extract_documents_from_old_db(db_path: Path):
    """Pull (text, metadata) pairs out of an existing Chroma sqlite."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, key, string_value FROM embedding_metadata "
        "WHERE string_value IS NOT NULL"
    ).fetchall()
    conn.close()

    by_id = {}
    for emb_id, key, val in rows:
        by_id.setdefault(emb_id, {})[key] = val

    docs, metas = [], []
    for _, kv in sorted(by_id.items()):
        text = kv.pop("chroma:document", None)
        if not text:
            continue
        meta = {k: v for k, v in kv.items() if v and v != "N/A"}
        docs.append(text)
        metas.append(meta)
    return docs, metas


def db_dimension(db_path: Path) -> int:
    """Return the embedding dimension recorded in a Chroma sqlite, or 0."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        row = cur.execute("SELECT dimension FROM collections LIMIT 1").fetchone()
        return int(row[0]) if row and row[0] else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def rebuild_db_if_needed(modality: str, db_dir: Path):
    sqlite_path = db_dir / "chroma.sqlite3"
    dim = db_dimension(sqlite_path)
    expected = 384
    print(f"\n  {modality.upper()} DB dim: {dim} (expected: {expected})")
    if dim == expected:
        print(f"  -> {modality.upper()} DB OK, no rebuild needed.")
        return

    print(f"  -> Rebuilding {modality.upper()} DB at {expected}-dim...")
    docs, metas = extract_documents_from_old_db(sqlite_path)
    print(f"     extracted {len(docs)} chunks")
    if not docs:
        print("     ERROR: no chunks extracted!")
        return

    # Backup old DB
    backup = db_dir.parent / f"{modality}_old_backup"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(db_dir, backup)
    print(f"     backup saved to {backup}")

    # Wipe and rebuild in place
    shutil.rmtree(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    from langchain_huggingface import HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    from langchain_community.vectorstores import Chroma
    db = Chroma(persist_directory=str(db_dir), embedding_function=embedder)

    batch = 64
    for i in range(0, len(docs), batch):
        db.add_texts(texts=docs[i:i+batch], metadatas=metas[i:i+batch])
        print(f"     {min(i+batch, len(docs))}/{len(docs)} embedded")

    db = None
    gc.collect()
    print(f"  -> {modality.upper()} rebuilt at {expected}-dim.")


def setup_chroma():
    banner("[2/4] Setting up Chroma vector stores")
    rebuild_db_if_needed("ct", config.CT_CHROMA_DIR)
    rebuild_db_if_needed("mri", config.MRI_CHROMA_DIR)


def warm_up_models():
    banner("[3/4] Warming up models (downloads sentence-transformer if needed)")
    from langchain_huggingface import HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    _ = embedder.embed_query("warm up")
    print("  Embedding model ready.")

    from sentence_transformers import CrossEncoder
    rr = CrossEncoder(config.RERANKER_MODEL)
    _ = rr.predict([["a", "b"]])
    print("  Reranker model ready.")


def smoke_test_yolo():
    banner("[4/4] Smoke-testing YOLO + a sample CT image")
    try:
        from ultralytics import YOLO
        m = YOLO(str(config.YOLO_KIDNEY))
        samples = sorted(Path(config.CT_SAMPLES).glob("*.jpg"))
        if samples:
            res = m.predict(str(samples[0]), conf=0.3, verbose=False)
            n = sum(len(r.boxes) for r in res)
            print(f"  Sample image: {samples[0].name[:40]}")
            print(f"  Detected: {n} stones")
        print("  YOLO OK.")
    except Exception as e:
        print(f"  YOLO smoke test failed: {e}")


def main():
    print("Medical AI Diagnostic System - First-Time Setup")
    print("=" * 60)
    unet_ok = check_files()
    setup_chroma()
    warm_up_models()
    smoke_test_yolo()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    if not unet_ok:
        print("\n  Reminder: copy segmentation_model.h5 into models/ before"
              "\n  using the MRI tab.")
    print("\n  Now run the app with:")
    print("    py -3.11 -m streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
