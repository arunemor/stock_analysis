import os
import json
import logging
from typing import List, Dict, Any

import weaviate
from weaviate.classes.config import Configure

from sentence_transformers import SentenceTransformer
from PIL import Image
from pdf2image import convert_from_path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument


logging.basicConfig(level=logging.INFO)

# ============================================================
# 1. MODEL → unified 512-dim (text + image)
# ============================================================

VECTOR_SIZE = 512
model = SentenceTransformer("clip-ViT-B-32")


# ============================================================
# 2. WEAVIATE CONNECT
# ============================================================

client = weaviate.connect_to_local(
    host="localhost",
    port=8081,
    grpc_port=50051,
    skip_init_checks=False
)


# ============================================================
# 3. BACKUP
# ============================================================

BACKUP_DIR = "backup"
os.makedirs(BACKUP_DIR, exist_ok=True)

def write_backup(data, step_name):
    path = os.path.join(BACKUP_DIR, f"{step_name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"[BACKUP SAVED] {path}")


# ============================================================
# 4. FIXED ensure_collection()  — Weaviate v4.7 compatible
# ============================================================

def ensure_collection():
    name = "PDFContent"

    try:
        col = client.collections.get(name)
        cfg = col.config.get()

        # In Weaviate v4.7 → vector size is stored here:
        try:
            existing_dim = cfg.vector_index_config.vector_size
        except:
            existing_dim = None

        # Check mismatch
        if existing_dim != VECTOR_SIZE:
            logging.warning(f"Vector size mismatch: {existing_dim} != {VECTOR_SIZE}")
            logging.warning("Deleting old collection…")
            client.collections.delete(name)
            raise Exception("Recreate")

        return col

    except:
        logging.info("Creating new collection with vector size 512...")

        return client.collections.create(
            name=name,

            # Correct: vector size is defined inside vector_index_config
            vectorizer_config=Configure.Vectorizer.none(
                vector_index_config=Configure.VectorIndex.hnsw(
                    vector_size=VECTOR_SIZE,
                    ef_construction=128,
                    max_connections=32
                )
            ),

            properties=[
                {"name": "content", "data_type": "text"},
                {"name": "content_type", "data_type": "text"},
                {"name": "metadata", "data_type": "text"},
            ]
        )


# ============================================================
# 5. DOCLING EXTRACTION
# ============================================================

converter = DocumentConverter()

def extract_docling_content(pdf_path):
    result = converter.convert(pdf_path)
    doc: DoclingDocument = result.document

    out = {
        "headers": [],
        "paragraphs": [],
        "tables": [],
        "lists": [],
        "figures": []
    }

    for item, level in doc.iterate_items():
        label = item.label.value if hasattr(item, "label") else None

        try:
            text = item.export_to_markdown(doc=doc).strip()
        except:
            continue

        if not text:
            continue

        if label == "section_header":
            out["headers"].append(text)
        elif label == "paragraph":
            out["paragraphs"].append(text)
        elif label == "list_item":
            out["lists"].append(text)
        elif label == "table":
            out["tables"].append(text)
        elif label == "picture":
            out["figures"].append(text)

    write_backup(out, "step_1_docling")
    return out


# ============================================================
# 6. IMAGE EXTRACTION
# ============================================================

def extract_images(pdf_path):
    pages = convert_from_path(pdf_path)
    os.makedirs("images", exist_ok=True)

    paths = []
    for i, p in enumerate(pages):
        path = f"images/page_{i+1}.png"
        p.save(path, "PNG")
        paths.append(path)

    write_backup({"images": paths}, "step_2_images")
    return paths


# ============================================================
# 7. CHUNK TEXT
# ============================================================

def chunk_text(text, max_len=300):
    words = text.split()
    chunks, cur = [], []

    for w in words:
        cur.append(w)
        if len(cur) >= max_len:
            chunks.append(" ".join(cur))
            cur = []

    if cur:
        chunks.append(" ".join(cur))

    return chunks


# ============================================================
# 8. EMBEDDINGS
# ============================================================

def embed_text(text):
    return model.encode(text).tolist()

def embed_image(path):
    img = Image.open(path)
    return model.encode(img).tolist()


# ============================================================
# 9. PIPELINE
# ============================================================

def process_pdf(pdf_path):
    logging.info("===== START PIPELINE =====")

    collection = ensure_collection()

    all_backup = {
        "chunks": [],
        "images": []
    }

    # Step 1
    extracted = extract_docling_content(pdf_path)

    # Step 2
    images = extract_images(pdf_path)

    # Step 3 — Text chunks
    blocks = []
    blocks += [(t, "header") for t in extracted["headers"]]
    blocks += [(t, "paragraph") for t in extracted["paragraphs"]]
    blocks += [(t, "list") for t in extracted["lists"]]
    blocks += [(t, "table") for t in extracted["tables"]]

    chunk_backup = []

    for text, ttype in blocks:
        for ch in chunk_text(text):
            vec = embed_text(ch)

            collection.data.insert(
                properties={
                    "content": ch,
                    "content_type": ttype,
                    "metadata": "{}"
                },
                vector=vec
            )

            item = {"content": ch, "type": ttype, "vector": vec}
            chunk_backup.append(item)
            all_backup["chunks"].append(item)

    write_backup(chunk_backup, "step_3_chunks")

    # Step 4 — Images
    img_backup = []

    for path in images:
        vec = embed_image(path)

        collection.data.insert(
            properties={
                "content": path,
                "content_type": "image",
                "metadata": json.dumps({"path": path})
            },
            vector=vec
        )

        entry = {"path": path, "vector": vec}
        img_backup.append(entry)
        all_backup["images"].append(entry)

    write_backup(img_backup, "step_4_image_vectors")

    # Final
    write_backup(all_backup, "step_final")

    logging.info("===== PIPELINE COMPLETE =====")
    client.close()


# ============================================================
# 10. SEARCH (Weaviate v4)
# ============================================================

def semantic_search(query, top_k=3):
    col = client.collections.get("PDFContent")
    vec = embed_text(query)

    res = col.query.near_vector(
        near_vector=vec,
        limit=top_k,
        return_properties=["content", "content_type", "metadata"]
    )

    return res.objects


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":
    pdf_path = "sample1.pdf"
    process_pdf(pdf_path)

    print("\n=== SEARCH RESULTS ===")
    results = semantic_search("shareholding pattern")

    for r in results:
        print("\nContent:", r.properties.get("content"))
        print("Type:", r.properties.get("content_type"))
        print("Metadata:", r.properties.get("metadata"))
