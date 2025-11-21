# filename: index_images_to_weaviate.py

import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.util import generate_uuid5

# ====================== CONFIG ======================
IMAGE_FOLDER = "./graphs_only"                  # your folder with images
JSON_BACKUP  = "./my_image_vectors_backup.json"
CLASS_NAME   = "MyLocalImages"
MODEL_NAME   = "sentence-transformers/clip-ViT-B-32"   # 512-dim open-source CLIP
# ===================================================

print("Loading CLIP model (first run takes ~30 seconds)...")
model = SentenceTransformer(MODEL_NAME)

# Connect to your local Weaviate running on localhost:8080
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051
)

print("Connected to Weaviate → version:", client.get_meta()["version"])

# Delete class if it already exists (clean start every time)
if client.collections.exists(CLASS_NAME):
    client.collections.delete(CLASS_NAME)
    print(f"Deleted existing collection '{CLASS_NAME}'")

# Create the collection (new v4 way)
collection = client.collections.create(
    name=CLASS_NAME,
    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),  # we bring our own vectors
    vector_index_config=weaviate.classes.config.Configure.VectorIndex.hnsw(
        distance_metric=weaviate.classes.config.VectorDistances.COSINE
    ),
    properties=[
        weaviate.classes.config.Property(name="filename", data_type=weaviate.classes.config.DataType.TEXT),
        weaviate.classes.config.Property(name="path",     data_type=weaviate.classes.config.DataType.TEXT),
    ]
)

print(f"Created collection '{CLASS_NAME}'")

# Find all images
image_paths = list(Path(IMAGE_FOLDER).rglob("*"))
image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}]

backup = []

print(f"\nProcessing {len(image_paths)} images...\n")

with collection.batch.dynamic() as batch:
    for img_path in tqdm(image_paths):
        try:
            image = Image.open(img_path).convert("RGB")

            # Generate 512-dim vector (already L2-normalized because of normalize_embeddings=True)
            vector = model.encode(image, show_progress_bar=False, normalize_embeddings=True).tolist()

            # Save to local JSON backup
            backup.append({
                "filename": img_path.name,
                "path"    : str(img_path),
                "vector"  : vector
            })

            # Add to Weaviate
            batch.add_object(
                properties={
                    "filename": img_path.name,
                    "path"    : str(img_path)
                },
                vector=vector,
                uuid=generate_uuid5(img_path)   # deterministic UUID → safe to re-run
            )

        except Exception as e:
            print(f"Failed {img_path}: {e}")

# Save the backup file so you can open and see all vectors
with open(JSON_BACKUP, "w", encoding="utf-8") as f:
    json.dump(backup, f, indent=2)

print("\nFinished!")
print(f"Indexed {len(backup)} images into Weaviate (localhost:8080)")
print(f"Backup JSON with all vectors → {JSON_BACKUP}")

client.close()

# Quick test query you can run anytime
print("\nQuick search example:")
print("""
from PIL import Image
query_image = Image.open("graphs_only/some_graph.png")
query_vector = model.encode(query_image, normalize_embeddings=True).tolist()

results = collection.query.near_vector(
    near_vector=query_vector,
    limit=5,
    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
)

for o in results.objects:
    print(o.properties['filename'], "→ distance:", o.metadata.distance)
""")