# test_search.py  ←  save as this name

from PIL import Image
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.query import MetadataQuery
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")  # removes the long "slow processor" warning

# ----------------------- CONFIG -----------------------
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
QUERY_IMAGE = Path("./graphs_only/005_.png")   # ← change if needed
# -----------------------------------------------------

# Load model (silence the warning)
print("Loading CLIP model...")
model = SentenceTransformer(MODEL_NAME)

# Connect to local Weaviate
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051
)

collection = client.collections.get("MyLocalImages")

# Check if the query image actually exists
if not QUERY_IMAGE.exists():
    print(f"Error: Image not found → {QUERY_IMAGE.resolve()}")
    print("Available images in ./graphs_only/:")
    for p in Path("./graphs_only").glob("*.*"):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            print("   •", p.name)
    client.close()
    exit()

# Load and encode the query image
print(f"Loading query image: {QUERY_IMAGE.name}")
query_image = Image.open(QUERY_IMAGE).convert("RGB")
query_vector = model.encode(query_image, normalize_embeddings=True).tolist()

# Search!
print("\nSearching for similar graphs...\n")
response = collection.query.near_vector(
    near_vector=query_vector,
    limit=6,
    return_metadata=MetadataQuery(distance=True)
)

print(f"{'Rank':<4} {'Filename':<35} {'Distance'}")
print("-" * 4 + " " + "-"*35 + "  " + "-"*8)
for i, obj in enumerate(response.objects, 1):
    filename = obj.properties.get("filename", "unknown")
    dist = obj.metadata.distance if obj.metadata.distance is not None else 0.0
    print(f"{i:<4} {filename:<35} {dist:.5f}")

# This should show your own image (005.png) with distance ≈ 0.00000 at rank 1

client.close()          # important: no more ResourceWarning
print("\nSuccess! Connection closed cleanly.")