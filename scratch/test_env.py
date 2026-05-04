from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully")
embeddings = model.encode(["hello world"])
print("Embeddings shape:", embeddings.shape)
