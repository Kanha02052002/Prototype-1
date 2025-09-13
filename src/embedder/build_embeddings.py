import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

def build_embeddings(kb_path='kb/kb_consolidated_1.txt', embed_dir='embeddings'):
    os.makedirs(embed_dir, exist_ok=True)

    with open(kb_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = [p.strip() for p in text.split('\n') if p.strip()]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    with open(os.path.join(embed_dir, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, os.path.join(embed_dir, 'faiss_index.bin'))
    print("Embeddings and FAISS index built and saved.")

if __name__ == "__main__":
    build_embeddings()