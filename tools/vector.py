import sys
import pdb
import heapq
import numpy as np
from text2vec import SentenceModel, EncoderType
from text2vec import Word2Vec

kb_text = None
kb_embeddings = None
w2v_model = None


def compute_save_emb(model, data_dir):
    # 保存向量化之后的结果
    new_data_dir = "".join(data_dir.split(".")[:-1])+".npy"
    sentences = []
    with open(data_dir, "r") as f:
        sentences = f.readlines()
        sentences = [i.strip() for i in sentences if i != "\n"]

    sentence_embeddings = model.encode(sentences)
    print(type(sentence_embeddings), sentence_embeddings.shape)
    np.save(new_data_dir, sentence_embeddings)
    return 0


def retrive_top_k(text, k=3):
    a = w2v_model.encode(text)
    top_k_indices = top_k_similar_vectors(a, kb_embeddings, k)
    result = []
    for i in top_k_indices:
        result.append(kb_text[i])
    return result


def top_k_similar_vectors(a, n, k):
    heap = []
    for i, v in enumerate(n):
        sim = np.dot(a, v) / (np.linalg.norm(a) * np.linalg.norm(v))
        if len(heap) < k:
            heapq.heappush(heap, (sim, i))
        elif sim > heap[0][0]:
            heapq.heappop(heap)
            heapq.heappush(heap, (sim, i))
    return [i for _, i in sorted(heap, reverse=True)]


def load_txt(data_dir):
    with open(data_dir, "r") as f:
        sentences = f.readlines()
        sentences = [i.strip() for i in sentences if i != "\n"]
    return sentences


def load_npy(data_dir):
    embeddings = np.load(data_dir)
    return embeddings


def server_init():
    global kb_text
    global kb_embeddings
    global w2v_model
    w2v_model = SentenceModel()
    kb_text = load_txt("knowledge_base/disease.txt")
    kb_embeddings = load_npy("knowledge_base/disease.npy")
    return None


if __name__ == "__main__":
    w2v_model = SentenceModel()
    compute_save_emb(
        w2v_model, "./knowledge_base/disease.txt")
