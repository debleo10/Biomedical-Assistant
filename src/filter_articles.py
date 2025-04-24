import os
import re
import markdown
import faiss
import json
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

def load_articles(directory):
    articles = {}
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            pubmed_id = filename.split(".")[0]
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                raw_md = f.read()
            html = markdown.markdown(raw_md)
            soup = BeautifulSoup(html, "html.parser")
            articles[pubmed_id] = soup.get_text()
    return articles
y=load_articles('../data/papers')

def get_matching_ids(y):
    keyword_pattern = r'\bcancer\w*|\bimmuno\w*'
    matching_ids = [k for k, v in y.items() if re.search(keyword_pattern, v, flags=re.IGNORECASE)]
    pmids=[i for i in y]
    # print(len(matching_ids))
    l=[]
    for i in pmids:
        l.append(y[i])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(l)

    embedding_dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index_ip  = faiss.IndexFlatIP(embedding_dimension)


    index_ip .add(embeddings)
    query_terms = ["This is a text related to cancer or immunology"]
    query_embeddings = model.encode(query_terms).astype('float32')
    faiss.normalize_L2(query_embeddings)
    cosine_similarities, indices = index_ip.search(query_embeddings, 50)
    z = []
    for sims, idxs in zip(cosine_similarities, indices):
        for sim, idx in zip(sims, idxs):
            if sim > 0.2 and idx != -1:
                z.append(pmids[idx])
    union_list = list(set(z) | set(matching_ids))
    return union_list
if __name__=="__main__":
    y = load_articles('../data/papers')
    ids=get_matching_ids(y)
    d={}
    d['filtered_articles']=ids
    with open("../outputs/filtered_articles.json", "w") as f:
        json.dump(d, f, indent=2)

    print("JSON file saved to outputs/filtered_articles.json")