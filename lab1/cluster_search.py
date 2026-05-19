# cluster_search_no_ui.py

import re
import nltk
import spacy
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Setup
# -------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

N_CLUSTERS = 5
TOP_KW_PER_CLUSTER = 20
SEARCH_TOP_N = 7

INPUT_CSV = "articles.csv"
TEXT_COL = "text"


# -------------------------
# Generic words
# -------------------------
GENERIC = {
    "people", "person", "year", "time", "new", "world", "day", "life", "man", "woman", "child",
    "many", "state", "government", "news", "video", "percent", "one", "two", "three", "also",
    "say", "make", "get", "like", "work", "company", "group", "month", "week", "today",
    "yesterday", "way", "thing", "police", "official", "officer", "city", "country", "case",
    "report", "story", "issue", "public"
}


# -------------------------
# Cleaning
# -------------------------
def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------
# Lemmatization
# -------------------------
def spacy_lemmatize(texts: pd.Series) -> list[str]:
    results = []

    for doc in nlp.pipe(texts.tolist(), batch_size=128):
        tokens = [
            t.lemma_.lower()
            for t in doc
            if t.is_alpha
            and len(t.text) > 2
            and t.pos_ in ("NOUN", "PROPN", "ADJ")
            and t.lemma_.lower() not in stop_words
        ]
        results.append(" ".join(tokens))

    return results


# -------------------------
# Top keywords per cluster
# -------------------------
def top_keywords_per_cluster(vectorizer, X, labels, top_n=20):
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_keywords = {}

    for c in range(N_CLUSTERS):
        idx = np.where(labels == c)[0]

        if len(idx) == 0:
            cluster_keywords[c] = []
            continue

        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = np.argsort(mean_tfidf)[::-1][:top_n]
        cluster_keywords[c] = feature_names[top_idx].tolist()

    return cluster_keywords


# -------------------------
# Automatic cluster names
# -------------------------
def make_cluster_label_auto(cluster_keywords, max_terms=3):
    labels = {}

    for c, kws in cluster_keywords.items():
        bigrams = [kw for kw in kws if " " in kw and kw not in GENERIC]
        unigrams = [
            kw for kw in kws
            if " " not in kw and kw not in GENERIC and len(kw) > 3
        ]

        chosen = bigrams[:max_terms] if bigrams else unigrams[:max_terms]

        if not chosen:
            chosen = kws[:max_terms]

        labels[c] = " / ".join(w.title() for w in chosen)

    return labels


def search_articles(query, vectorizer, X, df, top_n=7):
    q = basic_clean(query)
    q_lem = spacy_lemmatize(pd.Series([q]))[0]
    q_vec = vectorizer.transform([q_lem])

    sims = cosine_similarity(q_vec, X).ravel()
    top_idx = np.argsort(sims)[::-1][:top_n]

    results = df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]

    return results


def main():
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found in dataset.")

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    if "title" not in df.columns:
        df["title"] = "No title"

    print("1. Cleaning text...")
    df["clean_text"] = df[TEXT_COL].map(basic_clean)

    print("2. Lemmatizing text...")
    df["lemmas"] = spacy_lemmatize(df["clean_text"])

    print("3. Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        dtype=np.float32
    )
    X = vectorizer.fit_transform(df["lemmas"])

    print("4. Clustering articles...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X)
    df["cluster"] = labels

    print("5. Extracting keywords...")
    cluster_kws = top_keywords_per_cluster(vectorizer, X, labels, TOP_KW_PER_CLUSTER)
    cluster_names = make_cluster_label_auto(cluster_kws)
    df["cluster_name"] = df["cluster"].map(cluster_names)

    output_file = "articles_clustered.csv"
    df.to_csv(output_file, index=False)
    print(f"\nClustered dataset saved as: {output_file}\n")

    print("=" * 90)
    print("CLUSTER SUMMARY")
    print("=" * 90)

    for c in range(N_CLUSTERS):
        cluster_size = (df["cluster"] == c).sum()
        print(f"\nCluster {c}: {cluster_names[c]}")
        print(f"Number of articles: {cluster_size}")
        print("Top keywords:")
        print(", ".join(cluster_kws[c]))

        sample_titles = df[df["cluster"] == c]["title"].head(5).tolist()
        print("Sample articles:")
        for i, title in enumerate(sample_titles, 1):
            print(f"  {i}. {title}")

    print("\n" + "=" * 90)
    print("SEARCH MODE")
    print("=" * 90)

    while True:
        query = input("\nEnter a search phrase (or type 'q' to quit): ").strip()
        if query.lower() in ("q", "quit", "exit"):
            print("Program finished.")
            break

        results = search_articles(query, vectorizer, X, df, SEARCH_TOP_N)

        print("\nTop matching articles:")
        for i, (_, row) in enumerate(results.iterrows(), 1):
            print(f"\nResult #{i}")
            print(f"Title      : {row.get('title', 'No title')}")
            print(f"Cluster    : {row.get('cluster_name', 'Unknown')}")
            print(f"Similarity : {row.get('similarity', 0):.4f}")
            print(f"Text       : {str(row.get(TEXT_COL, ''))[:300]}...")


if __name__ == "__main__":
    main()