import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

N_CLUSTERS = 5
TOP_KW_PER_CLUSTER = 20
SEARCH_TOP_N = 7
INPUT_CSV = "articles.csv"
TEXT_COL = "text"

GENERIC = {
    "people", "person", "year", "time", "new", "world", "day", "life", "man", "woman", "child",
    "many", "state", "government", "news", "video", "percent", "one", "two", "three", "also",
    "say", "make", "get", "like", "work", "company", "group", "month", "week", "today",
    "yesterday", "way", "thing", "police", "official", "officer", "city", "country", "case",
    "report", "story", "issue", "public"
}


def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fast_preprocess(texts: pd.Series) -> list[str]:
    results = []

    for text in texts.astype(str):
        words = text.split()
        tokens = [
            STEMMER.stem(word)
            for word in words
            if word.isalpha()
            and len(word) > 2
            and word not in STOP_WORDS
        ]
        results.append(" ".join(tokens))

    return results


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

        labels[c] = " / ".join(w.title() for w in chosen) if chosen else f"Cluster {c}"

    return labels


class ArticleClusterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Article Clustering and Search System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        self.df = None
        self.vectorizer = None
        self.X = None
        self.cluster_keywords = None
        self.cluster_names = None
        self.model_ready = False

        self.build_ui()

    def build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(
            top_frame,
            text="Article Clustering and Search System",
            font=("Arial", 16, "bold")
        ).pack(anchor="w", pady=(0, 10))

        controls_frame = ttk.Frame(top_frame)
        controls_frame.pack(fill="x")

        self.load_btn = ttk.Button(
            controls_frame,
            text="Load and Process Dataset",
            command=self.start_processing
        )
        self.load_btn.pack(side="left", padx=(0, 10))

        self.status_var = tk.StringVar(value="Status: waiting to load articles.csv")
        ttk.Label(controls_frame, textvariable=self.status_var).pack(side="left", padx=10)

        search_frame = ttk.LabelFrame(self.root, text="Search Articles", padding=10)
        search_frame.pack(fill="x", padx=10, pady=10)

        self.search_entry = ttk.Entry(search_frame, font=("Arial", 11))
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.search_entry.bind("<Return>", lambda event: self.search_articles())

        self.search_btn = ttk.Button(search_frame, text="Search", command=self.search_articles)
        self.search_btn.pack(side="left")

        content_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        content_pane.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left_frame = ttk.LabelFrame(content_pane, text="Clusters and Keywords", padding=10)
        right_frame = ttk.LabelFrame(content_pane, text="Results / Articles", padding=10)

        content_pane.add(left_frame, weight=1)
        content_pane.add(right_frame, weight=1)

        self.cluster_text = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD, font=("Consolas", 10)
        )
        self.cluster_text.pack(fill="both", expand=True)

        self.result_text = scrolledtext.ScrolledText(
            right_frame, wrap=tk.WORD, font=("Consolas", 10)
        )
        self.result_text.pack(fill="both", expand=True)

    def set_status(self, text):
        self.status_var.set(f"Status: {text}")
        self.root.update_idletasks()

    def start_processing(self):
        self.load_btn.config(state="disabled")
        self.search_btn.config(state="disabled")
        self.cluster_text.delete("1.0", tk.END)
        self.result_text.delete("1.0", tk.END)

        thread = threading.Thread(target=self.process_dataset, daemon=True)
        thread.start()

    def process_dataset(self):
        try:
            self.set_status("loading dataset")

            df = pd.read_csv(INPUT_CSV)

            if TEXT_COL not in df.columns:
                raise ValueError(f"Column '{TEXT_COL}' not found in dataset.")

            df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

            if "title" not in df.columns:
                df["title"] = "No title"

            self.set_status("cleaning text")
            df["clean_text"] = df[TEXT_COL].map(basic_clean)

            self.set_status("Stemming")
            df["processed_text"] = fast_preprocess(df["clean_text"])

            self.set_status("vectorizing text")
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.8,
                dtype=np.float32
            )
            X = vectorizer.fit_transform(df["processed_text"])

            self.set_status("clustering articles")
            kmeans = KMeans(
                n_clusters=N_CLUSTERS,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(X)
            df["cluster"] = labels

            self.set_status("extracting keywords")
            cluster_keywords = top_keywords_per_cluster(vectorizer, X, labels, TOP_KW_PER_CLUSTER)
            cluster_names = make_cluster_label_auto(cluster_keywords)
            df["cluster_name"] = df["cluster"].map(cluster_names)

            df.to_csv("articles_clustered.csv", index=False)

            self.df = df
            self.vectorizer = vectorizer
            self.X = X
            self.cluster_keywords = cluster_keywords
            self.cluster_names = cluster_names
            self.model_ready = True

            self.root.after(0, self.show_clusters)
            self.root.after(0, lambda: self.set_status("ready"))
            self.root.after(0, lambda: self.search_btn.config(state="normal"))
            self.root.after(0, lambda: self.load_btn.config(state="normal"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.set_status("error"))
            self.root.after(0, lambda: self.load_btn.config(state="normal"))
            self.root.after(0, lambda: self.search_btn.config(state="normal"))

    def show_clusters(self):
        self.cluster_text.delete("1.0", tk.END)

        if self.df is None or self.cluster_keywords is None:
            return

        for c in range(N_CLUSTERS):
            cluster_name = self.cluster_names.get(c, f"Cluster {c}")
            cluster_size = (self.df["cluster"] == c).sum()
            keywords = self.cluster_keywords.get(c, [])

            self.cluster_text.insert(
                tk.END,
                f"CLUSTER {c}: {cluster_name}\n"
                f"Number of articles: {cluster_size}\n"
                f"Top keywords:\n"
                f"{', '.join(keywords)}\n\n"
                + "=" * 90 + "\n\n"
        )

    def search_articles(self):
        if not self.model_ready:
            messagebox.showwarning("Warning", "Please load and process the dataset first.")
            return

        query = self.search_entry.get().strip()
        if not query:
            messagebox.showinfo("Info", "Please enter a search phrase.")
            return

        q = basic_clean(query)
        q_processed = fast_preprocess(pd.Series([q]))[0]
        q_vec = self.vectorizer.transform([q_processed])

        sims = cosine_similarity(q_vec, self.X).ravel()
        top_idx = np.argsort(sims)[::-1][:SEARCH_TOP_N]

        results = self.df.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"Search query: {query}\n")
        self.result_text.insert(tk.END, "=" * 90 + "\n\n")

        for i, (_, row) in enumerate(results.iterrows(), 1):
            self.result_text.insert(tk.END, f"Result #{i}\n")
            self.result_text.insert(tk.END, f"Title      : {row.get('title', 'No title')}\n")
            self.result_text.insert(tk.END, f"Cluster    : {row.get('cluster_name', 'Unknown')}\n")
            self.result_text.insert(tk.END, f"Similarity : {row.get('similarity', 0):.4f}\n")
            self.result_text.insert(tk.END, "Article text:\n")
            self.result_text.insert(tk.END, str(row.get(TEXT_COL, ""))[:700] + "...\n")
            self.result_text.insert(tk.END, "\n" + "-" * 90 + "\n\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ArticleClusterApp(root)
    root.mainloop()