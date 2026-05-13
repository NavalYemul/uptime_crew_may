"""
nlp_basics.py — NLP Fundamentals
==================================
Covers: NLP pipeline, tokenization, stop words, stemming vs lemmatization,
        Bag of Words, TF-IDF, text statistics.

Run:  python -m day1.nlp_basics
"""

from __future__ import annotations

import string
from collections import Counter

import numpy as np
import nltk

# Download NLTK data silently on first run
_NLTK_RESOURCES = [
    "punkt", "punkt_tab", "stopwords",
    "wordnet", "omw-1.4", "averaged_perceptron_tagger",
]
for _r in _NLTK_RESOURCES:
    nltk.download(_r, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ══════════════════════════════════════════════════════
# WHAT IS NLP?
# ══════════════════════════════════════════════════════
"""
NLP = making computers understand & generate human language.

Key tasks (roughly in order of difficulty):
  Tokenization     → split text into words / sentences
  POS Tagging      → label: noun, verb, adjective...
  NER              → "Apple" (company) vs "apple" (fruit)
  Sentiment        → positive / negative / neutral
  Machine Translation
  Summarization
  Question Answering / Chatbots (LLMs = state of the art today)

Today: tokenization + basic preprocessing + BoW + TF-IDF
"""

# ══════════════════════════════════════════════════════
# SAMPLE DATA
# ══════════════════════════════════════════════════════

SAMPLE_TEXT = """
Natural Language Processing is a fascinating field of Artificial Intelligence.
It enables computers to understand, interpret, and generate human language.
Applications include chatbots, translation systems, and sentiment analysis.
Machine learning models power most modern NLP systems.
"""

CORPUS: list[str] = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand text.",
    "Word embeddings represent words as dense numerical vectors.",
    "Scikit-learn provides simple and efficient tools for data mining.",
    "Python is the most popular language for machine learning.",
    "Neural networks are inspired by the structure of the human brain.",
    "Feature extraction is a critical step in NLP pipelines.",
    "Text classification assigns categories to documents automatically.",
    "Tokenization is the first step in any NLP workflow.",
]


# ══════════════════════════════════════════════════════
# TOKENIZATION
# ══════════════════════════════════════════════════════

def tokenize_words(text: str) -> list[str]:
    """
    Split text into word tokens using NLTK's Punkt tokenizer.
    Smarter than text.split() — handles contractions, punctuation correctly.
    Java: StringTokenizer / String.split("\\s+") but linguistically aware.
    """
    return word_tokenize(text)


def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences. NLTK uses Punkt (unsupervised) model."""
    return sent_tokenize(text.strip())


# ══════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════

def normalize_tokens(tokens: list[str]) -> list[str]:
    """Lowercase + remove punctuation tokens."""
    return [t.lower() for t in tokens if t not in string.punctuation]


def remove_stopwords(tokens: list[str], language: str = "english") -> list[str]:
    """
    Remove high-frequency low-meaning words.
    Stop words: "the", "is", "at", "which", "on", "a", "an" ...
    These dilute similarity calculations without adding semantic value.
    """
    stop_words = set(stopwords.words(language))
    return [t for t in tokens if t not in stop_words]


def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Stemming — rule-based suffix stripping.
    'running' → 'run', 'studies' → 'studi'  (may not be a real word)
    Fast but crude. Good for search indexing.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Lemmatization — dictionary-based root form.
    'running' → 'run', 'studies' → 'study', 'better' → 'good'
    Slower but linguistically correct. Preferred for NLU tasks.
    """
    lem = WordNetLemmatizer()
    return [lem.lemmatize(t) for t in tokens]


def full_preprocess(text: str) -> list[str]:
    """
    Full NLP preprocessing pipeline:
    tokenize → normalize → remove stopwords → lemmatize
    """
    tokens = tokenize_words(text)
    tokens = normalize_tokens(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens


# ══════════════════════════════════════════════════════
# BAG OF WORDS (BoW)
# ══════════════════════════════════════════════════════

def bag_of_words(corpus: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Bag of Words: each document → word frequency vector.

    "I love Python" → {love:1, Python:1, ...} → [0, 0, 1, 0, 1, ...]
                                                   ^each column = a vocab word

    Limitation: ORDER is lost.
    "dog bites man" == "man bites dog" in BoW representation.
    """
    vec = CountVectorizer(stop_words="english", max_features=20)
    bow = vec.fit_transform(corpus).toarray()
    vocab = vec.get_feature_names_out().tolist()
    return bow, vocab


def tfidf_vectors(corpus: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    TF-IDF: Term Frequency × Inverse Document Frequency

    TF(t,d)  = (count of t in d) / (total terms in d)
    IDF(t)   = log(N / df(t))        — penalises very common words
    TF-IDF   = TF × IDF              — high for rare-but-important terms

    "machine" appears in many ML docs → low IDF → low TF-IDF
    "tokenisation" appears rarely     → high IDF → high TF-IDF
    """
    vec = TfidfVectorizer(stop_words="english", max_features=20)
    tfidf = vec.fit_transform(corpus).toarray()
    vocab = vec.get_feature_names_out().tolist()
    return tfidf, vocab


def word_frequency(tokens: list[str], top_n: int = 10) -> list[tuple[str, int]]:
    """Return top-N most frequent words."""
    return Counter(tokens).most_common(top_n)


# ══════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("NLP FUNDAMENTALS DEMO")
    print("=" * 60)

    # Tokenization
    tokens = tokenize_words(SAMPLE_TEXT)
    sentences = tokenize_sentences(SAMPLE_TEXT)
    print(f"\n[Tokens]    first 10 : {tokens[:10]}")
    print(f"[Sentences] count    : {len(sentences)}")

    # Stem vs Lemmatize comparison
    test_words = ["running", "studies", "better", "caring", "flies", "went"]
    stemmer = PorterStemmer()
    lem = WordNetLemmatizer()
    print(f"\n{'Word':12} {'Stemmed':14} {'Lemmatized':14}")
    print("─" * 42)
    for w in test_words:
        print(f"{w:12} {stemmer.stem(w):14} {lem.lemmatize(w):14}")

    # Full pipeline
    processed = full_preprocess(SAMPLE_TEXT)
    print(f"\n[Full pipeline] {processed}")

    # BoW matrix
    bow, vocab = bag_of_words(CORPUS)
    print(f"\n[BoW] vocab ({len(vocab)} words): {vocab}")
    print(f"[BoW] matrix shape: {bow.shape}")

    # TF-IDF
    tfidf, vocab_tf = tfidf_vectors(CORPUS)
    print(f"\n[TF-IDF] shape: {tfidf.shape}")
    print(f"[TF-IDF] vocab: {vocab_tf}")

    # Top words in corpus
    all_tokens = full_preprocess(" ".join(CORPUS))
    freq = word_frequency(all_tokens, top_n=10)
    print(f"\n[Top 10 words in corpus]: {freq}")
