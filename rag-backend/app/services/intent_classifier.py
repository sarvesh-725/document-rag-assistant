import re
import logging
import inspect
from pathlib import Path
from typing import Literal

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("intent_classifier")

import os
from dotenv import load_dotenv

load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
_model = None
classifier = LogisticRegression()


class _HashingEncoder:
    """Offline fallback with the same encode interface as SentenceTransformer."""

    def __init__(self) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=384, alternate_sign=False, norm="l2", ngram_range=(1, 2)
        )

    def encode(self, texts):
        return self.vectorizer.transform(texts)


def _get_model():
    """Load the semantic encoder lazily and tolerate package-version drift."""
    global _model
    if _model is not None:
        return _model

    cache_root = Path(__file__).resolve().parents[1] / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    try:
        from sentence_transformers import SentenceTransformer

        signature = inspect.signature(SentenceTransformer)
        kwargs = {}
        if "token" in signature.parameters and hf_token:
            kwargs["token"] = hf_token
        if "backend" in signature.parameters:
            kwargs["backend"] = "onnx"
        _model = SentenceTransformer("all-MiniLM-L6-v2", **kwargs)
    except Exception as exc:
        logger.warning("Semantic encoder unavailable; using local hashing fallback: %s", exc)
        _model = _HashingEncoder()
    return _model

TRAINING_DATA = [
    # Chitchat Examples
    ("sup", "chitchat"), ("how's it going", "chitchat"), ("tell me a joke", "chitchat"),
    ("what is up", "chitchat"), ("how are things", "chitchat"), ("good evening", "chitchat"),
    ("you rock", "chitchat"), ("haha that is funny", "chitchat"), ("talk to you later", "chitchat"),
    
    # Specific Fact Examples (Numbers, dates, specific facts)
    ("what was q3 revenue", "knowledge_specific"), 
    ("how much did the company make in 2023", "knowledge_specific"),
    ("what is the total budget", "knowledge_specific"),
    ("who is the ceo", "knowledge_specific"),
    ("who is the ceo of apple", "knowledge_specific"),
    ("when was the report published", "knowledge_specific"),
    ("what is the capital of france", "knowledge_specific"),
    ("tell me a fact about history", "knowledge_broad"),
    ("what is the highest mountain", "knowledge_specific"),
    
    # Broad Knowledge Examples (Explanations, Lists, Summaries)
    ("explain the revenue drop", "knowledge_broad"), 
    ("list all projects", "knowledge_broad"),
    ("summarize the financial report", "knowledge_broad"),
    ("what are the key takeaways", "knowledge_broad"),
    ("give me an overview of the strategy", "knowledge_broad"),
    ("explain how the new policy works", "knowledge_broad")
]

CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|yo|greetings|hola|heyy+)\b",
    r"^(bye|goodbye|see ya|ttyl|exit|quit)\b",
    r"\b(thanks|thank you|ty|appreciate it|cheers)\b",
    r"^how are you(\s*doing)?\??$",
    r"^(good\s*)(morning|afternoon|evening|night)\b"
]

def train_classifier():
    """Trains the Tier 2 classifier in memory. Takes < 1 second."""
    texts, labels = zip(*TRAINING_DATA)
    embeddings = _get_model().encode(texts)
    classifier.fit(embeddings, labels)
    logger.info("Tier 2 Semantic Classifier trained and ready!")

def check_tier1_heuristics(query: str) -> bool:
    """Returns True if the query matches obvious chitchat patterns."""
    cleaned_query = query.strip().lower()
    for pattern in CHITCHAT_PATTERNS:
        if re.search(pattern, cleaned_query):
            return True
    return False

def classify_intent(query: str) -> Literal["chitchat", "knowledge_specific", "knowledge_broad"]:
    """
    Classifies the user query into one of three categories:
    - chitchat
    - knowledge_specific
    - knowledge_broad
    """
    # ---- Tier 1: Quick Rule Check ----
    if check_tier1_heuristics(query):
        logger.info(f"Classified intent: 'chitchat' (Tier 1 Heuristics) for query: '{query}'")
        return "chitchat"
    
    # ---- Tier 2: Semantic Embedding Check ----
    if not hasattr(classifier, "classes_"):
        train_classifier()
    query_embedding = _get_model().encode([query])
    prediction = classifier.predict(query_embedding)[0]
    
    # Confidence threshold for chitchat
    if prediction == "chitchat":
        probabilities = classifier.predict_proba(query_embedding)[0]
        classes = list(classifier.classes_)
        if "chitchat" in classes:
            chitchat_prob = probabilities[classes.index("chitchat")]
            if chitchat_prob < 0.70:
                logger.info(f"Classified intent: 'knowledge_broad' (Fallback due to low chitchat confidence {chitchat_prob:.2f}) for query: '{query}'")
                return "knowledge_broad"

    logger.info(f"Classified intent: '{prediction}' (Tier 2 Semantic) for query: '{query}'")
    return prediction
