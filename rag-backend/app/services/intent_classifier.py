import re
import logging
from typing import Literal

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("intent_classifier")

# Initialize the embedding model.
# Note: Using "onnx" backend if available can reduce RAM usage, but defaults to standard if not configured.
import os
from dotenv import load_dotenv

load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2", backend="onnx", token=hf_token)
except Exception:
    logger.warning("ONNX backend failed or not available for SentenceTransformer. Falling back to standard backend.")
    model = SentenceTransformer("all-MiniLM-L6-v2", token=hf_token)

classifier = LogisticRegression()

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
    embeddings = model.encode(texts)
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
    query_embedding = model.encode([query])
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
