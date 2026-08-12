from app.services.intent_classifier import train_classifier, classify_intent

def test():
    train_classifier()
    
    queries = [
        "hi there",
        "how's it going?",
        "what is the Q3 revenue?",
        "explain the revenue drop",
        "tell me a joke",
        "what are the key takeaways from the document?"
    ]
    
    for q in queries:
        intent = classify_intent(q)
        print(f"Query: '{q}' -> Intent: '{intent}'")

if __name__ == "__main__":
    test()
