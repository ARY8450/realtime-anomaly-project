import re

CATEGORIES = {
    "earnings":  [r"\bearnings?\b", r"\bprofit\b", r"\brevenue\b", r"\bquarter\b", r"\bresults\b"],
    "policy":    [r"\bpolicy\b", r"\bregulation\b", r"\bRBI\b", r"\brules\b"],
    "outlook":   [r"\bguidance\b", r"\bforecast\b", r"\boutlook\b", r"\bupgrade\b"],
    "mna":        [r"\bm&a\b", r"\btakeover\b", r"\bmerger\b", r"\bdeal\b"],
    "sector":    [r"\bindustry\b", r"\bsector\b", r"\bsupply\b", r"\bdemand\b"]
}

def classify_category(text):
    """
    Classify the category of a given text using keyword-based matching.
    """
    text = text.lower()
    for category, keywords in CATEGORIES.items():
        if any(re.search(keyword, text) for keyword in keywords):
            return category
    return "other"  # Default category if no match

if __name__ == "__main__":
    # Test the classifier
    test_text = "The company's earnings report shows a strong performance this quarter."
    category = classify_category(test_text)
    print(f"Category: {category}")
