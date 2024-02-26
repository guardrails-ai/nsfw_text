import nltk
from transformers import pipeline

# Download NLTK data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
print("NLTK stuff loaded successfully.")

# Load pipeline once before actual initialization
# to avoid downloading during runtime
try:
    pipe = pipeline(
        "text-classification",
        model="michellejieli/NSFW_text_classifier",
    )
    print("Initial pipeline setup successful.")
except Exception as e:
    raise RuntimeError(f"Failed to setup pipeline: {e}") from e
