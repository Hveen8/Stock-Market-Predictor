import pandas as pd
import numpy as np
from google import genai

# --- Gemini API Embedding Function ---
client = genai.Client(api_key="GEMINI_API_KEY")  # Remember to censor KEY!!!!

def get_article_embedding(article_text):
    """
    Uses the Gemini API to embed the article text.
    Returns a numpy array representing the embedding.
    """
    result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=article_text
    )
    return np.array(result.embeddings)


def generate_target_label(article_text):
    """
    Generates a target label for an article based on simple heuristics.
    Returns:
        label: A list in the format [relevance, up_weight, down_weight, unchanged_weight]
    """
    text = article_text.lower()
    
    relevance_terms = [
        # High Confidence Names (1.0)
        ("apple inc.", 1.0),
        ("apple incorporated", 1.0),
        ("apple computer", 1.0),
        ("apple corporation", 1.0),
        ("apple co.", 1.0),
        ("apple headquarters", 1.0),
        ("cupertino-based company", 1.0),

        # Stock Ticker
        ("aapl", 0.5),
        ("aapl shares", 0.75),
        ("aapl stock", 0.75),
        ("nasdaq:aapl", 0.75),

        # General Apple Mentions (0.75)
        ("apple", 0.75),
        ("apple brand", 0.75),
        ("apple products", 0.75),
        ("apple ecosystem", 0.75),

        # Key People
        ("steve jobs", 0.75),
        ("tim cook", 0.75),
        ("jonny ive", 0.75),
        ("phil schiller", 0.75),
        ("craig federighi", 0.75),
        ("luca maestri", 0.75),

        # Major Products
        ("iphone", 0.75),
        ("iphone 15", 0.75),
        ("iphone 14", 0.75),
        ("iphone pro", 0.75),
        ("ipad", 0.75),
        ("ipad pro", 0.75),
        ("ipad air", 0.75),
        ("macbook", 0.75),
        ("macbook pro", 0.75),
        ("macbook air", 0.75),
        ("mac studio", 0.75),
        ("mac pro", 0.75),
        ("mac mini", 0.75),
        ("mac os", 0.75),
        ("macos", 0.75),

        # Services
        ("apple music", 0.75),
        ("apple tv", 0.75),
        ("apple tv+", 0.75),
        ("apple arcade", 0.75),
        ("apple news", 0.75),
        ("apple fitness+", 0.75),
        ("icloud", 0.75),
        ("apple id", 0.75),

        # Hardware
        ("apple watch", 0.75),
        ("apple pencil", 0.75),
        ("apple silicon", 0.75),
        ("m1 chip", 0.75),
        ("m2 chip", 0.75),
        ("m3 chip", 0.75),
        ("t2 chip", 0.75),
        ("homepod", 0.75),
        ("airpods", 0.75),
        ("airpods pro", 0.75),
        ("airpods max", 0.75),
        ("vision pro", 0.75),
        ("apple glasses", 0.75),

        # Operating Systems
        ("ios", 0.75),
        ("ios 17", 0.75),
        ("ipados", 0.75),
        ("macos ventura", 0.75),
        ("macos sonoma", 0.75),
        ("watchos", 0.75),
        ("tvos", 0.75),

        # Financials and Market
        ("apple stock", 0.75),
        ("apple shares", 0.75),
        ("apple earnings", 0.75),
        ("apple revenue", 0.75),
        ("apple profits", 0.75),
        ("apple forecast", 0.75),
        ("apple quarterly results", 0.75),
        ("apple guidance", 0.75),
        ("apple investors", 0.75),
        ("apple market cap", 0.75),
        ("apple dividends", 0.75),
        ("apple buybacks", 0.75),

        # News & Events
        ("apple event", 0.75),
        ("apple keynote", 0.75),
        ("wwdc", 0.75),
        ("apple launch", 0.75),
        ("spring event", 0.75),
        ("fall event", 0.75),

        # Legal & Regulatory
        ("apple lawsuit", 0.75),
        ("apple antitrust", 0.75),
        ("apple vs epic", 0.75),
        ("apple regulation", 0.75),
        ("apple privacy", 0.75),

        # Business Activities
        ("apple acquisition", 0.75),
        ("apple partnership", 0.75),
        ("apple investment", 0.75),
        ("apple r&d", 0.75),
        ("apple supply chain", 0.75),
        ("apple manufacturing", 0.75),
        ("foxconn", 0.75),
        ("apple retail", 0.75),
        ("apple online store", 0.75),

        # Technology & Innovation
        ("apple innovation", 0.75),
        ("apple ai", 0.75),
        ("apple machine learning", 0.75),
        ("apple chips", 0.75),
        ("apple ar", 0.75),
        ("apple vr", 0.75),
        ("apple car", 0.75),
        ("project titan", 0.75),
        ("apple security", 0.75),
        ("apple encryption", 0.75),

        # Software & Apps
        ("apple store", 0.75),
        ("app store", 0.75),
        ("apple developer", 0.75),
        ("xcode", 0.75),
        ("apple sdk", 0.75),
        ("testflight", 0.75),
        ("apple beta", 0.75),

        # Medium Relevance (0.5)
        ("smartphone market", 0.5),
        ("consumer electronics", 0.5),
        ("wearables", 0.5),
        ("tech giant", 0.5),
        ("big tech", 0.5),
        ("us tech stock", 0.5),
        ("silicon valley", 0.5),
        ("hardware company", 0.5),
        ("tablet market", 0.5),
        ("smartwatch sales", 0.5),
        ("mobile os", 0.5),
        ("voice assistant", 0.5),
        ("supply chain disruption", 0.5),

        # Low Relevance (0.25)
        ("consumer trends", 0.25),
        ("technology adoption", 0.25),
        ("semiconductor trends", 0.25),
        ("software update", 0.25),
        ("ai assistant", 0.25),
        ("smart home", 0.25),
        ("us markets", 0.25),
        ("tech stocks", 0.25),
        ("electronics retail", 0.25),
        ("app monetization", 0.25),
        ("digital marketplace", 0.25),
        ("cloud storage", 0.25),
        ("eco-friendly tech", 0.25),

        # Irrelevant (0.0)
        ("banana", 0.0),
        ("orange fruit", 0.0),
        ("pineapple", 0.0),
        ("fruit basket", 0.0),
        ("orchard", 0.0),
        ("apple pie", 0.0),
        ("apple cider", 0.0),
        ("fruit nutrition", 0.0),
        ("apple picking", 0.0),
        ("red delicious", 0.0),
    ]

    # --- Relevance Score ---
    relevance = 0.0  # Default relevance
    for term, score in relevance_terms:
        if term in text:
            relevance = max(relevance, score)

    # --- Directional Weights ---
    # Define keywords for directional sentiment related to business performance.
    up_terms = [
        "growth", "profit", "increase", "rise", "gain", "surge", "expansion",
        "record", "strong", "improved", "bullish", "beat", "outperform", "upgrade"
    ]
    down_terms = [
        "loss", "decline", "drop", "fall", "decrease", "slump", "weak",
        "bearish", "downgrade", "miss", "underperform", "cut", "reduction"
    ]
    unchanged_terms = [
        "stable", "steady", "unchanged", "flat", "consistent", "no change", "sideways"
    ]
    
    # Count occurrences for each category
    def count_terms(text, terms):
        count = 0
        for term in terms:
            count += text.count(term)
        return count

    up_count = count_terms(text, up_terms)
    down_count = count_terms(text, down_terms)
    unchanged_count = count_terms(text, unchanged_terms)
    
    total = up_count + down_count + unchanged_count
    if total == 0:
        # If no keywords found, default to a neutral distribution
        up_weight, down_weight, unchanged_weight = 0.0, 0.0, 1.0
    else:
        up_weight = up_count / total
        down_weight = down_count / total
        unchanged_weight = unchanged_count / total

    # Return the label in the format: [relevance, up, down, unchanged]
    return [relevance, up_weight, down_weight, unchanged_weight]


class ArticleProcessor:
    def __init__(self, excel_path, sheet_name=None):
        """
        Loads the Excel file into a DataFrame.
        Assumes the first column is a date and subsequent columns contain article text.
        """
        self.data = pd.read_excel(excel_path, sheet_name=sheet_name)

    import numpy as np