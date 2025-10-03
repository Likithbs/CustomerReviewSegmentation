from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlite3
import os
import re
from contextlib import contextmanager
import json

# AI/ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Optional: Hugging Face transformers for local fallback
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Optional: OpenAI for LLM API
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

app = FastAPI(
    title="Multi-Location Review Management API",
    description="AI-powered customer review analysis and response system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
API_KEY = os.getenv("API_KEY", "demo-key-12345")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "reviews.db")

# Initialize AI models
sentiment_analyzer = None
summarizer = None

if HF_AVAILABLE:
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"Warning: Could not load HF models: {e}")

# Pydantic Models
class Review(BaseModel):
    id: Optional[int] = None
    location: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5)
    text: str = Field(..., min_length=1)
    date: str
    sentiment: Optional[str] = None
    topic: Optional[str] = None

    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except:
            raise ValueError('Invalid date format')

class IngestRequest(BaseModel):
    reviews: List[Review]

class SuggestReplyResponse(BaseModel):
    reply: str
    tags: Dict[str, str]
    reasoning_log: str

class SearchResult(BaseModel):
    review: Review
    similarity_score: float

class AnalyticsResponse(BaseModel):
    sentiment_counts: Dict[str, int]
    topic_counts: Dict[str, int]
    rating_distribution: Dict[int, int]
    total_reviews: int

# Database
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT NOT NULL,
                rating INTEGER NOT NULL,
                text TEXT NOT NULL,
                date TEXT NOT NULL,
                sentiment TEXT,
                topic TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

# Security
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Helper Functions
def analyze_sentiment_from_rating(rating: int, text: str = "") -> str:
    """
    Primary sentiment analysis based on star rating.
    Star ratings are more reliable than text analysis.
    
    Rating mapping:
    5 stars -> positive
    4 stars -> positive
    3 stars -> neutral
    2 stars -> negative
    1 star  -> negative
    """
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:  # rating <= 2
        return 'negative'

def analyze_sentiment_text_only(text: str) -> str:
    """Fallback text-based sentiment analysis (only used when rating is unavailable)"""
    if sentiment_analyzer:
        try:
            result = sentiment_analyzer(text[:512])[0]
            label = result['label'].lower()
            score = result['score']
            
            if label == 'positive' and score > 0.6:
                return 'positive'
            elif label == 'negative' and score > 0.6:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            print(f"HF sentiment analysis error: {e}")
    
    # Enhanced fallback: weighted keyword-based with negation handling
    positive_words = {
        'excellent': 2, 'amazing': 2, 'outstanding': 2, 'fantastic': 2, 'superb': 2,
        'wonderful': 2, 'perfect': 2, 'love': 2, 'loved': 2, 'best': 2,
        'great': 1.5, 'good': 1.5, 'nice': 1.5, 'pleasant': 1.5, 'enjoyed': 1.5,
        'delicious': 1.5, 'awesome': 1.5, 'incredible': 1.5, 'exceptional': 1.5,
        'recommend': 1, 'happy': 1, 'satisfied': 1, 'impressed': 1, 'fresh': 1,
        'clean': 1, 'friendly': 1, 'helpful': 1, 'quick': 1, 'tasty': 1,
        'beautiful': 1, 'lovely': 1, 'enjoy': 1, 'favorite': 1.5, 'favourite': 1.5
    }
    
    negative_words = {
        'terrible': 2, 'horrible': 2, 'awful': 2, 'disgusting': 2, 'worst': 2,
        'pathetic': 2, 'atrocious': 2, 'unacceptable': 2, 'appalling': 2,
        'bad': 1.5, 'poor': 1.5, 'disappointing': 1.5, 'disappointed': 1.5,
        'mediocre': 1, 'bland': 1, 'cold': 1, 'rude': 1, 'slow': 1, 'dirty': 1,
        'overpriced': 1, 'expensive': 0.5, 'lacking': 1, 'unfortunately': 0.5,
        'never': 1, 'avoid': 1.5, 'waste': 1.5, 'regret': 1.5, 'angry': 1.5,
        'hate': 2, 'hated': 2, 'dislike': 1.5, 'disliked': 1.5, 'unhappy': 1.5,
        'unsatisfied': 1.5, 'unpleasant': 1.5, 'nasty': 1.5, 'gross': 1.5
    }
    
    # Negation words that flip sentiment
    negations = {'not', 'no', 'never', "n't", 'neither', 'nor', 'nothing', 'nowhere', 'nobody', 'none', "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't"}
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b|n\'t', text_lower)
    
    pos_score = 0
    neg_score = 0
    
    # Track negation context
    negation_window = 3  # words after negation to consider
    i = 0
    while i < len(words):
        word = words[i]
        
        # Check if current word is negation
        is_negated = False
        if i > 0:
            # Check previous words for negation
            for j in range(max(0, i - negation_window), i):
                if words[j] in negations:
                    is_negated = True
                    break
        
        # Score the word
        if word in positive_words:
            if is_negated:
                neg_score += positive_words[word]  # Flip to negative
            else:
                pos_score += positive_words[word]
        elif word in negative_words:
            if is_negated:
                pos_score += negative_words[word] * 0.5  # Flip to positive (but weaker)
            else:
                neg_score += negative_words[word]
        
        i += 1
    
    # Consider exclamation marks as intensity boosters
    exclamation_count = text.count('!')
    if exclamation_count > 0:
        if pos_score > neg_score:
            pos_score *= (1 + exclamation_count * 0.1)
        elif neg_score > pos_score:
            neg_score *= (1 + exclamation_count * 0.1)
    
    # Decision logic with threshold
    threshold = 0.5  # Minimum difference to be decisive
    
    if pos_score > neg_score + threshold:
        return 'positive'
    elif neg_score > pos_score + threshold:
        return 'negative'
    else:
        return 'neutral'

def get_sentiment_intensity(rating: int, text: str) -> str:
    """
    Get a more granular sentiment with intensity based on rating and text analysis.
    Returns: very_positive, positive, neutral, negative, very_negative
    """
    base_sentiment = analyze_sentiment_from_rating(rating, text)
    
    # Check for extreme language in text
    extreme_positive = ['amazing', 'excellent', 'outstanding', 'perfect', 'incredible', 'love', 'loved', 'best ever', 'fantastic']
    extreme_negative = ['terrible', 'horrible', 'worst', 'disgusting', 'awful', 'hate', 'hated', 'never again', 'pathetic']
    
    text_lower = text.lower()
    
    if base_sentiment == 'positive':
        if rating == 5 and any(word in text_lower for word in extreme_positive):
            return 'very_positive'
        return 'positive'
    elif base_sentiment == 'negative':
        if rating == 1 and any(word in text_lower for word in extreme_negative):
            return 'very_negative'
        return 'negative'
    else:
        return 'neutral'

def extract_topic(text: str) -> str:
    """Extract primary topic from review text"""
    topics = {
        'service': ['service', 'staff', 'employee', 'waiter', 'server', 'friendly', 'helpful', 'rude', 'waiting', 'waited', 'wait', 'customer service', 'manager', 'attention', 'attentive'],
        'food': ['food', 'meal', 'dish', 'taste', 'flavor', 'delicious', 'bland', 'fresh', 'cooked', 'menu', 'appetizer', 'entree', 'dessert', 'pizza', 'burger', 'salad', 'chicken', 'beef', 'seafood', 'portion', 'quality'],
        'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitary', 'spotless', 'messy', 'bathroom', 'restroom', 'table', 'floor', 'neat', 'tidy', 'filthy'],
        'price': ['price', 'expensive', 'cheap', 'value', 'cost', 'affordable', 'overpriced', 'worth', 'money', 'dollar', 'pricing', 'bill', 'charged'],
        'ambiance': ['ambiance', 'atmosphere', 'decor', 'music', 'lighting', 'cozy', 'noisy', 'quiet', 'crowded', 'space', 'seating', 'comfortable', 'interior', 'vibe', 'setting']
    }
    
    text_lower = text.lower()
    topic_scores = {}
    
    for topic, keywords in topics.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            topic_scores[topic] = score
    
    if topic_scores:
        return max(topic_scores.items(), key=lambda x: x[1])[0]
    return 'general'

def redact_sensitive_info(text: str) -> str:
    """Redact emails and phone numbers"""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    return text

def generate_reply_local(review: Review) -> SuggestReplyResponse:
    """Generate reply using local models or templates"""
    sentiment = review.sentiment or analyze_sentiment_from_rating(review.rating, review.text)
    topic = review.topic or extract_topic(review.text)
    
    reasoning_log = f"Rating-based Sentiment: {sentiment}, Topic: {topic}, Stars: {review.rating}/5"
    
    # Template-based reply generation with rating-specific responses
    if review.rating == 5:
        reply = f"Thank you so much for the amazing 5-star review! We're absolutely thrilled to hear you had such a wonderful experience at our {review.location} location"
        if topic != 'general':
            reply += f", especially regarding our {topic}"
        reply += ". We can't wait to serve you again!"
        
    elif review.rating == 4:
        reply = f"Thank you for your great 4-star feedback! We're delighted you enjoyed your visit to our {review.location} location"
        if topic != 'general':
            reply += f", particularly our {topic}"
        reply += ". We're always striving for 5 stars, so please let us know how we can make your next visit even better!"
        
    elif review.rating == 3:
        reply = f"Thank you for taking the time to share your feedback about our {review.location} location. We appreciate your honest 3-star review"
        if topic != 'general':
            reply += f" regarding {topic}"
        reply += ". We're always working to improve and would love the opportunity to provide you with a better experience. Please feel free to reach out to our management team with any specific suggestions."
        
    elif review.rating == 2:
        reply = f"We're truly sorry to hear that your experience at our {review.location} location didn't meet your expectations. Your 2-star review is concerning to us"
        if topic != 'general':
            reply += f", especially regarding {topic}"
        reply += ". We'd like to make this right. Please contact our management team directly so we can address your concerns and ensure a better experience next time."
        
    else:  # rating == 1
        reply = f"We sincerely apologize for the disappointing experience you had at our {review.location} location. A 1-star review tells us we fell far short of our standards"
        if topic != 'general':
            reply += f", particularly with {topic}"
        reply += ". This is unacceptable, and we want to make it right. Please contact our management team immediately so we can address this personally and regain your trust."
    
    # Summarize if available
    if summarizer and len(review.text) > 100:
        try:
            summary = summarizer(review.text[:1024], max_length=50, min_length=10, do_sample=False)[0]['summary_text']
            reasoning_log += f"\nText Summary: {summary}"
        except:
            pass
    
    return SuggestReplyResponse(
        reply=redact_sensitive_info(reply),
        tags={"sentiment": sentiment, "topic": topic, "rating": str(review.rating)},
        reasoning_log=reasoning_log
    )

async def generate_reply_openai(review: Review) -> SuggestReplyResponse:
    """Generate reply using OpenAI API"""
    if not OPENAI_API_KEY:
        return generate_reply_local(review)
    
    sentiment = review.sentiment or analyze_sentiment_from_rating(review.rating, review.text)
    topic = review.topic or extract_topic(review.text)
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""Generate a professional, empathetic response to this customer review:

Location: {review.location}
Rating: {review.rating}/5 stars
Review: {review.text}

Guidelines:
- Be concise (2-3 sentences)
- Show empathy and professionalism
- Address the specific star rating given
- For 5-star: Express gratitude and excitement
- For 4-star: Thank them and ask how to improve to 5 stars
- For 3-star: Acknowledge feedback and commitment to improvement
- For 2-star: Apologize and offer to make it right
- For 1-star: Sincerely apologize and urgently request contact
- Address specific concerns mentioned
- Invite further contact if appropriate

Response:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer service assistant for a multi-location business. You understand the importance of star ratings in customer reviews."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        reply = response.choices[0].message.content.strip()
        reasoning_log = f"Generated via OpenAI API. Rating-based Sentiment: {sentiment}, Topic: {topic}, Stars: {review.rating}/5"
        
        return SuggestReplyResponse(
            reply=redact_sensitive_info(reply),
            tags={"sentiment": sentiment, "topic": topic, "rating": str(review.rating)},
            reasoning_log=reasoning_log
        )
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return generate_reply_local(review)

# API Endpoints
@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ai_backend": "openai" if OPENAI_API_KEY else ("huggingface" if HF_AVAILABLE else "local"),
        "sentiment_method": "rating-based (primary)",
        "models_loaded": {
            "sentiment": sentiment_analyzer is not None,
            "summarizer": summarizer is not None
        }
    }

@app.post("/ingest", dependencies=[Depends(verify_api_key)])
async def ingest_reviews(request: IngestRequest):
    """Ingest and analyze multiple reviews"""
    with get_db() as conn:
        inserted_count = 0
        for review in request.reviews:
            # Use rating-based sentiment analysis (primary method)
            sentiment = analyze_sentiment_from_rating(review.rating, review.text)
            topic = extract_topic(review.text)
            
            conn.execute("""
                INSERT INTO reviews (location, rating, text, date, sentiment, topic)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (review.location, review.rating, review.text, review.date, sentiment, topic))
            inserted_count += 1
        
        conn.commit()
    
    return {
        "message": f"Successfully ingested {inserted_count} reviews", 
        "count": inserted_count,
        "sentiment_method": "rating-based"
    }

@app.get("/reviews", dependencies=[Depends(verify_api_key)])
async def get_reviews(
    location: Optional[str] = None,
    sentiment: Optional[str] = None,
    rating: Optional[int] = Query(None, ge=1, le=5),
    q: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    """Get reviews with filtering and pagination"""
    with get_db() as conn:
        query = "SELECT * FROM reviews WHERE 1=1"
        params = []
        
        if location:
            query += " AND location = ?"
            params.append(location)
        
        if sentiment:
            query += " AND sentiment = ?"
            params.append(sentiment)
        
        if rating:
            query += " AND rating = ?"
            params.append(rating)
        
        if q:
            query += " AND text LIKE ?"
            params.append(f"%{q}%")
        
        # Count total
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        total = conn.execute(count_query, params).fetchone()[0]
        
        # Paginate
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([page_size, (page - 1) * page_size])
        
        rows = conn.execute(query, params).fetchall()
        reviews = [dict(row) for row in rows]
    
    return {
        "reviews": reviews,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }

@app.get("/reviews/{review_id}", dependencies=[Depends(verify_api_key)])
async def get_review(review_id: int):
    """Get a single review by ID"""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Review not found")
        return dict(row)

@app.delete("/reviews/{review_id}", dependencies=[Depends(verify_api_key)])
async def delete_review(review_id: int):
    """Delete a review by ID"""
    with get_db() as conn:
        # Check if review exists
        row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Review not found")
        
        # Delete the review
        conn.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
        conn.commit()
    
    return {
        "message": "Review deleted successfully",
        "id": review_id
    }

@app.post("/reviews/{review_id}/suggest-reply", dependencies=[Depends(verify_api_key)])
async def suggest_reply(review_id: int):
    """Generate AI-powered reply suggestion"""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Review not found")
        
        review = Review(**dict(row))
    
    # Generate reply using available AI backend
    if OPENAI_API_KEY:
        response = await generate_reply_openai(review)
    else:
        response = generate_reply_local(review)
    
    return response

@app.get("/analytics", dependencies=[Depends(verify_api_key)])
async def get_analytics(location: Optional[str] = None):
    """Get analytics summary with optional location filter"""
    with get_db() as conn:
        base_query = "SELECT COUNT(*) FROM reviews"
        params = []
        
        if location:
            base_query += " WHERE location = ?"
            params.append(location)
        
        total = conn.execute(base_query, params).fetchone()[0]
        
        # Sentiment counts
        sentiment_query = "SELECT sentiment, COUNT(*) as count FROM reviews"
        if location:
            sentiment_query += " WHERE location = ?"
        sentiment_query += " GROUP BY sentiment"
        sentiment_rows = conn.execute(sentiment_query, params).fetchall()
        
        # Topic counts
        topic_query = "SELECT topic, COUNT(*) as count FROM reviews"
        if location:
            topic_query += " WHERE location = ?"
        topic_query += " GROUP BY topic"
        topic_rows = conn.execute(topic_query, params).fetchall()
        
        # Rating distribution
        rating_query = "SELECT rating, COUNT(*) as count FROM reviews"
        if location:
            rating_query += " WHERE location = ?"
        rating_query += " GROUP BY rating"
        rating_rows = conn.execute(rating_query, params).fetchall()
    
    return AnalyticsResponse(
        sentiment_counts={row[0]: row[1] for row in sentiment_rows if row[0]},
        topic_counts={row[0]: row[1] for row in topic_rows if row[0]},
        rating_distribution={row[0]: row[1] for row in rating_rows},
        total_reviews=total
    )

@app.get("/analytics/by-location", dependencies=[Depends(verify_api_key)])
async def get_analytics_by_location():
    """Get analytics grouped by location"""
    with get_db() as conn:
        locations = conn.execute("SELECT DISTINCT location FROM reviews").fetchall()
        
        result = {}
        for loc_row in locations:
            location = loc_row[0]
            
            # Get stats for this location
            total = conn.execute("SELECT COUNT(*) FROM reviews WHERE location = ?", (location,)).fetchone()[0]
            
            sentiment_rows = conn.execute("""
                SELECT sentiment, COUNT(*) as count 
                FROM reviews 
                WHERE location = ?
                GROUP BY sentiment
            """, (location,)).fetchall()
            
            rating_rows = conn.execute("""
                SELECT rating, COUNT(*) as count 
                FROM reviews 
                WHERE location = ?
                GROUP BY rating
            """, (location,)).fetchall()
            
            avg_rating = conn.execute("""
                SELECT AVG(rating) 
                FROM reviews 
                WHERE location = ?
            """, (location,)).fetchone()[0]
            
            result[location] = {
                "total_reviews": total,
                "average_rating": round(avg_rating, 2) if avg_rating else 0,
                "sentiment_counts": {row[0]: row[1] for row in sentiment_rows if row[0]},
                "rating_distribution": {row[0]: row[1] for row in rating_rows}
            }
    
    return result

@app.get("/search", dependencies=[Depends(verify_api_key)])
async def search_similar_reviews(q: str, k: int = Query(5, ge=1, le=20)):
    """Search for similar reviews using TF-IDF and cosine similarity"""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM reviews").fetchall()
        if not rows:
            return {"results": []}
        
        reviews = [dict(row) for row in rows]
        texts = [r['text'] for r in reviews]
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([q])
            
            # Cosine similarity
            similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include relevant results
                    results.append(SearchResult(
                        review=Review(**reviews[idx]),
                        similarity_score=float(similarities[idx])
                    ))
            
            return {"results": results, "query": q}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)