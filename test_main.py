import pytest
from fastapi.testclient import TestClient
from main import app, init_db, API_KEY
import os
import tempfile

# Setup test database
test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
os.environ['DB_PATH'] = test_db.name

client = TestClient(app)
headers = {"X-API-Key": API_KEY}

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    yield
    os.unlink(test_db.name)

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_ingest_reviews_happy_path():
    """Test successful review ingestion"""
    payload = {
        "reviews": [
            {
                "location": "New York",
                "rating": 5,
                "text": "Excellent service and amazing food!",
                "date": "2024-01-15T10:30:00"
            },
            {
                "location": "Boston",
                "rating": 2,
                "text": "Poor service, food was cold",
                "date": "2024-01-16T14:20:00"
            }
        ]
    }
    
    response = client.post("/ingest", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert "Successfully ingested" in data["message"]

def test_ingest_reviews_missing_api_key():
    """Test ingestion without API key"""
    payload = {
        "reviews": [
            {
                "location": "New York",
                "rating": 5,
                "text": "Great!",
                "date": "2024-01-15T10:30:00"
            }
        ]
    }
    
    response = client.post("/ingest", json=payload)
    assert response.status_code == 422  # Missing header

def test_ingest_reviews_invalid_rating():
    """Test ingestion with invalid rating"""
    payload = {
        "reviews": [
            {
                "location": "New York",
                "rating": 6,  # Invalid: > 5
                "text": "Great!",
                "date": "2024-01-15T10:30:00"
            }
        ]
    }
    
    response = client.post("/ingest", json=payload, headers=headers)
    assert response.status_code == 422

def test_get_reviews():
    """Test retrieving reviews"""
    # First ingest some reviews
    payload = {
        "reviews": [
            {
                "location": "NYC",
                "rating": 5,
                "text": "Excellent!",
                "date": "2024-01-15T10:30:00"
            },
            {
                "location": "LA",
                "rating": 3,
                "text": "Average experience",
                "date": "2024-01-16T14:20:00"
            }
        ]
    }
    client.post("/ingest", json=payload, headers=headers)
    
    # Get reviews
    response = client.get("/reviews", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "reviews" in data
    assert len(data["reviews"]) == 2
    assert data["total"] == 2

def test_get_reviews_with_filter():
    """Test filtering reviews by location"""
    # Ingest reviews
    payload = {
        "reviews": [
            {"location": "NYC", "rating": 5, "text": "Great!", "date": "2024-01-15T10:30:00"},
            {"location": "LA", "rating": 3, "text": "OK", "date": "2024-01-16T14:20:00"},
            {"location": "NYC", "rating": 4, "text": "Good", "date": "2024-01-17T09:00:00"}
        ]
    }
    client.post("/ingest", json=payload, headers=headers)
    
    # Filter by location
    response = client.get("/reviews?location=NYC", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert all(r["location"] == "NYC" for r in data["reviews"])

def test_get_review_by_id():
    """Test retrieving single review"""
    # Ingest a review
    payload = {
        "reviews": [
            {"location": "NYC", "rating": 5, "text": "Amazing!", "date": "2024-01-15T10:30:00"}
        ]
    }
    client.post("/ingest", json=payload, headers=headers)
    
    # Get the review
    response = client.get("/reviews/1", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert data["location"] == "NYC"

def test_get_review_not_found():
    """Test retrieving non-existent review"""
    response = client.get("/reviews/999", headers=headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_suggest_reply():
    """Test AI reply suggestion"""
    # Ingest a review
    payload = {
        "reviews": [
            {"location": "NYC", "rating": 5, "text": "Excellent service!", "date": "2024-01-15T10:30:00"}
        ]
    }
    client.post("/ingest", json=payload, headers=headers)
    
    # Get suggestion
    response = client.post("/reviews/1/suggest-reply", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "tags" in data
    assert "reasoning_log" in data
    assert len(data["reply"]) > 0

def test_analytics():
    """Test analytics endpoint"""
    # Ingest reviews
    payload = {
        "reviews": [
            {"location": "NYC", "rating": 5, "text": "Great food!", "date": "2024-01-15T10:30:00"},
            {"location": "LA", "rating": 2, "text": "Poor service", "date": "2024-01-16T14:20:00"},
            {"location": "NYC", "rating": 4, "text": "Good ambiance", "date": "2024-01-17T09:00:00"}
        ]
    }
    client.post("/ingest", json=payload, headers=headers)
    
    response = client.get("/analytics", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment_counts" in data
    assert "topic_counts" in data
    assert "rating_distribution" in data
    assert data["total_reviews"] == 3

def test_search_similar_reviews():
    """Test TF-IDF search"""
    # Ingest reviews
    payload = {
        "reviews": [
            {"location": "NYC", "rating": 5, "text": "The pizza was amazing and delicious", "date": "2024-01-15T10:30:00"},
            {"location": "LA", "rating": 4, "text": "Great pasta and wonderful service", "date": "2024-01-16T14:20:00"},
            {"location": "NYC", "rating": 3, "text": "The ambiance was nice but food was cold", "date": "2024-01-17T09:00:00"}
        ]
    }
    client.post("/ingest", json=payload, headers=headers)
    
    response = client.get("/search?q=delicious pizza", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
    # First result should be most similar
    assert data["results"][0]["similarity_score"] > 0