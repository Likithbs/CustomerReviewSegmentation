import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration
st.set_page_config(
    page_title="Review Management System",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
API_KEY = st.secrets.get("API_KEY", "demo-key-12345")
HEADERS = {"X-API-Key": API_KEY}

# Styling
st.markdown("""
<style>
    .review-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #f44336;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #FF9800;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def make_request(method, endpoint, **kwargs):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        kwargs['headers'] = HEADERS
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def get_sentiment_color(sentiment):
    """Get color class for sentiment"""
    colors = {
        'positive': 'sentiment-positive',
        'negative': 'sentiment-negative',
        'neutral': 'sentiment-neutral'
    }
    return colors.get(sentiment, '')

def star_rating(rating):
    """Display star rating"""
    return "⭐" * rating + "☆" * (5 - rating)

# Sidebar Navigation
st.sidebar.title("🏢 Review Manager")
page = st.sidebar.radio(
    "Navigate",
    ["📥 Ingest", "📊 Dashboard", "🔍 Browse & Search", "📈 Analytics", "🔎 Similar Search"]
)

# Check API Health
with st.sidebar.expander("🔧 System Status"):
    health = make_request("GET", "/health")
    if health:
        st.success("✅ API Connected")
        st.json(health)
    else:
        st.error("❌ API Disconnected")

# Page: Ingest Reviews
if page == "📥 Ingest":
    st.title("📥 Ingest Customer Reviews")
    st.markdown("Upload reviews in JSON format to analyze sentiment and topics.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload JSON File")
        uploaded_file = st.file_uploader("Choose a JSON file", type=['json'])
        
        if uploaded_file:
            try:
                reviews_data = json.load(uploaded_file)
                
                # Validate format
                if isinstance(reviews_data, list):
                    reviews_data = {"reviews": reviews_data}
                
                st.success(f"Loaded {len(reviews_data.get('reviews', []))} reviews")
                
                # Preview
                df_preview = pd.DataFrame(reviews_data['reviews'])
                st.dataframe(df_preview.head(10), use_container_width=True)
                
                if st.button("🚀 Ingest Reviews", type="primary"):
                    with st.spinner("Processing reviews..."):
                        result = make_request("POST", "/ingest", json=reviews_data)
                        if result:
                            st.success(f"✅ {result['message']}")
                            st.balloons()
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.subheader("Manual Entry")
        with st.form("manual_review"):
            location = st.text_input("Location", value="New York")
            rating = st.slider("Rating", 1, 5, 5)
            text = st.text_area("Review Text", height=150)
            date = st.date_input("Date", datetime.now())
            
            if st.form_submit_button("Add Review"):
                review_data = {
                    "reviews": [{
                        "location": location,
                        "rating": rating,
                        "text": text,
                        "date": date.isoformat()
                    }]
                }
                result = make_request("POST", "/ingest", json=review_data)
                if result:
                    st.success("✅ Review added!")
    
    # Sample JSON format
    with st.expander("📋 Sample JSON Format"):
        sample = {
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
        st.json(sample)

# Page: Dashboard
elif page == "📊 Dashboard":
    st.title("📊 Dashboard Overview")
    
    # Get analytics
    analytics = make_request("GET", "/analytics")
    
    if analytics:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{analytics['total_reviews']}</h2>
                <p>Total Reviews</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            positive_count = analytics['sentiment_counts'].get('positive', 0)
            positive_pct = (positive_count / analytics['total_reviews'] * 100) if analytics['total_reviews'] > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h2>{positive_pct:.1f}%</h2>
                <p>Positive Reviews</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_rating = sum(int(r) * c for r, c in analytics['rating_distribution'].items()) / analytics['total_reviews'] if analytics['total_reviews'] > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h2>{avg_rating:.1f} ⭐</h2>
                <p>Average Rating</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            negative_count = analytics['sentiment_counts'].get('negative', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h2>{negative_count}</h2>
                <p>Need Attention</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Distribution")
            if analytics['sentiment_counts']:
                fig = px.pie(
                    values=list(analytics['sentiment_counts'].values()),
                    names=list(analytics['sentiment_counts'].keys()),
                    color=list(analytics['sentiment_counts'].keys()),
                    color_discrete_map={'positive': '#4CAF50', 'negative': '#f44336', 'neutral': '#FF9800'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available")
        
        with col2:
            st.subheader("⭐ Rating Distribution")
            if analytics['rating_distribution']:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(analytics['rating_distribution'].keys()),
                        y=list(analytics['rating_distribution'].values()),
                        marker_color='#667eea'
                    )
                ])
                fig.update_layout(
                    xaxis_title="Rating",
                    yaxis_title="Count",
                    xaxis=dict(tickmode='linear', tick0=1, dtick=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rating data available")
        
        # Topics
        st.subheader("🏷️ Topic Distribution")
        if analytics['topic_counts']:
            topic_df = pd.DataFrame({
                'Topic': list(analytics['topic_counts'].keys()),
                'Count': list(analytics['topic_counts'].values())
            }).sort_values('Count', ascending=False)
            
            fig = px.bar(
                topic_df,
                x='Topic',
                y='Count',
                color='Count',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic data available")

# Page: Browse & Search
elif page == "🔍 Browse & Search":
    st.title("🔍 Browse & Search Reviews")
    
    # Filters
    with st.expander("🔧 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location_filter = st.text_input("Location", placeholder="e.g., New York")
        
        with col2:
            sentiment_filter = st.selectbox("Sentiment", ["All", "positive", "negative", "neutral"])
            sentiment_param = None if sentiment_filter == "All" else sentiment_filter
        
        with col3:
            search_query = st.text_input("Search Text", placeholder="Search in reviews...")
    
    # Pagination
    col1, col2 = st.columns([3, 1])
    with col1:
        page_num = st.number_input("Page", min_value=1, value=1, step=1)
    with col2:
        page_size = st.selectbox("Per Page", [10, 25, 50, 100], index=0)
    
    # Fetch reviews
    params = {
        "page": page_num,
        "page_size": page_size
    }
    if location_filter:
        params["location"] = location_filter
    if sentiment_param:
        params["sentiment"] = sentiment_param
    if search_query:
        params["q"] = search_query
    
    reviews_data = make_request("GET", "/reviews", params=params)
    
    if reviews_data and reviews_data['reviews']:
        st.info(f"Showing {len(reviews_data['reviews'])} of {reviews_data['total']} reviews (Page {page_num}/{reviews_data['total_pages']})")
        
        # Display reviews
        for review in reviews_data['reviews']:
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    sentiment_class = get_sentiment_color(review.get('sentiment', ''))
                    st.markdown(f"""
                    <div class="review-card">
                        <h4>📍 {review['location']} - {star_rating(review['rating'])}</h4>
                        <p><strong>Sentiment:</strong> <span class="{sentiment_class}">{review.get('sentiment', 'N/A').upper()}</span> | 
                        <strong>Topic:</strong> {review.get('topic', 'N/A')} | 
                        <strong>Date:</strong> {review['date'][:10]}</p>
                        <p>{review['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button(f"💬 Suggest Reply", key=f"reply_{review['id']}"):
                        st.session_state[f'show_reply_{review["id"]}'] = True
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{review['id']}", type="secondary"):
                        st.session_state[f'confirm_delete_{review["id"]}'] = True
                
                # Show reply if requested
                if st.session_state.get(f'show_reply_{review["id"]}', False):
                    with st.spinner("Generating reply..."):
                        reply_data = make_request("POST", f"/reviews/{review['id']}/suggest-reply")
                        
                        if reply_data:
                            st.success("✨ AI-Generated Reply:")
                            
                            # Editable reply
                            edited_reply = st.text_area(
                                "Edit reply if needed:",
                                value=reply_data['reply'],
                                height=100,
                                key=f"reply_text_{review['id']}"
                            )
                            
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if st.button("📋 Copy", key=f"copy_{review['id']}"):
                                    st.success("Copied to clipboard! (Use Ctrl+C)")
                            
                            # Tags and reasoning
                            with st.expander("🔍 Analysis Details"):
                                st.write("**Tags:**", reply_data['tags'])
                                st.write("**Reasoning:**", reply_data['reasoning_log'])
                    
                    if st.button("❌ Close", key=f"close_{review['id']}"):
                        st.session_state[f'show_reply_{review["id"]}'] = False
                        st.rerun()
                
                # Show delete confirmation
                if st.session_state.get(f'confirm_delete_{review["id"]}', False):
                    st.warning("⚠️ Are you sure you want to delete this review?")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        if st.button("✅ Yes, Delete", key=f"confirm_yes_{review['id']}", type="primary"):
                            result = make_request("DELETE", f"/reviews/{review['id']}")
                            if result:
                                st.success("✅ Review deleted successfully!")
                                st.session_state[f'confirm_delete_{review["id"]}'] = False
                                st.rerun()
                    
                    with col2:
                        if st.button("❌ Cancel", key=f"confirm_no_{review['id']}"):
                            st.session_state[f'confirm_delete_{review["id"]}'] = False
                            st.rerun()
                
                st.markdown("---")
    else:
        st.warning("No reviews found matching your criteria.")

# Page: Analytics
elif page == "📈 Analytics":
    st.title("📈 Advanced Analytics")
    
    analytics = make_request("GET", "/analytics")
    
    if analytics and analytics['total_reviews'] > 0:
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Insights", "📉 Trends"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Breakdown")
                sent_df = pd.DataFrame({
                    'Sentiment': list(analytics['sentiment_counts'].keys()),
                    'Count': list(analytics['sentiment_counts'].values())
                })
                fig = px.bar(sent_df, x='Sentiment', y='Count', color='Sentiment',
                           color_discrete_map={'positive': '#4CAF50', 'negative': '#f44336', 'neutral': '#FF9800'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Topic Analysis")
                topic_df = pd.DataFrame({
                    'Topic': list(analytics['topic_counts'].keys()),
                    'Count': list(analytics['topic_counts'].values())
                }).sort_values('Count', ascending=True)
                fig = px.bar(topic_df, x='Count', y='Topic', orientation='h', color='Count')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🎯 Key Insights")
            
            # Calculate insights
            total = analytics['total_reviews']
            positive_pct = (analytics['sentiment_counts'].get('positive', 0) / total * 100)
            negative_pct = (analytics['sentiment_counts'].get('negative', 0) / total * 100)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%", 
                         delta=f"{positive_pct - 50:.1f}% vs baseline")
                
                if negative_pct > 20:
                    st.warning(f"⚠️ {negative_pct:.1f}% negative reviews - attention needed!")
                else:
                    st.success("✅ Negative review rate is under control")
            
            with col2:
                # Top topic
                if analytics['topic_counts']:
                    top_topic = max(analytics['topic_counts'].items(), key=lambda x: x[1])
                    st.metric("Most Discussed Topic", top_topic[0].title(), 
                             f"{top_topic[1]} mentions")
                
                avg_rating = sum(int(r) * c for r, c in analytics['rating_distribution'].items()) / total
                st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
        
        with tab3:
            st.subheader("📉 Rating Trends")
            st.info("Trend analysis requires time-series data. Feature coming soon!")
            
            # Show rating distribution as a line chart
            rating_df = pd.DataFrame({
                'Rating': list(analytics['rating_distribution'].keys()),
                'Count': list(analytics['rating_distribution'].values())
            }).sort_values('Rating')
            
            fig = px.line(rating_df, x='Rating', y='Count', markers=True)
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("📭 No reviews available for analysis. Upload some reviews first!")

# Page: Similar Search (RAG-lite)
elif page == "🔎 Similar Search":
    st.title("🔎 Semantic Review Search")
    st.markdown("Find similar reviews using TF-IDF and cosine similarity.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Search query:", placeholder="e.g., great food but slow service")
    
    with col2:
        k = st.number_input("Results", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching for similar reviews..."):
            results = make_request("GET", "/search", params={"q": query, "k": k})
            
            if results and results.get('results'):
                st.success(f"Found {len(results['results'])} similar reviews")
                
                for idx, item in enumerate(results['results'], 1):
                    review = item['review']
                    similarity = item['similarity_score']
                    
                    with st.container():
                        col1, col2 = st.columns([5, 1])
                        
                        with col1:
                            sentiment_class = get_sentiment_color(review.get('sentiment', ''))
                            st.markdown(f"""
                            <div class="review-card">
                                <h4>#{idx} - 📍 {review['location']} - {star_rating(review['rating'])}</h4>
                                <p><strong>Similarity:</strong> {similarity:.3f} | 
                                <strong>Sentiment:</strong> <span class="{sentiment_class}">{review.get('sentiment', 'N/A').upper()}</span> | 
                                <strong>Topic:</strong> {review.get('topic', 'N/A')}</p>
                                <p>{review['text']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            progress_val = int(similarity * 100)
                            st.progress(progress_val)
                            st.caption(f"{progress_val}%")
                
            else:
                st.warning("No similar reviews found. Try a different query.")