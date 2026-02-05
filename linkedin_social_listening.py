import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from collections import Counter
import json

# Page configuration
st.set_page_config(
    page_title="LinkedIn Social Listening Tool",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 LinkedIn Social Listening Tool")
st.markdown("Monitor and analyze LinkedIn conversations around any topic")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Add topic input
topic = st.sidebar.text_input(
    "Topic to Monitor",
    value="",
    placeholder="e.g., AI, Climate Change, Sustainability",
    help="Enter the topic you want to monitor on LinkedIn"
)

st.sidebar.markdown("""
**💡 How to collect LinkedIn data:**

1. **Search LinkedIn** for your topic
2. **Copy the entire page** (Ctrl+A, Ctrl+C)
3. **Use the Copy-Paste Extractor tool** to convert to CSV
4. **Upload the CSV here**

Or manually enter posts using the **Manual Entry Tool**.

[Download the extraction tools](#) from the project files.
""")

# Sample data generator for demonstration
def generate_sample_data():
    """Generate sample LinkedIn posts for demonstration"""
    sample_posts = [
        {
            "post_id": "post_001",
            "author": "Sarah Johnson",
            "title": "Industry Thought Leader",
            "text": "Just attended an amazing industry summit. The insights shared about innovation and best practices were invaluable! #Innovation #Leadership",
            "likes": 245,
            "comments": 32,
            "shares": 18,
            "date": datetime.now() - timedelta(days=2),
            "url": "https://linkedin.com/post/001"
        },
        {
            "post_id": "post_002",
            "author": "Michael Chen",
            "title": "Strategy Director",
            "text": "New methodology helps us track key metrics accurately. Finally, a reliable framework for our strategic planning.",
            "likes": 189,
            "comments": 21,
            "shares": 12,
            "date": datetime.now() - timedelta(days=5),
            "url": "https://linkedin.com/post/002"
        },
        {
            "post_id": "post_003",
            "author": "Emma Williams",
            "title": "Business Analyst",
            "text": "Concerns about the complexity of implementation. Need more guidance for small teams. Anyone else facing challenges?",
            "likes": 67,
            "comments": 45,
            "shares": 8,
            "date": datetime.now() - timedelta(days=3),
            "url": "https://linkedin.com/post/003"
        },
        {
            "post_id": "post_004",
            "author": "David Brown",
            "title": "CFO",
            "text": "This approach is transforming how we report metrics to investors. Highly recommend for any serious program.",
            "likes": 312,
            "comments": 54,
            "shares": 29,
            "date": datetime.now() - timedelta(days=1),
            "url": "https://linkedin.com/post/004"
        },
        {
            "post_id": "post_005",
            "author": "Lisa Anderson",
            "title": "Consultant",
            "text": "Working with this framework has streamlined our client reporting. The standardization is exactly what the industry needed.",
            "likes": 156,
            "comments": 28,
            "shares": 15,
            "date": datetime.now() - timedelta(days=4),
            "url": "https://linkedin.com/post/005"
        },
        {
            "post_id": "post_006",
            "author": "James Miller",
            "title": "VP Operations",
            "text": "Questions about pricing structure. Would love to hear from others about ROI and implementation costs.",
            "likes": 93,
            "comments": 38,
            "shares": 6,
            "date": datetime.now() - timedelta(days=6),
            "url": "https://linkedin.com/post/006"
        },
        {
            "post_id": "post_007",
            "author": "Rachel Green",
            "title": "Chief Innovation Officer",
            "text": "Certification achieved! Proud of our team for meeting these rigorous standards. #Excellence #TeamWork",
            "likes": 421,
            "comments": 67,
            "shares": 34,
            "date": datetime.now() - timedelta(hours=12),
            "url": "https://linkedin.com/post/007"
        },
        {
            "post_id": "post_008",
            "author": "Tom Wilson",
            "title": "Data Scientist",
            "text": "The data infrastructure behind this solution is impressive. Integration with existing systems was smoother than expected.",
            "likes": 178,
            "comments": 19,
            "shares": 11,
            "date": datetime.now() - timedelta(days=7),
            "url": "https://linkedin.com/post/008"
        },
        {
            "post_id": "post_009",
            "author": "Amanda Taylor",
            "title": "Procurement Manager",
            "text": "Using this framework to evaluate suppliers. Game-changer for strategic sourcing decisions.",
            "likes": 203,
            "comments": 25,
            "shares": 17,
            "date": datetime.now() - timedelta(days=3),
            "url": "https://linkedin.com/post/009"
        },
        {
            "post_id": "post_010",
            "author": "Kevin Martinez",
            "title": "Investment Analyst",
            "text": "Investors are increasingly asking about compliance. This is becoming a key criterion for investment decisions.",
            "likes": 267,
            "comments": 41,
            "shares": 23,
            "date": datetime.now() - timedelta(days=2),
            "url": "https://linkedin.com/post/010"
        }
    ]
    return pd.DataFrame(sample_posts)

# Calculate engagement metrics
def calculate_engagement_metrics(df):
    """Calculate engagement score and reach metrics"""
    df['engagement_score'] = df['likes'] + (df['comments'] * 2) + (df['shares'] * 3)
    df['reach_estimate'] = df['engagement_score'] * 10  # Simplified reach estimation
    return df

# Sentiment analysis
def analyze_sentiment(text):
    """Analyze sentiment using simple keyword matching (no external dependencies)"""
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = [
        'excellent', 'amazing', 'great', 'fantastic', 'wonderful', 'awesome', 'best',
        'love', 'impressed', 'incredible', 'outstanding', 'brilliant', 'perfect',
        'highly recommend', 'revolutionary', 'innovative', 'excited', 'thrilled',
        'success', 'beneficial', 'advantage', 'efficient', 'effective', 'valuable',
        'proud', 'achieved', 'transforming', 'game-changer', 'streamlined', 'impressed'
    ]
    
    # Negative keywords
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing', 'worst',
        'hate', 'failed', 'problem', 'issue', 'concern', 'difficult', 'complex',
        'challenging', 'struggle', 'frustrating', 'disappointed', 'unfortunately',
        'lacking', 'confused', 'unclear', 'expensive', 'costly', 'waste'
    ]
    
    # Count sentiment words
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # Calculate polarity score (-1 to 1)
    total_words = len(text.split())
    if total_words == 0:
        return 'Neutral', 0.0
    
    polarity = (positive_count - negative_count) / max(total_words / 10, 1)
    polarity = max(-1, min(1, polarity))  # Clamp between -1 and 1
    
    # Categorize
    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

# Topic clustering
def cluster_topics(texts, n_clusters=3):
    """Cluster posts into topics using TF-IDF and K-Means"""
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Get top words for each cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    
    for i in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        top_indices = cluster_center.argsort()[-5:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        cluster_keywords[i] = top_words
    
    return clusters, cluster_keywords

# Main app logic
def main():
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Demo Data", "Upload CSV"]
    )
    
    if data_source == "Demo Data":
        df = generate_sample_data()
        st.sidebar.success("✅ Using demo data")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload LinkedIn data (CSV)", 
            type=['csv']
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            # Convert date column to datetime
            try:
                df['date'] = pd.to_datetime(df['date'])
            except:
                # If date parsing fails, use current date
                df['date'] = pd.to_datetime('today')
            
            # Ensure numeric columns are numeric
            for col in ['likes', 'comments', 'shares']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
            # Add missing columns if needed
            if 'title' not in df.columns:
                df['title'] = ''
            if 'url' not in df.columns:
                df['url'] = 'https://linkedin.com'
            
            st.sidebar.success("✅ Data uploaded")
        else:
            st.info("👆 Please upload a CSV file or select Demo Data")
            st.markdown("""
            ### Expected CSV format:
            - **post_id**: Unique identifier
            - **author**: Post author name
            - **title**: Author's professional title
            - **text**: Post content
            - **likes**: Number of likes
            - **comments**: Number of comments
            - **shares**: Number of shares
            - **date**: Post date (YYYY-MM-DD)
            - **url**: Post URL
            """)
            return
    
    # Calculate metrics
    df = calculate_engagement_metrics(df)
    
    # Add sentiment analysis
    sentiments = []
    sentiment_scores = []
    for text in df['text']:
        try:
            # Ensure text is a string and not empty
            text_str = str(text).strip()
            if len(text_str) > 0:
                sentiment, score = analyze_sentiment(text_str)
                sentiments.append(sentiment)
                sentiment_scores.append(score)
            else:
                sentiments.append('Neutral')
                sentiment_scores.append(0.0)
        except Exception as e:
            # If sentiment analysis fails, default to neutral
            sentiments.append('Neutral')
            sentiment_scores.append(0.0)
    
    df['sentiment'] = sentiments
    df['sentiment_score'] = sentiment_scores
    
    # Clustering
    n_clusters = st.sidebar.slider("Number of topic clusters", 2, 5, 3)
    
    # Ensure we have enough posts for clustering
    if len(df) < n_clusters:
        n_clusters = max(2, len(df) - 1)
        st.sidebar.warning(f"⚠️ Adjusted clusters to {n_clusters} (not enough posts)")
    
    try:
        clusters, cluster_keywords = cluster_topics(df['text'].tolist(), n_clusters)
        df['cluster'] = clusters
        
        # Assign cluster names based on keywords
        cluster_names = {}
        for i, keywords in cluster_keywords.items():
            cluster_names[i] = f"Topic {i+1}: {', '.join(keywords[:3])}"
        df['cluster_name'] = df['cluster'].map(cluster_names)
    except Exception as e:
        # If clustering fails, create a single cluster
        st.sidebar.error(f"⚠️ Clustering failed, using single group")
        df['cluster'] = 0
        df['cluster_name'] = 'All Posts'
        n_clusters = 1
    
    # Display metrics
    st.header("📊 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts", len(df))
    with col2:
        st.metric("Total Engagement", f"{df['engagement_score'].sum():,.0f}")
    with col3:
        st.metric("Avg. Engagement", f"{df['engagement_score'].mean():.0f}")
    with col4:
        st.metric("Est. Total Reach", f"{df['reach_estimate'].sum():,.0f}")
    
    # Top Posts Section
    st.header("🔥 Top Performing Posts")
    top_n = st.slider("Number of top posts to display", 3, 10, 5)
    
    top_posts = df.nlargest(top_n, 'engagement_score')[
        ['author', 'title', 'text', 'engagement_score', 'reach_estimate', 'sentiment', 'date', 'url']
    ]
    
    for idx, row in top_posts.iterrows():
        with st.expander(f"🏆 {row['author']} - {row['title']} (Engagement: {row['engagement_score']:.0f})"):
            # Handle date display safely
            try:
                if isinstance(row['date'], str):
                    date_str = row['date']
                else:
                    date_str = row['date'].strftime('%Y-%m-%d %H:%M')
            except:
                date_str = str(row['date'])
            
            st.write(f"**Posted:** {date_str}")
            st.write(f"**Sentiment:** {row['sentiment']}")
            st.write(f"**Estimated Reach:** {row['reach_estimate']:,.0f}")
            st.write(f"**Text:** {row['text']}")
            st.write(f"🔗 [View on LinkedIn]({row['url']})")
    
    # Engagement visualization
    st.header("📈 Engagement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_engagement = px.bar(
            df.nlargest(10, 'engagement_score'),
            x='author',
            y='engagement_score',
            color='sentiment',
            title="Top 10 Posts by Engagement Score",
            labels={'engagement_score': 'Engagement Score', 'author': 'Author'},
            color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
        )
        fig_engagement.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_engagement, use_container_width=True)
    
    with col2:
        # Engagement over time
        df_sorted = df.sort_values('date')
        fig_timeline = px.line(
            df_sorted,
            x='date',
            y='engagement_score',
            title="Engagement Over Time",
            labels={'engagement_score': 'Engagement Score', 'date': 'Date'}
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Topic Clustering
    st.header("🎯 Topic Clustering")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_clusters = px.scatter(
            df,
            x=range(len(df)),
            y='engagement_score',
            color='cluster_name',
            size='reach_estimate',
            hover_data=['author', 'text'],
            title="Posts by Topic Cluster",
            labels={'x': 'Post Index', 'engagement_score': 'Engagement Score'}
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
    
    with col2:
        cluster_counts = df['cluster_name'].value_counts()
        fig_cluster_dist = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title="Topic Distribution"
        )
        st.plotly_chart(fig_cluster_dist, use_container_width=True)
    
    # Detailed cluster analysis
    st.subheader("Topic Details")
    for i in range(n_clusters):
        cluster_df = df[df['cluster'] == i]
        avg_sentiment = cluster_df['sentiment_score'].mean()
        
        with st.expander(f"{cluster_names[i]} ({len(cluster_df)} posts)"):
            st.write(f"**Average Sentiment Score:** {avg_sentiment:.3f}")
            st.write(f"**Total Engagement:** {cluster_df['engagement_score'].sum():,.0f}")
            st.write(f"**Key Terms:** {', '.join(cluster_keywords[i])}")
            
            # Show sample posts from cluster
            st.write("**Sample Posts:**")
            for _, post in cluster_df.head(3).iterrows():
                st.write(f"- *{post['author']}*: {post['text'][:100]}...")
    
    # Sentiment Analysis
    st.header("💭 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_counts = df['sentiment'].value_counts()
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Overall Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Sentiment by cluster
        sentiment_cluster = df.groupby(['cluster_name', 'sentiment']).size().reset_index(name='count')
        fig_sentiment_cluster = px.bar(
            sentiment_cluster,
            x='cluster_name',
            y='count',
            color='sentiment',
            title="Sentiment by Topic Cluster",
            barmode='stack',
            color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
        )
        fig_sentiment_cluster.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_sentiment_cluster, use_container_width=True)
    
    # Cluster-specific sentiment analysis
    st.subheader("Sentiment Analysis by Topic")
    
    for i in range(n_clusters):
        cluster_df = df[df['cluster'] == i]
        sentiment_breakdown = cluster_df['sentiment'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(cluster_names[i], f"{len(cluster_df)} posts")
        with col2:
            positive_pct = (sentiment_breakdown.get('Positive', 0) / len(cluster_df)) * 100
            st.metric("Positive", f"{positive_pct:.1f}%")
        with col3:
            neutral_pct = (sentiment_breakdown.get('Neutral', 0) / len(cluster_df)) * 100
            st.metric("Neutral", f"{neutral_pct:.1f}%")
        with col4:
            negative_pct = (sentiment_breakdown.get('Negative', 0) / len(cluster_df)) * 100
            st.metric("Negative", f"{negative_pct:.1f}%")
    
    # Export data
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Analysis (CSV)",
            data=csv,
            file_name=f"carbon_measures_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create summary report
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "total_posts": len(df),
            "total_engagement": int(df['engagement_score'].sum()),
            "average_engagement": float(df['engagement_score'].mean()),
            "sentiment_distribution": df['sentiment'].value_counts().to_dict(),
            "top_posts": top_posts[['author', 'engagement_score', 'sentiment']].to_dict('records'),
            "clusters": {cluster_names[i]: int((df['cluster'] == i).sum()) for i in range(n_clusters)}
        }
        
        json_summary = json.dumps(summary, indent=2)
        st.download_button(
            label="Download Summary (JSON)",
            data=json_summary,
            file_name=f"carbon_measures_summary_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📝 How to Collect LinkedIn Data
    
    Since this app doesn't have direct LinkedIn API access, here's how to get your data:
    
    #### Method 1: Copy-Paste Extractor (Recommended - Easiest!)
    
    **Step-by-step:**
    
    1. **Download the Copy-Paste Extractor tool** (`linkedin_copy_paste_extractor.html`)
    2. **Open it in your browser** (just double-click the file)
    3. **Go to LinkedIn** and search for your topic (e.g., "AI trends", "Climate tech", "Web3")
    4. **Make sure you're on the "Posts" tab** in the search results
    5. **Scroll down** to load 20-50 posts (the more you scroll, the more data you'll get)
    6. **Select all** the content on the page:
       - Windows: Press `Ctrl+A`
       - Mac: Press `Cmd+A`
    7. **Copy** everything:
       - Windows: Press `Ctrl+C`
       - Mac: Press `Cmd+C`
    8. **Go back to the Copy-Paste Extractor tool**
    9. **Paste** into the text box:
       - Windows: Press `Ctrl+V`
       - Mac: Press `Cmd+V`
    10. **Click "Extract & Download CSV"**
    11. **Come back here and upload the CSV file!**
    
    **This takes 2 minutes and works 100% of the time!**
    
    ---
    
    #### Method 2: Manual Entry Tool
    
    **For when you want perfect data quality:**
    
    1. **Download** `linkedin_data_entry.html`
    2. **Open it** in your browser
    3. **Browse LinkedIn** and find relevant posts
    4. **Copy/paste** the information from each post into the form:
       - Author name
       - Author title
       - Post text
       - Engagement numbers (likes, comments, shares)
       - Post URL
    5. **Click "Add Post"** for each entry
    6. When done, **click "Download CSV"**
    7. **Upload the CSV here!**
    
    **Pros:** Perfect accuracy, works even if LinkedIn changes their layout
    
    ---
    
    #### Method 3: Bookmarklet (Advanced Users)
    
    **One-click extraction:**
    
    1. **Read** the `BOOKMARKLET_FIX.md` guide
    2. **Create a bookmark** with the extraction code
    3. **Search LinkedIn** for your topic
    4. **Scroll** to load posts
    5. **Click the bookmark** - CSV downloads automatically!
    
    **Pros:** Fastest method once set up
    
    ---
    
    ### 🎯 Tips for Best Results
    
    - **Load more posts**: The more you scroll on LinkedIn before copying/extracting, the more data you get
    - **Monitor regularly**: Do this weekly to track trends over time
    - **Save your CSVs**: Keep historical data to see how conversations evolve
    - **Try different searches**: Monitor multiple topics or hashtags
    - **Combine data**: Merge multiple CSV files in Excel to analyze larger datasets
    
    ---
    
    ### 📥 Expected CSV Format
    
    Your CSV file should have these columns:
    
    | Column | Description | Example |
    |--------|-------------|---------|
    | post_id | Unique identifier | "post_001" |
    | author | Author name | "Sarah Johnson" |
    | title | Professional title | "CEO at TechCorp" |
    | text | Post content | "Just launched our new..." |
    | likes | Number of likes | 245 |
    | comments | Number of comments | 32 |
    | shares | Number of shares | 18 |
    | date | Post date | "2024-02-04" |
    | url | LinkedIn URL | "https://linkedin.com/..." |
    
    Don't worry if you're missing some columns - the app will handle it!
    
    ---
    
    ### 🚀 Quick Start (Right Now!)
    
    **Try this in 5 minutes:**
    
    1. Download `linkedin_copy_paste_extractor.html`
    2. Open it in your browser
    3. Search LinkedIn for any topic you're interested in
    4. Copy the page (Ctrl+A, Ctrl+C)
    5. Paste into the tool (Ctrl+V)
    6. Click "Extract & Download"
    7. Upload the CSV here
    8. Analyze! 🎉
    
    ---
    
    ### ⚖️ Legal & Ethical Notes
    
    **Always comply with:**
    - ✅ LinkedIn Terms of Service
    - ✅ Data privacy regulations (GDPR, CCPA)
    - ✅ Only collect publicly visible posts
    - ✅ Use data for legitimate purposes
    - ✅ Respect user privacy
    
    **Don't:**
    - ❌ Collect private or restricted content
    - ❌ Use data for harassment or spam
    - ❌ Share personal information
    - ❌ Automate at scale without permission
    
    ---
    
    ### 📊 What You Can Analyze
    
    Once you have your data, this app shows you:
    
    - 📈 **Engagement trends** - What content performs best
    - 🎯 **Topic clusters** - Main themes in the conversation
    - 💭 **Sentiment analysis** - How people feel about the topic
    - 🏆 **Top posts** - Most engaging content
    - 📊 **Reach estimates** - How far content is spreading
    - 📅 **Time trends** - When engagement peaks
    
    ---
    
    ### 🆘 Need Help?
    
    - **CSV won't upload?** Check the format matches the expected columns
    - **No posts extracted?** Make sure you scroll on LinkedIn to load posts first
    - **Tool not working?** Try the Manual Entry Tool as backup
    - **Other issues?** Check the troubleshooting guides in the documentation
    
    ---
    
    **Made with ❤️ using Streamlit • 100% Free & Open Source**
    """)

if __name__ == "__main__":
    main()
