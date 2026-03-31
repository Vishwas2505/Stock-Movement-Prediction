import feedparser

def get_latest_news(ticker):
    # Fetch RSS feed for the specific stock
    url = f"https://news.google.com/rss/search?q={ticker}+stock+when:1d"
    feed = feedparser.parse(url)
    
    # Get the top 3 headlines
    headlines = [entry.title for entry in feed.entries[:3]]
    return " ".join(headlines) if headlines else "Neutral market trend."