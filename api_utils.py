#api_utils.py
from gnews import GNews
from newspaper import Article
import re
from googlesearch import search
from typing import List, Dict
from pytrends.request import TrendReq
import pandas as pd

import feedparser
import json
from dateutil import parser as date_parser
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import ast
from llm_utils import *

import json
import urllib.request
import urllib.parse
from config import GNEWS_API_KEY, PERPLEXITY_API_KEY
from datetime import datetime, timedelta

#import nltk
#nltk.download('punkt')


# Get json with news metadata, url is as google news rss
def get_news(keywords: str, num_articles: int = 3):
    """
	
    Fetches a list of news articles based on the given keywords.

    Parameters:

        keywords (str): The keywords to search for.

        num_articles (int): The maximum number of articles to return. Defaults to 3.
    Returns:
        List[Dict]: A list of news articles (each as a dictionary) from GNews.
                    Returns None if no articles are found.
    """

    if not isinstance(keywords, str) or not keywords.strip():
        raise ValueError("`keywords` must be a non-empty string.")
    if not isinstance(num_articles, int) or num_articles <= 0:
        raise ValueError("`num_articles` must be a positive integer.")
    
    google_news = GNews()
    articles = google_news.get_news(keywords)[:num_articles]
    if articles:
        return articles
    else:
        return None

#get newspart from the perplexity
def generate_search_bar_news_perplexity(keywords, API_KEY=PERPLEXITY_API_KEY):
    date = (datetime.now()- timedelta(days=30)).strftime("%d.%m.%Y")
    system_content = f"""You are the assistant that will help quickly get information on a certain topic to support small talk.

            Output Rules:
                          Return the full, detailed JSON object with the following structure:

                                 {{
                            "news": {{
                                     "news_text": 3 bullet points with the news, you will get the news and need to summarise them,
                                                  you must provide 3 sentances, the news must be connected to the egeneral information you
                                                  provided before, if the news ar ecompletly from the another field, indicate that
                                     "news_fun_facts": fun fact about the news,
                                     "picture_url": url of the picture about this topic
                             }},
                               
                             "interesting_trivia": {{
                                     "trivia_text": 3 interesting anecdotes or stories or memes, they must be provided as separate sentances,
                                     "trivia_fun_fact": funcfacts about the trivia,
                                     "picture_url":url of the picture about this topic
                             }},
                               
                             "opinions": {{
                                     "opinions_text": opinions of the other people about this topic,
                                     "opinions_fun_fact": fun cfact about the opinions
                        
                             }}
                                }}

                        Critical Requirements:
                        - Always respond in valid JSON only (no extra commentary, no markdown).
                        - Do not include any additional text or explanation outside the JSON.

                        Be precise and coincise, **Don`t use Wikipedia or Wikimedia.**
                        The latest news can be from the {date}.

                    """
    user_input= f"""
                Focus on the keywords {keywords}.

                Find three recent news about the keywords.  
                Generate an engaging and concise news summary in JSON format. 
                Assume the reader knows little about the topic, but needs to be able to keep a high-level conversation.
        """



    output = generate_perplexity_output(system_content, user_input)
    return output['choices'][0]['message']['content']


'''
def resolve_real_urls_only(news, wait_time=1):
    """
    Resolves Google News RSS redirect URLs and adds the real article URL to each item.
    
    Parameters:
        news (list of dict): Each dict should contain a 'url' key with a Google News RSS article link.
        wait_time (int): Seconds to wait after redirection.
    
    Returns:
        list of dict: Updated list with each item including 'resolved_url' key.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("user-agent=Mozilla/5.0")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    updated_news = []

    try:
        for i, item in enumerate(news):
            print(item)
            rss_url = item.get("url")
            if not rss_url:
                continue

            print(f"🔄 ({i+1}/{len(news)}) Resolving: {rss_url}")
            try:
                driver.get(rss_url)
                time.sleep(wait_time)
                real_url = driver.current_url

                if "news.google.com" in real_url:
                    print(f"❌ Could not resolve: {rss_url}")
                    continue

                item["url"] = real_url
                updated_news.append(item)
                print(f"✅ Resolved → {real_url}")

            except Exception as e:
                print(f"❌ Error resolving {rss_url}: {e}")
                continue

    finally:
        driver.quit()

    return updated_news


'''

'''
def add_text_and_image_to_articles(news_items, min_word_count=100):
    """
    Scrapes article text and top image URL from resolved URLs, and adds them to each news item.
    
    Parameters:
        news_items (list): List of dicts, each containing a 'resolved_url' key.
        min_word_count (int): Minimum number of words required to keep the article.
    
    Returns:
        list: Filtered list of dicts with added 'text', 'scraped_title', and 'image_url' fields.
    """
    enriched = []

    for i, item in enumerate(news_items):
        resolved_url = item.get("url")
        if not resolved_url:
            continue

        print(f"📰 ({i+1}/{len(news_items)}) Scraping: {resolved_url}")
        try:
            article = Article(resolved_url)
            article.download()
            article.parse()

            text = article.text.strip()
            if len(text.split()) < min_word_count:
                print(f"⚠️ Skipped (too short: {len(text.split())} words)")
                continue

            item["title"] = article.title
            item["content"] = text
            item["image"] = article.top_image  # may be empty string if no image found
            item['publishedAt'] = item.get('published date')
            

            enriched.append(item)
            print(f"✅ Scraped: {article.title} ({len(text.split())} words)")

        except Exception as e:
            print(f"❌ Failed to scrape {resolved_url}: {e}")
            continue

    return enriched
'''
'''
#Function that starts scrapping

def get_news_from_google_news(keywords: str, num_articles: int = 3):

    google_news = GNews()
    articles = google_news.get_news(keywords)[:num_articles]
    #articles = resolve_real_urls_only(articles, wait_time=1)
    articles = add_text_and_image_to_articles(articles, min_word_count=100)

    return articles
 ''' 

#############################################
#   Get top news frmo RSS feeds
#############################################


# Global RSS feeds
RSS_SOURCES = [
    ("Al Jazeera",         "https://www.aljazeera.com/xml/rss/all.xml"),
    ("BBC World",          "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Reuters Top News",   "http://feeds.reuters.com/reuters/topNews"),
]

def fetch_rss_items(rss_sources=RSS_SOURCES):
    """
    Fetches all entries from given RSS sources.
    Returns a list of dicts with:
      - title, text, link, source
      - published (ISO string or None)
      - published_dt (datetime or None)
    """
    items = []
    for source, url in rss_sources:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            text = getattr(entry, "summary", "") or getattr(entry, "description", "")
            combined = f"{entry.title}\n\n{text}"
            published_dt = None
            published_iso = None
            if hasattr(entry, "published"):
                try:
                    published_dt = date_parser.parse(entry.published)
                    published_iso = published_dt.isoformat()
                except Exception:
                    pass
            items.append({
                "title":        entry.title,
                "text":         combined,
                "link":         entry.link,
                "source":       source,
                "published":    published_iso,
                "published_dt": published_dt
            })
    return items

def select_top_items(items, top_n=3, use_tfidf=True):
    """
    From a list of items (in whatever order), select top_n:
      - if use_tfidf=False: just first top_n items
      - if use_tfidf=True: TF–IDF + greedy max-distance seeded from idx 0
    """
    if not use_tfidf:
        return items[:top_n]
    
    docs = [it["text"] for it in items]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vec.fit_transform(docs)
    
    selected_idxs = [0]
    for _ in range(1, min(top_n, len(docs))):
        best_idx, best_min = None, -1
        for i in range(len(docs)):
            if i in selected_idxs:
                continue
            sims = cosine_similarity(tfidf[i], tfidf[selected_idxs]).flatten()
            dists = 1 - sims
            min_dist = dists.min()
            if min_dist > best_min:
                best_min, best_idx = min_dist, i
        selected_idxs.append(best_idx)
    return [items[i] for i in selected_idxs]

def fetch_article(url):
    """
    Uses newspaper3k to download & parse the article.
    Returns (full_text, image_url_or_None).
    """
    art = Article(url)
    art.download()
    art.parse()
    return art.text, art.top_image or None

def get_top_news_full_json(
    top_n=3,
    use_tfidf=True,
    sort_by_date=False,
    rss_sources=RSS_SOURCES
):
    """
    Returns a JSON array of the top_n articles, each with:
      - title
      - link
      - source
      - published (ISO string or null)
      - full_text
      - image (URL or null)

    Parameters:
      top_n       Number of articles to return.
      use_tfidf   If True, pick a diverse set; if False, take in-order.
      sort_by_date If True, sort all items by published date desc before selecting.
    """
    # 1. Fetch and parse dates
    items = fetch_rss_items(rss_sources)
    if not items:
        return json.dumps([])

    # 2. Optional sort by date
    if sort_by_date:
        items.sort(
            key=lambda x: x["published_dt"] or datetime.min,
            reverse=True
        )

    # 3. Select top items
    selected = select_top_items(items, top_n, use_tfidf)

    # 4. Fetch full text & image, build output
    output = []
    for art in selected:
        try:
            full_text, image_url = fetch_article(art["link"])
        except Exception as e:
            full_text, image_url = f"[Error fetching article: {e}]", None
        output.append({
            "title":      art["title"],
            "link":       art["link"],
            "source":     art["source"],
            "published":  art["published"],
            "full_text":  full_text,
            "image":      image_url
        })

    return output


#############################################
#   Get trending topics 
#############################################

def extract_keywords_from_raw(raw_response):
    """
    Extracts a list of lists from a raw string and returns it as a dictionary
    with key 'keywords'. Each list represents a keyword list entry.
    """
    try:
        # Step 1: Clean the outer quotes if present
        if raw_response.startswith('"') and raw_response.endswith('"'):
            raw_response = raw_response[1:-1]

        # Step 2: Replace escaped newline characters and quotes
        cleaned = raw_response.replace('\\n\\n', '\n\n').replace('\\"', '"').strip()

        # Step 3: Split the string by double newlines
        list_strings = [line.strip() for line in cleaned.split("\n\n") if line.strip()]

        # Step 4: Convert each list string into an actual Python list
        keyword_lists = [ast.literal_eval(line) for line in list_strings]

        # Step 5: Return as dictionary
        return {"keywords": keyword_lists}

    except (SyntaxError, ValueError) as e:
        return {"error": f"Failed to parse input: {e}"}


def fetch_trending_topics(country_codes= ['IN', 'ID', 'US', 'BR', 'MX', 'NG', 'JP', 'DE', 'RU'], top_n=3):
    """
    Fetches top N trending topics for each country in the list.

    Parameters:
        country_codes (list of str): List of country codes (e.g., 'US', 'IN')
        top_n (int): Number of trends to fetch per country

    Returns:
        list of dicts: Each dict contains 'country' and 'title'
    """
    all_trends = []

    for country_code in country_codes:
        url = f'https://trends.google.com/trending/rss?geo={country_code}'
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                print(f"No entries found for {country_code}")
                continue
            for entry in feed.entries[:top_n]:
                #all_trends.append({
                #    'country': country_code,
                #    'title': entry.title
                #})
                all_trends.append(entry.title)
        except Exception as e:
            print(f"Failed to fetch trends for {country_code}: {e}")

    return all_trends

def get_trending_topics(country_codes= ['IN', 'ID', 'US', 'BR', 'MX', 'NG', 'JP', 'DE', 'RU'], template_dir="prompts"):
    trends_all = fetch_trending_topics(country_codes= country_codes)
    with open(os.path.join(template_dir, f"trending_topics.txt"), "r") as f:
        template = f.read()

    prompt = template.format(
            searches = trends_all)
    raw_response = get_llm_response(prompt)
    print(raw_response)
    raw_response = extract_keywords_from_raw(raw_response)['keywords'][0]['keywords']
    return raw_response



##############################################

'''def get_image(keyword):
    """
    Fetches the first image URL from Bing News RSS feed in English (en-US) for the given keyword.
    """
    # Use English market (en-US) for Bing RSS
    rss_url = f"https://www.bing.com/news/search?q={keyword}&format=rss&mkt=en-US&setLang=en"
    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        # Try Bing's custom News:Image tag
        image_url = entry.get("news_image")
        if image_url:
            return image_url

        # Fallback: extract image from HTML in summary or description
        text = entry.get("summary", "") or entry.get("description", "")
        match = re.search(r'<img[^>]+src="([^">]+)"', text)
        if match:
            return match.group(1)

    return None
'''


import requests
import certifi
import feedparser
import html
import re
from urllib.parse import urljoin

IMG_TAG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)

def _clean_url(u: str) -> str | None:
    if not u:
        return None
    u = html.unescape(u).strip()
    if u.startswith("data:"):
        return None
    # Protocol-relative URLs -> https
    if u.startswith("//"):
        u = "https:" + u
    return u

def get_images(keyword: str, *, market: str = "en-US", lang: str = "en",
               limit: int | None = None) -> list[str]:
    """
    Fetch image URLs from Bing News RSS for the given keyword.
    Returns a de-duplicated, ordered list of image URLs.

    Args:
        keyword: search query (e.g., "football")
        market: Bing market, e.g. "en-US"
        lang: interface language, e.g. "en"
        limit: max number of image URLs to return (None = no limit)
    """
    params = {"q": keyword, "format": "rss", "mkt": market, "setLang": lang}
    headers = {"User-Agent": "rss-image-fetcher/1.0 (+https://example.org)"}

    # Use requests+certifi to avoid SSL: CERTIFICATE_VERIFY_FAILED
    resp = requests.get("https://www.bing.com/news/search",
                        params=params, headers=headers, timeout=20,
                        verify=certifi.where())
    resp.raise_for_status()

    feed = feedparser.parse(resp.content)

    seen = set()
    images: list[str] = []

    for entry in feed.entries:
        candidates: list[str] = []

        # 1) media:content (feedparser -> entry['media_content'] = [{'url': ...}])
        for m in entry.get("media_content", []) or []:
            candidates.append(m.get("url"))

        # 2) media:thumbnail (feedparser -> entry['media_thumbnail'])
        for t in entry.get("media_thumbnail", []) or []:
            candidates.append(t.get("url"))

        # 3) <enclosure> or similar (feedparser -> entry['links'] with rel/type)
        for link in entry.get("links", []) or []:
            rel = (link.get("rel") or "").lower()
            typ = (link.get("type") or "").lower()
            if rel in {"enclosure", "preview", "image"} or typ.startswith("image/"):
                candidates.append(link.get("href") or link.get("url"))

        # 4) Bing custom keys occasionally exposed
        for k in ("news_image", "image"):
            candidates.append(entry.get(k))

        # 5) HTML fallback in summary/description/content:encoded
        html_blobs = []
        if "summary" in entry and entry["summary"]:
            html_blobs.append(entry["summary"])
        if "description" in entry and entry["description"]:
            html_blobs.append(entry["description"])
        for c in entry.get("content", []) or []:  # list of dicts with 'value'
            v = c.get("value")
            if v:
                html_blobs.append(v)
        for blob in html_blobs:
            for m in IMG_TAG_RE.finditer(blob):
                candidates.append(m.group(1))

        # Normalize, dedupe, and collect
        for u in candidates:
            u = _clean_url(u)
            if not u:
                continue
            # Some feeds use relative paths (rare) — make absolute if possible
            if u.startswith("/"):
                # entry.link is usually the article URL; fall back to Bing
                base = entry.get("link") or "https://www.bing.com/"
                u = urljoin(base, u)
            if u not in seen:
                seen.add(u)
                images.append(u)
                if limit is not None and len(images) >= limit:
                    return images

    return images

def get_image(keyword: str) -> str | None:
    """Backwards-compatible helper: return just the first image URL or None."""
    imgs = get_images(keyword, limit=1)
    return imgs[0] if imgs else None

get_image("football")



#Old Code to get the articles text 
'''
def get_urls_from_google(keywords, region="us", num_results=15, lang="en"):
    """
    Fetch URLs from Google search results using `googlesearch`.

    Args:
        keywords (str): Search query.
        region (str): Region code (used to map to TLD). Default is "us".
        num_results (int): Number of results to retrieve. Default is 15.
        lang (str): Language code. Default is "en".

    Returns:
        List[str]: A list of URLs returned by the search, or an empty list if failed.
    """
    region_to_tld = {
        "us": "com",
        "uk": "co.uk",
        "ca": "ca",
        "au": "com.au",
        "de": "de"
    }
    tld = region_to_tld.get(region, "com")

    try:
        results = list(search(
            keywords,
            #tld=tld,
            lang=lang,
            #stop=num_results,
            #pause=2
        ))
        return results
    except Exception as e:
        print(f"Error in Google search: {e}")
        return []


def get_articles(urls, max_articles=5):
    """
    Extract article content from a list of URLs.

    Args:
        urls (List[str]): List of URL strings.
        max_articles (int): Maximum number of articles to extract. Default is 5.

    Returns:
        List[str]: A list of article text contents extracted from the given URLs.
    """
    articles = []
    for url in urls:
        try:
            # URLs are strings, not objects with .url attribute
            art = extract_main_text_from_url(url)
            if art:  # Only append non-empty articles
                articles.append(art)
        except Exception as e:
            print(f'Invalid URL {url}: {e}')
        if len(articles) >= max_articles:
            break
    return articles


def extract_main_text_from_url(url: str) -> str:
    """
    Download and parse the main text content from a news article URL.

    Args:
        url (str): The URL of the news article.

    Returns:
        str: The main text of the article, or an empty string on failure.
    """
    if not url or not url.startswith("http"):
        return ""
    try:
        article = Article(url, language='en')
        article.download()
        article.parse()
        text_content = re.sub(r'\n+', '\n', article.text).strip()
        return text_content
    except Exception as e:
        print(f"Failed to extract content from {url}: {e}")
        return ""

'''

#Alternative way to get the news on the certain topic in Gnews

def get_news_from_gnews(keywords, max_articles=10, lang="en", country="us", apikey=GNEWS_API_KEY):
    """
    Fetches news articles from the GNews API for given keywords.

    Parameters:
        keywords (str): Keywords to search for (e.g. "iran israel").
        max_articles (int): Max number of articles to return (up to 100).
        lang (str): Language code (e.g. "en", "de", "fr").
        country (str): Country code (e.g. "us", "de", "gb").
        apikey (str): GNews API key.

    Returns:
        list: List of article dictionaries.
    """
    encoded_keywords = urllib.parse.quote(keywords)
    url = (
        f"https://gnews.io/api/v4/search"
        f"?q={encoded_keywords}"
        f"&lang={lang}"
        f"&country={country}"
        f"&max={max_articles}"
        f"&apikey={apikey}"
        f"&expand='content'"
        
    )

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("articles", [])
    except Exception as e:
        print(f"❌ Error fetching from GNews: {e}")
        return []




