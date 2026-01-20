"""
Aria Ambient Intelligence - News Watcher

Monitors RSS feeds and news sources for relevant articles.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import xml.etree.ElementTree as ET

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from .base import Watcher, WatcherConfig
from ..models import Signal
from ..constants import SignalType, CHECK_INTERVALS

logger = logging.getLogger(__name__)


class NewsWatcher(Watcher):
    """
    Monitors RSS feeds for relevant news articles.

    Features:
    - Multiple RSS feed support
    - Article deduplication
    - Content extraction
    - Keyword filtering

    Usage:
        watcher = NewsWatcher()
        watcher.add_feed("https://example.com/rss")
        signals = await watcher.observe()
    """

    name = "news"
    description = "Monitors RSS feeds for relevant news"
    default_signal_type = SignalType.NEWS_ARTICLE

    def __init__(self, config: WatcherConfig = None, feeds: List[str] = None):
        config = config or WatcherConfig(
            check_interval=CHECK_INTERVALS.get("news", 300),
            custom_settings={
                "feeds": feeds or [],
                "max_age_hours": 24,  # Ignore articles older than this
            }
        )
        super().__init__(config)

        self._feeds: List[str] = feeds or []
        self._seen_ids: Set[str] = set()  # Track seen article IDs
        self._max_seen = 1000  # Prevent unlimited growth

    def add_feed(self, url: str) -> None:
        """Add an RSS feed URL."""
        if url not in self._feeds:
            self._feeds.append(url)
            logger.info(f"Added RSS feed: {url}")

    def remove_feed(self, url: str) -> bool:
        """Remove an RSS feed URL."""
        if url in self._feeds:
            self._feeds.remove(url)
            return True
        return False

    def list_feeds(self) -> List[str]:
        """List all configured feeds."""
        return list(self._feeds)

    async def observe(self) -> List[Signal]:
        """
        Fetch and process all RSS feeds.

        Returns:
            List of signals for new articles
        """
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed, news watcher disabled")
            return []

        signals = []
        feeds = self._feeds + self.get_setting("feeds", [])

        if not feeds:
            return signals

        async with aiohttp.ClientSession() as session:
            # Fetch all feeds concurrently
            tasks = [self._fetch_feed(session, feed) for feed in feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Feed fetch error: {result}")
                    continue

                for article in result:
                    signal = self._article_to_signal(article)
                    if signal:
                        signals.append(signal)

        # Limit seen IDs cache
        if len(self._seen_ids) > self._max_seen:
            # Keep only recent half
            self._seen_ids = set(list(self._seen_ids)[-self._max_seen // 2:])

        return signals

    async def _fetch_feed(
        self,
        session: "aiohttp.ClientSession",
        feed_url: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch and parse a single RSS feed.

        Returns:
            List of article dictionaries
        """
        articles = []

        try:
            async with session.get(feed_url, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"Feed returned {response.status}: {feed_url}")
                    return articles

                content = await response.text()
                articles = self._parse_rss(content, feed_url)

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching feed: {feed_url}")
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")

        return articles

    def _parse_rss(self, content: str, feed_url: str) -> List[Dict[str, Any]]:
        """
        Parse RSS/Atom feed content.

        Returns:
            List of article dictionaries
        """
        articles = []

        try:
            root = ET.fromstring(content)

            # Handle RSS 2.0
            for item in root.findall(".//item"):
                article = self._parse_rss_item(item, feed_url)
                if article:
                    articles.append(article)

            # Handle Atom
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                article = self._parse_atom_entry(entry, feed_url)
                if article:
                    articles.append(article)

        except ET.ParseError as e:
            logger.error(f"XML parse error for {feed_url}: {e}")
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")

        return articles

    def _parse_rss_item(
        self,
        item: ET.Element,
        feed_url: str
    ) -> Optional[Dict[str, Any]]:
        """Parse an RSS 2.0 item element."""
        try:
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            description = item.findtext("description", "").strip()
            pub_date = item.findtext("pubDate", "")
            guid = item.findtext("guid", "")

            if not title:
                return None

            # Generate unique ID
            article_id = guid or hashlib.md5(f"{title}{link}".encode()).hexdigest()

            return {
                "id": article_id,
                "title": title,
                "url": link,
                "content": self._clean_html(description),
                "published": pub_date,
                "source": feed_url,
            }

        except Exception as e:
            logger.debug(f"Error parsing RSS item: {e}")
            return None

    def _parse_atom_entry(
        self,
        entry: ET.Element,
        feed_url: str
    ) -> Optional[Dict[str, Any]]:
        """Parse an Atom entry element."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        try:
            title = entry.findtext("atom:title", "", ns).strip()

            # Get link href
            link_elem = entry.find("atom:link[@rel='alternate']", ns)
            if link_elem is None:
                link_elem = entry.find("atom:link", ns)
            link = link_elem.get("href", "") if link_elem is not None else ""

            # Get content or summary
            content = entry.findtext("atom:content", "", ns).strip()
            if not content:
                content = entry.findtext("atom:summary", "", ns).strip()

            published = entry.findtext("atom:published", "", ns)
            if not published:
                published = entry.findtext("atom:updated", "", ns)

            entry_id = entry.findtext("atom:id", "", ns)

            if not title:
                return None

            # Generate unique ID
            article_id = entry_id or hashlib.md5(f"{title}{link}".encode()).hexdigest()

            return {
                "id": article_id,
                "title": title,
                "url": link,
                "content": self._clean_html(content),
                "published": published,
                "source": feed_url,
            }

        except Exception as e:
            logger.debug(f"Error parsing Atom entry: {e}")
            return None

    def _article_to_signal(self, article: Dict[str, Any]) -> Optional[Signal]:
        """
        Convert an article to a Signal.

        Returns:
            Signal object, or None if article should be skipped
        """
        article_id = article.get("id", "")

        # Skip if already seen
        if article_id in self._seen_ids:
            return None

        # Check age
        max_age_hours = self.get_setting("max_age_hours", 24)
        if article.get("published"):
            try:
                # Try common date formats
                pub_date = None
                for fmt in [
                    "%a, %d %b %Y %H:%M:%S %z",
                    "%a, %d %b %Y %H:%M:%S %Z",
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%SZ",
                ]:
                    try:
                        pub_date = datetime.strptime(article["published"], fmt)
                        break
                    except ValueError:
                        continue

                if pub_date:
                    # Make naive for comparison
                    if pub_date.tzinfo:
                        pub_date = pub_date.replace(tzinfo=None)
                    age = datetime.now() - pub_date
                    if age > timedelta(hours=max_age_hours):
                        return None

            except Exception:
                pass  # If date parsing fails, include the article

        # Mark as seen
        self._seen_ids.add(article_id)

        return self.create_signal(
            title=article.get("title", ""),
            content=article.get("content", "")[:2000],  # Truncate long content
            url=article.get("url", ""),
            raw_data={
                "article_id": article_id,
                "published": article.get("published", ""),
                "feed_source": article.get("source", ""),
            },
            expires_in_seconds=max_age_hours * 3600,
        )

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        import re

        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', ' ', text)
        # Decode common entities
        clean = clean.replace("&amp;", "&")
        clean = clean.replace("&lt;", "<")
        clean = clean.replace("&gt;", ">")
        clean = clean.replace("&quot;", '"')
        clean = clean.replace("&#39;", "'")
        clean = clean.replace("&nbsp;", " ")
        # Collapse whitespace
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()

    def _validate_config(self) -> List[str]:
        """Validate news watcher configuration."""
        errors = []

        max_age = self.get_setting("max_age_hours", 24)
        if not isinstance(max_age, (int, float)) or max_age < 1:
            errors.append("max_age_hours must be a positive number")

        feeds = self.get_setting("feeds", [])
        if not isinstance(feeds, list):
            errors.append("feeds must be a list")

        return errors

    def clear_seen(self) -> None:
        """Clear the seen articles cache."""
        self._seen_ids.clear()
