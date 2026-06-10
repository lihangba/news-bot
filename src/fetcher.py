"""RSS 抓取模块 — 并发抓取多个 RSS 源，带重试与超时保护。"""

from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
import feedparser
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import Article

# 北京时间
TZ_BEIJING = timezone(timedelta(hours=8))

# ---- 配置常量 ----
DEFAULT_TIMEOUT = 15
DEFAULT_RETRIES = 3
DEFAULT_USER_AGENT = "NewsBot/1.0 (Daily Morning Report)"


def _parse_published(entry: dict[str, Any]) -> datetime | None:
    """尝试从 feedparser entry 中提取发布时间，并统一转为北京时间。

    优先级: published_parsed > published > updated_parsed
    """
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed and len(parsed) >= 6:
            try:
                dt = datetime(*parsed[:6], tzinfo=timezone.utc)
                return dt.astimezone(TZ_BEIJING)
            except (ValueError, OSError):
                continue

    # 兜底：尝试字符串字段
    raw = entry.get("published") or entry.get("updated") or ""
    if raw:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(TZ_BEIJING)
        except Exception:
            pass

    return None


def _extract_summary(entry: dict[str, Any]) -> str:
    """从 entry 中提取最合适的摘要文本。

    优先级: summary > content[0].value > description > title
    """
    summary = entry.get("summary", "")
    if not summary:
        content = entry.get("content", [])
        if content:
            summary = content[0].get("value", "")
    if not summary:
        summary = entry.get("description", "")
    if not summary:
        summary = entry.get("title", "")
    return summary


def _build_article(source_config: dict[str, Any], entry: dict[str, Any]) -> Article:
    """将 RSS 条目 + 源配置组装为 Article 对象。"""
    title = (entry.get("title") or "").strip()
    link = (entry.get("link") or entry.get("links", [{}])[0].get("href") or "").strip()
    summary = _extract_summary(entry)
    published_at = _parse_published(entry)

    return Article(
        source=source_config["name"],
        category=source_config["category"],
        priority=source_config.get("priority", 5),
        title=title,
        url=link,
        summary=summary,
        published_at=published_at,
    )


class Fetcher:
    """RSS 抓取器。"""

    def __init__(
        self,
        sources: list[dict[str, Any]],
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
        user_agent: str = DEFAULT_USER_AGENT,
        rsshub_base: str = "https://rsshub.app",
    ):
        self.sources = [s for s in sources if s.get("enabled", True)]
        self.timeout = timeout
        self.user_agent = user_agent
        self.rsshub_base = rsshub_base.rstrip("/")
        self._retry_decorator = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
            ),
            reraise=True,
        )

    def _fetch_one(self, source: dict[str, Any]) -> list[Article]:
        """抓取单个 RSS 源，返回解析后的 Article 列表。"""
        name = source["name"]
        url = source["url"]

        # 替换 {{RSSHUB_BASE}} 占位符（注意不是 ${} ，避免和环境变量解析冲突）
        url = url.replace("{{RSSHUB_BASE}}", self.rsshub_base)

        logger.info(f"开始抓取: {name} ({url})")

        @self._retry_decorator
        def _do_request() -> httpx.Response:
            return httpx.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/rss+xml, application/xml, text/xml, application/atom+xml, */*;q=0.9",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
                follow_redirects=True,
            )

        try:
            response = _do_request()
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"[{name}] HTTP {e.response.status_code}: {url}")
            return []
        except Exception as e:
            logger.error(f"[{name}] 请求失败: {e}")
            return []

        # 解析 RSS/Atom
        feed = feedparser.parse(response.text)
        if feed.bozo and not feed.entries:
            # feedparser 解析失败且没有条目 — 可能是非标准 XML
            logger.warning(
                f"[{name}] RSS 解析异常: {feed.bozo_exception} — 尝试宽松模式"
            )
            # 仍然尝试提取条目（feedparser 有时即使报错也能解析）
            if not feed.entries:
                logger.error(f"[{name}] 无有效条目，跳过")
                return []

        articles = []
        for entry in feed.entries:
            try:
                article = _build_article(source, entry)
                if article.title and article.url:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"[{name}] 解析条目失败: {e}")
                continue

        logger.info(f"[{name}] 抓取完成: {len(articles)} 条")
        return articles

    def fetch_all(self) -> list[Article]:
        """顺序抓取所有源（避免并发对 RSSHub 的压力），单源失败不影响整体。

        Returns:
            所有源抓取到的 Article 列表
        """
        all_articles: list[Article] = []
        success_count = 0
        fail_count = 0

        for source in self.sources:
            try:
                articles = self._fetch_one(source)
                if articles:
                    all_articles.extend(articles)
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"[{source['name']}] 抓取异常: {e}")
                fail_count += 1

        logger.info(
            f"抓取阶段完成: 成功 {success_count}/{len(self.sources)} 个源, "
            f"总计 {len(all_articles)} 条原始文章, {fail_count} 个源失败"
        )
        return all_articles
