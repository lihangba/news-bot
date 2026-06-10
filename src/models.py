"""数据模型 — Article 数据类及哈希计算。"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime


def _clean_html(raw: str) -> str:
    """去除 HTML 标签，返回纯文本。"""
    if not raw:
        return ""
    # 简单正则去除标签，避免引入 BeautifulSoup 依赖在此模块
    clean = re.sub(r"<[^>]+>", " ", raw)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _compute_hash(text: str) -> str:
    """计算 SHA256 前 16 位作为短哈希。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Article:
    """一条新闻文章。

    Attributes:
        source: 来源名称 (如 "华尔街见闻·快讯")
        category: 分类 ("finance" | "ai")
        priority: 来源优先级 (越大越优先)
        title: 原始标题
        url: 原文链接
        summary: 清洗后的纯文本摘要
        published_at: 发布时间
        url_hash: URL 的 SHA256 前 16 位
        title_hash: 标题的 SHA256 前 16 位
        chinese_summary: 由 AI 生成的中文摘要 (推送阶段填充)
    """
    source: str
    category: str
    priority: int
    title: str
    url: str
    summary: str
    published_at: datetime | None = None
    url_hash: str = ""
    title_hash: str = ""
    chinese_summary: str = ""

    def __post_init__(self):
        if not self.url_hash:
            self.url_hash = _compute_hash(self.url)
        if not self.title_hash:
            self.title_hash = _compute_hash(self.title)
        # 清洗摘要中的 HTML
        if self.summary:
            self.summary = _clean_html(self.summary)

    @property
    def display_title(self) -> str:
        """推送用的标题，截断过长内容。"""
        if len(self.title) > 80:
            return self.title[:77] + "..."
        return self.title

    @property
    def display_summary(self) -> str:
        """推送用的摘要 (中文)，不超过 50 字。"""
        text = self.chinese_summary or self.summary
        if len(text) > 50:
            return text[:47] + "..."
        return text

    @property
    def is_english(self) -> bool:
        """判断标题是否为纯英文。"""
        # 如果标题中中文字符少于 3 个，视为英文
        chinese_chars = len(re.findall(r"[一-鿿]", self.title))
        return chinese_chars < 3
