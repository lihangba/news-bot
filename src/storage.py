"""存储模块 — SQLite 持久化，记录已推送文章，支持硬去重查询。"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger

from .models import Article

# 北京时间时区
TZ_BEIJING = timezone(timedelta(hours=8))

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,
    url_hash    TEXT    NOT NULL,
    title_hash  TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    url         TEXT    NOT NULL,
    summary     TEXT    NOT NULL DEFAULT '',
    published_at TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_url_hash  ON articles(url_hash);
CREATE INDEX IF NOT EXISTS idx_title_hash ON articles(title_hash);
CREATE INDEX IF NOT EXISTS idx_created_at ON articles(created_at);
"""


class Storage:
    """SQLite 存储管理。

    使用方式::

        storage = Storage("news.db")
        storage.initialize()
        exists = storage.exists_by_hash(url_hash, title_hash)
        storage.save_articles(articles)
    """

    def __init__(self, db_path: str = "news.db"):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def initialize(self):
        """创建表结构和索引（幂等）。"""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        logger.debug(f"数据库已就绪: {self.db_path}")

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # 去重查询
    # ------------------------------------------------------------------

    def exists_by_hash(self, url_hash: str, title_hash: str) -> bool:
        """检查 URL 哈希或标题哈希是否已存在。"""
        row = self.conn.execute(
            "SELECT 1 FROM articles WHERE url_hash = ? OR title_hash = ? LIMIT 1",
            (url_hash, title_hash),
        ).fetchone()
        return row is not None

    def get_existing_hashes(self) -> set[tuple[str, str]]:
        """返回所有已知的 (url_hash, title_hash) 集合（用于批量硬去重）。"""
        rows = self.conn.execute(
            "SELECT url_hash, title_hash FROM articles"
        ).fetchall()
        # 返回包含两个 hash 的集合，任一命中即算重复
        result: set[tuple[str, str]] = set()
        for row in rows:
            result.add((row["url_hash"], row["title_hash"]))
        return result

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def save_article(self, article: Article):
        """保存单篇文章记录。"""
        self.conn.execute(
            """
            INSERT INTO articles (source, url_hash, title_hash, title, url, summary, published_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article.source,
                article.url_hash,
                article.title_hash,
                article.title,
                article.url,
                article.summary,
                article.published_at.isoformat() if article.published_at else None,
            ),
        )
        self.conn.commit()

    def save_articles(self, articles: list[Article]):
        """批量保存文章。"""
        data = [
            (
                a.source,
                a.url_hash,
                a.title_hash,
                a.title,
                a.url,
                a.summary,
                a.published_at.isoformat() if a.published_at else None,
            )
            for a in articles
        ]
        self.conn.executemany(
            """
            INSERT INTO articles (source, url_hash, title_hash, title, url, summary, published_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        self.conn.commit()
        logger.info(f"已保存 {len(articles)} 篇文章到数据库")

    # ------------------------------------------------------------------
    # 历史查询（供软去重用）
    # ------------------------------------------------------------------

    def get_recent_articles(self, days: int = 3, limit: int = 100) -> list[dict]:
        """获取近 N 天已推送的文章（供 DeepSeek 软去重做历史参考）。

        Args:
            days: 回溯天数，默认 3 天
            limit: 最多返回条数，防止上下文过长

        Returns:
            [{"title": "...", "summary": "...", "source": "...", "date": "..."}, ...]
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            """SELECT title, summary, source, published_at, created_at
               FROM articles
               WHERE created_at >= ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        return [
            {
                "title": r["title"],
                "summary": r["summary"][:200],
                "source": r["source"],
                "date": (r["published_at"] or r["created_at"] or "")[:10],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # JSON 持久化（跨 GitHub Actions 运行保留状态）
    # ------------------------------------------------------------------

    def export_to_json(self, path: str = "history.json"):
        """将全部文章记录导出为 JSON 文件，用于跨 Actions 运行的持久化。

        Args:
            path: 导出文件路径
        """
        rows = self.conn.execute(
            "SELECT source, url_hash, title_hash, title, url, summary, "
            "published_at, created_at FROM articles ORDER BY created_at"
        ).fetchall()
        data = [dict(r) for r in rows]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"已导出 {len(data)} 条记录到 {path}")

    def import_from_json(self, path: str = "history.json") -> int:
        """从 JSON 文件导入文章记录（幂等：已存在的哈希自动跳过）。

        用于 GitHub Actions 启动时恢复上一次运行的去重状态。

        Args:
            path: 导入文件路径

        Returns:
            实际导入的新记录数
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.info(f"{path} 不存在，跳过导入（首次运行？）")
            return 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"{path} 读取失败: {e}，跳过导入")
            return 0

        if not data:
            logger.info(f"{path} 为空，跳过导入")
            return 0

        imported = 0
        for record in data:
            url_hash = record.get("url_hash", "")
            title_hash = record.get("title_hash", "")
            if not url_hash or not title_hash:
                continue

            # 幂等检查
            exists = self.conn.execute(
                "SELECT 1 FROM articles WHERE url_hash = ? OR title_hash = ? LIMIT 1",
                (url_hash, title_hash),
            ).fetchone()
            if exists:
                continue

            self.conn.execute(
                """INSERT INTO articles
                   (source, url_hash, title_hash, title, url, summary, published_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.get("source", ""),
                    url_hash,
                    title_hash,
                    record.get("title", ""),
                    record.get("url", ""),
                    record.get("summary", ""),
                    record.get("published_at"),
                    record.get("created_at"),
                ),
            )
            imported += 1

        self.conn.commit()
        logger.info(
            f"从 {path} 导入了 {imported} 条新记录 "
            f"(文件共 {len(data)} 条, 跳过 {len(data) - imported} 条已存在)"
        )
        return imported

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def purge_old(self, retention_days: int = 7):
        """删除超过保留期限的历史记录。"""
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor = self.conn.execute(
            "DELETE FROM articles WHERE created_at < ?", (cutoff,)
        )
        deleted = cursor.rowcount
        if deleted:
            self.conn.commit()
            logger.info(f"清理了 {deleted} 条过期记录 (retention={retention_days}d)")
