"""去重模块 — 硬去重（哈希查库）+ 软去重（DeepSeek 语义判断）。"""

from loguru import logger

from .models import Article
from .storage import Storage
from .ai import DeepSeekClient


class HardDedup:
    """硬去重：基于 URL 哈希和标题哈希查数据库。

    任一哈希命中即判定为重复。
    """

    def __init__(self, storage: Storage):
        self.storage = storage
        self._known_hashes: set[tuple[str, str]] | None = None

    def _load_hashes(self):
        """延迟加载已知哈希集合。"""
        if self._known_hashes is None:
            self._known_hashes = self.storage.get_existing_hashes()
            logger.debug(f"已加载 {len(self._known_hashes)} 条历史哈希")

    def is_duplicate(self, article: Article) -> bool:
        """检查单篇文章是否为硬重复。"""
        self._load_hashes()
        assert self._known_hashes is not None

        # 检查 url_hash 或 title_hash 是否命中
        for url_h, title_h in self._known_hashes:
            if article.url_hash == url_h or article.title_hash == title_h:
                return True
        return False

    def deduplicate(self, articles: list[Article]) -> list[Article]:
        """批量硬去重，返回非重复的文章列表。"""
        self._load_hashes()
        assert self._known_hashes is not None

        # 先对输入内部去重（同一批次中 URL/标题相同的合并）
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[Article] = []

        for a in articles:
            if a.url_hash in seen_urls or a.title_hash in seen_titles:
                continue
            seen_urls.add(a.url_hash)
            seen_titles.add(a.title_hash)
            unique.append(a)

        batch_deduped = len(articles) - len(unique)
        if batch_deduped:
            logger.info(f"批次内去重移除 {batch_deduped} 条")

        # 与历史数据库比对
        result: list[Article] = []
        for a in unique:
            is_dup = False
            for url_h, title_h in self._known_hashes:
                if a.url_hash == url_h or a.title_hash == title_h:
                    is_dup = True
                    break
            if not is_dup:
                result.append(a)

        removed = len(unique) - len(result)
        logger.info(
            f"硬去重: {len(articles)} → {len(result)} "
            f"(批次内去重 {batch_deduped}, 历史去重 {removed})"
        )
        return result


class SoftDedup:
    """软去重：一次 API 调用，同时比对「今日候选」与「近3天历史」。

    流程：
    1. 按优先级排序，取前 20 条候选
    2. 从 Storage 取近 3 天已推送文章作为历史参考
    3. 调用 DeepSeek，返回每条候选的 is_duplicate + reason
    4. 过滤掉 is_duplicate=True 的文章
    """

    def __init__(
        self,
        ai_client: DeepSeekClient,
        storage: Storage,
        enabled: bool = True,
        max_candidates: int = 20,
        history_days: int = 3,
    ):
        self.ai = ai_client
        self.storage = storage
        self.enabled = enabled
        self.max_candidates = max_candidates
        self.history_days = history_days

    def deduplicate(self, articles: list[Article]) -> list[Article]:
        """执行语义去重（批量模式）。

        Args:
            articles: 通过硬去重后的文章列表

        Returns:
            软去重后的文章列表
        """
        if not self.enabled:
            logger.info("软去重已关闭，跳过")
            return articles

        if not articles:
            return []

        # 1. 按优先级降序排列，取前 max_candidates 篇
        sorted_articles = sorted(articles, key=lambda a: a.priority, reverse=True)
        candidates = sorted_articles[: self.max_candidates]
        overflow = sorted_articles[self.max_candidates :]

        if overflow:
            logger.info(
                f"候选文章超过上限 ({len(articles)} → 取前 {len(candidates)} 篇进入软去重, "
                f"{len(overflow)} 篇低优先级直接保留)"
            )

        if len(candidates) <= 1:
            return articles  # 只有 0 或 1 篇，无需语义去重

        # 2. 取近 N 天历史
        try:
            history = self.storage.get_recent_articles(days=self.history_days)
            logger.debug(f"获取历史记录: {len(history)} 条 (近 {self.history_days} 天)")
        except Exception as e:
            logger.warning(f"获取历史记录失败，仅做候选间比对: {e}")
            history = []

        # 3. 一次 API 调用完成全部去重判断
        try:
            decisions = self.ai.detect_duplicates(candidates, history)
        except Exception as e:
            logger.error(f"软去重 API 调用失败: {e}")
            return articles  # 降级：保留全部

        # 4. 根据判定结果过滤
        dup_ids = {d["id"] for d in decisions if d["is_duplicate"]}
        kept = []
        for d in decisions:
            if d["is_duplicate"]:
                logger.info(
                    f"  软去重移除 #{d['id']}: {candidates[d['id']].title[:50]} — {d.get('reason', '')}"
                )
            else:
                kept.append(candidates[d["id"]])

        # 将低优先级溢出的文章补回
        kept.extend(overflow)

        # 按优先级降序排列
        kept.sort(key=lambda a: a.priority, reverse=True)
        logger.info(
            f"软去重: {len(articles)} → {len(kept)} "
            f"(移除 {len(dup_ids)} 条重复, 历史参考 {len(history)} 条)"
        )
        return kept
