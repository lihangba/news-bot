#!/usr/bin/env python3
"""财经+AI 新闻早报机器人 — 主入口。

编排完整流水线：抓取 → 去重 → AI处理 → 推送 → 存储。

用法::

    python main.py                        # 默认读取 config.yaml
    python main.py --config custom.yaml   # 自定义配置路径
"""

import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import yaml
from loguru import logger

from src.logger import setup_logging
from src.storage import Storage
from src.fetcher import Fetcher
from src.dedup import HardDedup, SoftDedup
from src.ai import DeepSeekClient
from src.push import FeishuPusher


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def _resolve_env(value: str) -> str:
    """将字符串中的 ${VAR_NAME} 替换为环境变量值。"""
    if not isinstance(value, str):
        return value

    def _replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(r"\$\{(\w+)\}", _replacer, value)


def _resolve_config(obj):
    """递归遍历配置对象，替换所有 ${VAR} 占位符。"""
    if isinstance(obj, dict):
        return {k: _resolve_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_config(v) for v in obj]
    elif isinstance(obj, str):
        return _resolve_env(obj)
    return obj


def load_config(path: str = "config.yaml") -> dict:
    """加载并解析 YAML 配置文件。"""
    config_path = Path(path)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = _resolve_config(config)
    logger.info(f"配置已加载: {config_path}")
    return config


# ---------------------------------------------------------------------------
# 文章精选
# ---------------------------------------------------------------------------

def select_top_articles(articles: list, max_count: int = 8) -> list:
    """从去重后的文章中精选 Top-N。

    策略：按优先级降序 → 同优先级按发布时间倒序 → 取前 max_count 条。
    同时保证财经和 AI 各至少 1 条（如果存在的话）。
    """
    if len(articles) <= max_count:
        return articles

    # 按 (priority desc, published_at desc) 排序
    def _sort_key(a):
        # priority 越大越靠前，published_at 越新越靠前
        ts = a.published_at.timestamp() if a.published_at else 0
        return (-a.priority, -ts)

    sorted_articles = sorted(articles, key=_sort_key)

    # 确保覆盖两个类别
    finance_top = [a for a in sorted_articles if a.category == "finance"]
    ai_top = [a for a in sorted_articles if a.category == "ai"]

    selected = []
    # 先各取一篇最优的
    if finance_top:
        selected.append(finance_top[0])
    if ai_top:
        selected.append(ai_top[0])

    # 剩余名额按排名填充，避免重复
    existing_urls = {a.url_hash for a in selected}
    for a in sorted_articles:
        if len(selected) >= max_count:
            break
        if a.url_hash not in existing_urls:
            selected.append(a)
            existing_urls.add(a.url_hash)

    logger.info(
        f"精选: {len(articles)} → {len(selected)} "
        f"(财经 {sum(1 for a in selected if a.category == 'finance')}, "
        f"AI {sum(1 for a in selected if a.category == 'ai')})"
    )
    return selected


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(config: dict) -> bool:
    """执行一次完整的早报推送流程。

    Returns:
        True 表示流程正常完成（即使部分非关键步骤失败）
    """
    # ---- 1. 初始化日志 ----
    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/news-bot.log"),
        rotation=log_cfg.get("rotation", "7 days"),
        retention=log_cfg.get("retention", "14 days"),
    )

    logger.info("=" * 60)
    logger.info(f"财经+AI 新闻早报机器人启动 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ---- 2. 初始化存储 ----
    storage_cfg = config.get("storage", {})
    db_path = storage_cfg.get("db_path", "news.db")
    history_json = storage_cfg.get("history_json", "history.json")
    storage = Storage(db_path=db_path)
    try:
        storage.initialize()
        # 从 history.json 恢复上次运行的持久化状态
        imported = storage.import_from_json(history_json)
        logger.info(f"从 {history_json} 恢复 {imported} 条历史记录")
        storage.purge_old(retention_days=storage_cfg.get("retention_days", 7))
    except Exception as e:
        logger.error(f"存储初始化失败: {e}")
        return False

    # ---- 3. 抓取 RSS ----
    source_list = config.get("sources", [])
    if not source_list:
        logger.error("未配置任何 RSS 源，请检查 config.yaml")
        return False

    fetcher_cfg = config.get("fetcher", {})
    rsshub_cfg = config.get("rsshub", {})
    fetcher = Fetcher(
        sources=source_list,
        timeout=fetcher_cfg.get("timeout_seconds", 15),
        max_retries=fetcher_cfg.get("max_retries", 3),
        user_agent=fetcher_cfg.get("user_agent", "NewsBot/1.0"),
        rsshub_base=rsshub_cfg.get("base_url", "https://rsshub.app"),
    )

    try:
        raw_articles = fetcher.fetch_all()
    except Exception as e:
        logger.error(f"RSS 抓取阶段异常: {e}")
        raw_articles = []

    if not raw_articles:
        logger.warning("今日未抓取到任何文章，流程结束")
        return True  # 不算失败，只是没新闻

    # ---- 4. 硬去重 ----
    hard_dedup = HardDedup(storage)
    try:
        after_hard = hard_dedup.deduplicate(raw_articles)
    except Exception as e:
        logger.error(f"硬去重异常: {e}")
        after_hard = raw_articles

    if not after_hard:
        logger.info("硬去重后无新文章，流程结束")
        return True

    # ---- 5. 初始化 AI 客户端 ----
    ds_cfg = config.get("deepseek", {})
    try:
        ai_client = DeepSeekClient(
            api_key=ds_cfg.get("api_key"),
            base_url=ds_cfg.get("base_url", "https://api.deepseek.com"),
            model=ds_cfg.get("model", "deepseek-chat"),
            temperature=ds_cfg.get("temperature", 0.3),
            max_tokens=ds_cfg.get("max_tokens", 2048),
        )
    except ValueError as e:
        logger.error(f"DeepSeek 客户端初始化失败: {e}")
        ai_client = None

    # ---- 6. 软去重 ----
    dedup_cfg = config.get("dedup", {})
    soft_enabled = dedup_cfg.get("soft_dedup_enabled", True)

    if ai_client and soft_enabled:
        try:
            soft_dedup = SoftDedup(
                ai_client,
                storage=storage,
                enabled=True,
                max_candidates=dedup_cfg.get("max_candidates", 20),
                history_days=dedup_cfg.get("history_days", 3),
            )
            after_soft = soft_dedup.deduplicate(after_hard)
        except Exception as e:
            logger.error(f"软去重异常，回退到仅硬去重结果: {e}")
            after_soft = after_hard
    else:
        if not ai_client:
            logger.warning("AI 客户端不可用，跳过软去重")
        after_soft = after_hard

    if not after_soft:
        logger.info("软去重后无文章，流程结束")
        return True

    # ---- 7. 精选 Top-N ----
    push_cfg = config.get("push", {})
    max_articles = push_cfg.get("max_articles", 8)
    selected = select_top_articles(after_soft, max_count=max_articles)

    # ---- 8. AI 摘要生成（只为最终推送的文章生成，节省 token） ----
    if ai_client:
        try:
            summaries = ai_client.generate_summaries(selected)
            for article, summary in zip(selected, summaries):
                article.chinese_summary = summary
        except Exception as e:
            logger.error(f"摘要生成失败，使用原始摘要: {e}")
    else:
        logger.warning("AI 客户端不可用，使用原始摘要")

    # ---- 9. 推送到飞书 ----
    feishu_cfg = config.get("feishu", {})
    webhook_url = feishu_cfg.get("webhook_url", "")
    if not webhook_url:
        logger.error("飞书 Webhook URL 未配置，无法推送")
        # 即使不能推送，也要保存文章防止下次重复
        try:
            storage.save_articles(after_soft)
        except Exception:
            pass
        return False

    pusher = FeishuPusher(webhook_url=webhook_url)
    try:
        pushed = pusher.send_daily_report(
            articles=selected,
            max_per_card=push_cfg.get("max_per_card", 4),
            max_cards=push_cfg.get("max_cards", 2),
            header_color=feishu_cfg.get("header_color", "blue"),
            card_title=feishu_cfg.get("card_title", "📰 财经+AI 早报"),
        )
    except Exception as e:
        logger.error(f"推送异常: {e}")
        pushed = 0

    # ---- 10. 保存已处理文章到数据库 ----
    try:
        storage.save_articles(after_soft)
    except Exception as e:
        logger.error(f"保存文章到数据库失败: {e}")

    # ---- 10b. 导出到 history.json（跨 Actions 运行持久化） ----
    try:
        storage.export_to_json(history_json)
    except Exception as e:
        logger.error(f"导出 {history_json} 失败: {e}")

    # ---- 11. 清理 ----
    try:
        storage.close()
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info(
        f"流程结束: 原始 {len(raw_articles)} → "
        f"硬去重 {len(after_hard)} → "
        f"软去重 {len(after_soft)} → "
        f"推送 {len(selected)} 条 ({pushed} 张卡片成功)"
    )
    logger.info("=" * 60)

    return pushed > 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="财经+AI 新闻早报机器人",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    args = parser.parse_args()

    # 切换到脚本所在目录（确保相对路径正确）
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    success = run(load_config(args.config))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
