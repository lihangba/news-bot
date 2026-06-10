"""推送模块 — 构建飞书消息卡片并通过 Webhook 发送。"""

import json
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from .models import Article

# 飞书卡片 header 颜色映射
CATEGORY_COLORS = {
    "finance": "blue",
    "ai": "purple",
}

# 来源 emoji 映射
CATEGORY_EMOJI = {
    "finance": "💰",
    "ai": "🤖",
}


def _format_time(dt: datetime | None) -> str:
    """格式化时间为简短显示。"""
    if dt is None:
        return "未知时间"
    return dt.strftime("%m-%d %H:%M")


def _build_article_element(article: Article) -> dict[str, Any]:
    """为单篇文章构建飞书卡片的 element（div + action + divider）。

    Returns:
        一个包含 div(正文)、action(按钮)、hr(分割线) 的列表
    """
    category_emoji = CATEGORY_EMOJI.get(article.category, "📌")
    time_str = _format_time(article.published_at)

    # Markdown 正文
    md_lines = [
        f"**{article.display_title}**",
        f"{article.display_summary}",
        f"{category_emoji} {article.source} · {time_str}",
    ]
    md_content = "\n".join(md_lines)

    return [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": md_content,
            },
        },
        {
            "tag": "action",
            "layout": "flow",
            "actions": [
                {
                    "tag": "button",
                    "text": {
                        "tag": "plain_text",
                        "content": "🔗 阅读原文",
                    },
                    "url": article.url,
                    "type": "default",
                }
            ],
        },
        {"tag": "hr"},
    ]


def _build_card(
    articles: list[Article],
    card_index: int,
    total_cards: int,
    header_color: str = "blue",
    card_title: str = "📰 财经+AI 早报",
) -> dict[str, Any]:
    """构建一张飞书交互卡片。

    Args:
        articles: 本卡片要展示的文章（3-4 条）
        card_index: 卡片序号（从 1 开始）
        total_cards: 总卡片数
        header_color: 卡片头部颜色
        card_title: 卡片标题前缀

    Returns:
        飞书 interactive 消息 JSON
    """
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 标题
    if total_cards > 1:
        title = f"{card_title} ({card_index}/{total_cards}) | {today_str}"
    else:
        title = f"{card_title} | {today_str}"

    # 构建 elements
    elements: list[dict[str, Any]] = []
    for article in articles:
        elements.extend(_build_article_element(article))

    # 底部说明
    elements.append(
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": (
                        f"🤖 AI 生成摘要 · 仅供参考 · "
                        f"共 {len(articles)} 条新闻 · {today_str}"
                    ),
                }
            ],
        }
    )

    card = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title,
                },
                "template": header_color,
            },
            "elements": elements,
        },
    }

    return card


class FeishuPusher:
    """飞书 Webhook 推送器。"""

    def __init__(self, webhook_url: str):
        if not webhook_url:
            raise ValueError("飞书 Webhook URL 未设置")
        self.webhook_url = webhook_url

    def send_card(self, card: dict[str, Any]) -> bool:
        """发送一张卡片到飞书。

        Args:
            card: 飞书 interactive 消息 JSON

        Returns:
            True 表示发送成功
        """
        try:
            response = httpx.post(
                self.webhook_url,
                json=card,
                timeout=15,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            # 飞书 Webhook 返回 { "StatusCode": 0, "StatusMessage": "success" }
            body = response.json()
            status_code = body.get("StatusCode", body.get("code", -1))
            if status_code != 0:
                logger.error(
                    f"飞书推送返回错误: StatusCode={status_code}, "
                    f"msg={body.get('StatusMessage', body.get('msg', 'unknown'))}"
                )
                return False

            logger.info(f"飞书卡片推送成功 (Status: {status_code})")
            return True

        except httpx.HTTPStatusError as e:
            logger.error(f"飞书推送 HTTP 错误: {e.response.status_code} — {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"飞书推送异常: {e}")
            return False

    def send_daily_report(
        self,
        articles: list[Article],
        max_per_card: int = 4,
        max_cards: int = 2,
        header_color: str = "blue",
        card_title: str = "📰 财经+AI 早报",
    ) -> int:
        """发送每日早报 — 将文章分拆到多张卡片并依次推送。

        Args:
            articles: 已精选的文章列表
            max_per_card: 每张卡片最多条数
            max_cards: 最多卡片数
            header_color: 卡片头部颜色
            card_title: 卡片标题前缀

        Returns:
            成功推送的卡片数
        """
        if not articles:
            logger.warning("无文章可推送")
            return 0

        # 按 category 分组，财经在前 AI 在后
        finance_articles = [a for a in articles if a.category == "finance"]
        ai_articles = [a for a in articles if a.category == "ai"]
        ordered = finance_articles + ai_articles

        # 分卡
        cards_articles: list[list[Article]] = []
        for i in range(0, len(ordered), max_per_card):
            chunk = ordered[i : i + max_per_card]
            if chunk:
                cards_articles.append(chunk)
            if len(cards_articles) >= max_cards:
                break

        total_cards = len(cards_articles)
        logger.info(
            f"准备推送: {len(ordered)} 篇文章 → {total_cards} 张卡片 "
            f"(max_per_card={max_per_card}, max_cards={max_cards})"
        )

        success_count = 0
        for idx, chunk in enumerate(cards_articles, start=1):
            card = _build_card(
                chunk,
                card_index=idx,
                total_cards=total_cards,
                header_color=header_color,
                card_title=card_title,
            )

            if self.send_card(card):
                success_count += 1
            else:
                logger.error(f"卡片 {idx}/{total_cards} 推送失败")

        logger.info(f"推送完成: {success_count}/{total_cards} 张卡片成功")
        return success_count
