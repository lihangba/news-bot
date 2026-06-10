"""AI 模块 — DeepSeek 客户端封装（OpenAI 兼容接口），负责语义去重与摘要生成。"""

import json
import os
import re
from loguru import logger
from openai import OpenAI


class DeepSeekClient:
    """DeepSeek API 客户端。

    使用 OpenAI 兼容接口调用 deepseek-chat，温度固定 0.3 确保输出稳定。

    使用方式::

        client = DeepSeekClient(api_key="sk-xxx", base_url="https://api.deepseek.com")
        groups = client.detect_duplicates(articles)   # → [[0, 3], [1, 5]]
        summaries = client.generate_summaries(articles)  # → ["摘要1", "摘要2", ...]
    """

    # ---- 去重 Prompt ----
    DEDUP_SYSTEM_PROMPT = """\
你是一个严格的新闻去重编辑。你的任务是逐一判断「今天候选新闻」是否与「近3天已推送新闻」报道了同一核心事件。

## 判断铁律（非常重要）
- 仅当候选新闻与某条历史新闻的**核心事件完全相同**时才判定为重复。
  核心事件指：同一家公司同一条公告、同一条政策发布、同一起具体事件（某公司财报、某产品发布、某人被调查）。
- **以下情况绝不可判为重复**：
  - 同一领域的不同新闻（如"特斯拉发布新车"和"特斯拉股价上涨"不重复）
  - 同一话题但不同角度（如"央行降息"和"降息对楼市影响"不重复）
  - 不同公司的同类事件（如"Google发布新模型"和"OpenAI发布新模型"不重复）
  - 泛指 vs 具体事件（如"AI行业受关注"和"OpenAI融资10亿"不重复）

## 判断流程（逐条处理每条 candidate）
1. 与 history 中所有新闻逐一比对
2. 任一历史新闻的核心事件与本条候选相同 → is_duplicate=true
3. 所有历史新闻均不重复 → is_duplicate=false

## 输出格式
返回严格的 JSON 数组，与 candidates 一一对应，顺序必须一致：
[
  {"id": 0, "is_duplicate": false, "reason": ""},
  {"id": 1, "is_duplicate": true, "reason": "与历史新闻「央行宣布降准0.5个百分点」报道同一政策发布"},
  {"id": 2, "is_duplicate": false, "reason": ""},
  ...
]
reason 字段：is_duplicate 为 true 时必须写明与哪条历史新闻重复、因何重复；为 false 时填空字符串。"""

    # ---- 摘要生成 Prompt ----
    SUMMARY_SYSTEM_PROMPT = """\
你是一个专业的中文财经科技新闻编辑。你的任务是为每条新闻生成一句通顺的中文摘要。

## 要求
1. 每条摘要**不超过 50 个字**。
2. **中文原文**：提炼核心信息，精炼成一两句话，不得照搬原文。
3. **英文原文**：翻译成通顺的中文，注意中英文表达习惯差异，可适当意译，确保中文读者能看懂。
4. 保留关键信息：人名、公司名、数字、百分比、时间节点。
5. 语言风格：客观、简洁、信息密度高，像《华尔街见闻》的快讯风格。
6. 禁止纯直译，禁止生硬机翻。

## 输出格式
返回一个严格的 JSON 数组，每个元素是一条摘要字符串，与输入的 articles 数组一一对应：
["摘要1", "摘要2", "摘要3", ...]"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError(
                "DeepSeek API key 未设置 — 请通过参数传入或设置 DEEPSEEK_API_KEY 环境变量"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(
            f"DeepSeek 客户端已初始化 (model={model}, temperature={temperature})"
        )

    # ------------------------------------------------------------------
    # 底层调用
    # ------------------------------------------------------------------

    def _chat_json(self, system: str, user: str, label: str = "AI调用") -> str:
        """发送 chat 请求并返回原始文本（期望 JSON 响应）。"""
        logger.info(f"[{label}] 发送请求, model={self.model}, prompt_len={len(user)}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = response.choices[0].message.content or ""
            logger.info(
                f"[{label}] 响应完成, tokens: "
                f"in={response.usage.prompt_tokens if response.usage else '?'}, "
                f"out={response.usage.completion_tokens if response.usage else '?'}"
            )
            return content
        except Exception as e:
            logger.error(f"[{label}] 调用失败: {e}")
            raise

    @staticmethod
    def _extract_json(text: str, label: str = "") -> str:
        """从可能包含 Markdown 代码块的响应中提取纯净 JSON。"""
        # 尝试提取 ```json ... ``` 中的内容
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            return match.group(1).strip()
        # 尝试匹配裸 JSON 数组或对象
        text = text.strip()
        if (text.startswith("[") or text.startswith("{")):
            return text
        logger.warning(f"[{label}] 响应似乎不是 JSON 格式: {text[:200]}")
        return text

    # ------------------------------------------------------------------
    # 语义去重
    # ------------------------------------------------------------------

    def detect_duplicates(
        self, candidates: list, history: list[dict]
    ) -> list[dict]:
        """检测语义重复 — 一次 API 调用，逐条比对候选新闻与近3天历史。

        Args:
            candidates: 今天的候选 Article 列表（≤20 篇，已经过优先级排序）
            history: 近 3 天已推送文章列表 [{"title":..., "summary":...}, ...]

        Returns:
            [{"id": 0, "is_duplicate": False, "reason": ""},
             {"id": 1, "is_duplicate": True, "reason": "与历史新闻「xxx」重复..."}, ...]
        """
        if not candidates:
            return []

        # 构建候选新闻输入（含 id，供模型标注）
        candidate_items = []
        for i, a in enumerate(candidates):
            candidate_items.append({
                "id": i,
                "title": a.title,
                "summary": a.summary[:200] if a.summary else "",
            })

        # 构建历史新闻输入（只传 title + summary，节省 token）
        history_items = []
        for h in history:
            history_items.append({
                "title": h.get("title", ""),
                "summary": h.get("summary", "")[:200],
            })

        user_prompt = f"""## 今天候选新闻（需逐条判断是否重复，共 {len(candidate_items)} 条）
{json.dumps(candidate_items, ensure_ascii=False, indent=2)}

## 近3天已推送新闻（去重参考基准，共 {len(history_items)} 条）
{json.dumps(history_items, ensure_ascii=False, indent=2)}

请严格按 JSON 数组格式返回每条候选新闻的去重判断结果。"""

        try:
            raw = self._chat_json(
                system=self.DEDUP_SYSTEM_PROMPT,
                user=user_prompt,
                label="语义去重",
            )
            json_str = self._extract_json(raw, "去重")
            results: list[dict] = json.loads(json_str)

            # 校验
            if not isinstance(results, list):
                logger.warning(f"去重返回格式异常: {type(results)}, 视为全部不重复")
                return [{"id": i, "is_duplicate": False, "reason": ""}
                        for i in range(len(candidates))]

            # 确保与候选一一对应
            if len(results) != len(candidates):
                logger.warning(
                    f"去重结果数量不匹配: 期望 {len(candidates)}, 实际 {len(results)}"
                )
                # 补齐或截断
                while len(results) < len(candidates):
                    results.append(
                        {"id": len(results), "is_duplicate": False, "reason": ""}
                    )
                results = results[: len(candidates)]

            # 清洗字段
            cleaned = []
            for r in results:
                cleaned.append({
                    "id": int(r.get("id", 0)),
                    "is_duplicate": bool(r.get("is_duplicate", False)),
                    "reason": str(r.get("reason", ""))
                    if r.get("is_duplicate")
                    else "",
                })

            dup_count = sum(1 for c in cleaned if c["is_duplicate"])
            logger.info(
                f"语义去重完成: {len(candidates)} 条候选 + {len(history_items)} 条历史 "
                f"→ 判定 {dup_count} 条重复, {len(candidates) - dup_count} 条保留"
            )
            return cleaned

        except json.JSONDecodeError as e:
            logger.error(f"去重响应 JSON 解析失败: {e}\n原始响应: {raw[:500]}")
            return [{"id": i, "is_duplicate": False, "reason": ""}
                    for i in range(len(candidates))]
        except Exception as e:
            logger.error(f"语义去重异常: {e}")
            return [{"id": i, "is_duplicate": False, "reason": ""}
                    for i in range(len(candidates))]

    # ------------------------------------------------------------------
    # 摘要生成
    # ------------------------------------------------------------------

    def generate_summaries(self, articles: list) -> list[str]:
        """为文章列表生成中文摘要（≤50 字）。

        Args:
            articles: Article 列表

        Returns:
            与输入一一对应的中文摘要列表
        """
        if not articles:
            return []

        # 构建输入
        items = []
        for i, a in enumerate(articles):
            items.append({
                "id": i,
                "title": a.title,
                "original_summary": a.summary[:300] if a.summary else "",
                "language": "英文" if a.is_english else "中文",
            })

        user_prompt = f"""请为以下 {len(items)} 条新闻各生成一句中文摘要（每条 ≤50 字）：

{json.dumps(items, ensure_ascii=False, indent=2)}

返回 JSON 数组，与输入顺序一一对应：["摘要1", "摘要2", ...]"""

        try:
            raw = self._chat_json(
                system=self.SUMMARY_SYSTEM_PROMPT,
                user=user_prompt,
                label="摘要生成",
            )
            json_str = self._extract_json(raw, "摘要")
            summaries: list[str] = json.loads(json_str)

            if not isinstance(summaries, list) or len(summaries) != len(articles):
                logger.warning(
                    f"摘要数量不匹配: 期望 {len(articles)}, 实际 {len(summaries) if isinstance(summaries, list) else '非列表'}"
                )
                # 补齐或截断
                if isinstance(summaries, list):
                    while len(summaries) < len(articles):
                        summaries.append(articles[len(summaries)].summary[:50])
                    summaries = summaries[:len(articles)]
                else:
                    summaries = [a.summary[:50] for a in articles]

            # 确保每条摘要不超过 50 字
            trimmed = []
            for s in summaries:
                s = str(s) if s else ""
                trimmed.append(s[:50] if len(s) > 50 else s)

            logger.info(f"摘要生成完成: {len(trimmed)} 条")
            return trimmed

        except json.JSONDecodeError as e:
            logger.error(f"摘要响应 JSON 解析失败: {e}\n原始响应: {raw[:500]}")
            # 兜底：直接使用原始摘要截断
            return [a.summary[:50] for a in articles]
        except Exception as e:
            logger.error(f"摘要生成异常: {e}")
            return [a.summary[:50] for a in articles]
