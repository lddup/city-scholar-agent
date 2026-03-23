"""本模块作用：在整个智能体中负责基于论文切块结果执行最小检索，并返回可供问答使用的相关片段。"""

import re
from dataclasses import dataclass


COMMON_STOP_WORDS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "与",
    "及",
    "对",
    "中",
    "及其",
    "一个",
    "一种",
    "如何",
    "什么",
    "哪些",
    "这个",
    "那个",
    "我们",
    "你们",
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "what",
    "which",
}


@dataclass
class RetrievedChunk:
    """本数据结构作用：保存一次检索命中的文本块及其最小检索信息。"""

    chunk_id: str
    document_id: str
    text: str
    snippet: str
    score: float
    matched_terms: list[str]
    metadata: dict[str, object]


def normalize_search_text(text: str) -> str:
    """对输入文本做基础归一化处理。

    输入：
        text: 原始文本。
    输出：
        统一大小写并压缩空白后的文本。
    异常：
        无。
    """

    normalized_text = text.lower()
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    return normalized_text.strip()


def deduplicate_terms(terms: list[str]) -> list[str]:
    """按出现顺序对检索词去重。

    输入：
        terms: 原始检索词列表。
    输出：
        去重后的检索词列表。
    异常：
        无。
    """

    unique_terms: list[str] = []
    seen_terms: set[str] = set()

    for term in terms:
        cleaned_term = term.strip().lower()
        if not cleaned_term or cleaned_term in seen_terms:
            continue
        seen_terms.add(cleaned_term)
        unique_terms.append(cleaned_term)

    return unique_terms


def extract_search_terms(text: str) -> list[str]:
    """从问题文本中提取适合最小检索的关键词。

    输入：
        text: 用户问题文本。
    输出：
        去重后的检索词列表。
    异常：
        无。
    """

    normalized_text = normalize_search_text(text)
    english_terms = re.findall(r"[a-z0-9]{2,}", normalized_text)
    chinese_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", text)

    terms: list[str] = []
    for term in english_terms:
        if term not in COMMON_STOP_WORDS:
            terms.append(term)

    for phrase in chinese_phrases:
        if phrase not in COMMON_STOP_WORDS:
            terms.append(phrase)
        if len(phrase) >= 2:
            for index in range(len(phrase) - 1):
                bigram = phrase[index : index + 2]
                if bigram not in COMMON_STOP_WORDS:
                    terms.append(bigram)
        if len(phrase) >= 3:
            for index in range(len(phrase) - 2):
                trigram = phrase[index : index + 3]
                if trigram not in COMMON_STOP_WORDS:
                    terms.append(trigram)

    return deduplicate_terms(terms)


def calculate_chunk_score(
    question: str,
    chunk_record: dict[str, object],
) -> tuple[float, list[str]]:
    """计算单个文本块与问题之间的基础相关性分数。

    输入：
        question: 用户问题文本。
        chunk_record: 单个文本块记录。
    输出：
        第一项是分数，第二项是命中的检索词列表。
    异常：
        无。
    """

    question_text = normalize_search_text(question)
    query_terms = extract_search_terms(question)
    chunk_text = str(chunk_record.get("text", ""))
    normalized_chunk_text = normalize_search_text(chunk_text)

    metadata = chunk_record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    file_name = normalize_search_text(str(metadata.get("file_name", "")))

    matched_terms: list[str] = []
    score = 0.0

    if question_text and question_text in normalized_chunk_text:
        score += 8.0

    for term in query_terms:
        text_count = normalized_chunk_text.count(term)
        name_count = file_name.count(term)
        total_count = text_count + name_count
        if total_count <= 0:
            continue

        matched_terms.append(term)
        if len(term) >= 6:
            term_weight = 2.4
        elif len(term) >= 4:
            term_weight = 1.8
        else:
            term_weight = 1.1

        score += min(text_count, 3) * term_weight
        score += min(name_count, 2) * (term_weight + 0.3)

    if query_terms:
        coverage_ratio = len(matched_terms) / len(query_terms)
        score += coverage_ratio * 3.0

    return score, matched_terms


def build_chunk_snippet(
    chunk_text: str,
    matched_terms: list[str],
    max_length: int = 180,
) -> str:
    """从命中文本块中截取便于展示的片段。

    输入：
        chunk_text: 文本块原文。
        matched_terms: 命中的检索词列表。
        max_length: 片段最大长度。
    输出：
        适合展示给用户的片段字符串。
    异常：
        无。
    """

    cleaned_text = re.sub(r"\s+", " ", chunk_text).strip()
    if len(cleaned_text) <= max_length:
        return cleaned_text

    if matched_terms:
        first_term_position = -1
        for term in matched_terms:
            position = cleaned_text.lower().find(term.lower())
            if position >= 0 and (first_term_position < 0 or position < first_term_position):
                first_term_position = position

        if first_term_position >= 0:
            start_index = max(first_term_position - 40, 0)
            end_index = min(start_index + max_length, len(cleaned_text))
            snippet = cleaned_text[start_index:end_index].strip()
            if start_index > 0:
                snippet = "..." + snippet
            if end_index < len(cleaned_text):
                snippet = snippet + "..."
            return snippet

    return cleaned_text[:max_length].rstrip() + "..."


def retrieve_relevant_chunks(
    question: str,
    chunk_records: list[dict[str, object]],
    top_k: int = 3,
    min_score: float = 1.0,
) -> list[RetrievedChunk]:
    """根据用户问题召回最相关的论文片段。

    输入：
        question: 用户问题文本。
        chunk_records: 已构建的知识库文本块记录列表。
        top_k: 最多返回多少条结果。
        min_score: 最低命中分数阈值。
    输出：
        按相关性排序的检索结果列表。
    异常：
        当问题为空时，抛出 ValueError。
    """

    if not question or not question.strip():
        raise ValueError("问题不能为空。")

    scored_chunks: list[RetrievedChunk] = []
    for chunk_record in chunk_records:
        score, matched_terms = calculate_chunk_score(question, chunk_record)
        if score < min_score:
            continue

        metadata = chunk_record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        chunk_text = str(chunk_record.get("text", ""))
        scored_chunks.append(
            RetrievedChunk(
                chunk_id=str(chunk_record.get("chunk_id", "")),
                document_id=str(chunk_record.get("document_id", "")),
                text=chunk_text,
                snippet=build_chunk_snippet(chunk_text, matched_terms),
                score=score,
                matched_terms=matched_terms,
                metadata=metadata,
            )
        )

    scored_chunks.sort(
        key=lambda item: (
            -item.score,
            str(item.metadata.get("file_name", "")),
            int(item.metadata.get("chunk_index", 0)),
        )
    )
    return scored_chunks[:top_k]


def run_retriever_demo() -> None:
    """执行 retriever 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印检索结果示例。
    异常：
        无。
    """

    demo_records = [
        {
            "chunk_id": "demo_chunk_0001",
            "document_id": "demo_paper",
            "text": "城市更新项目中，公共服务设施的可达性提升与居民满意度存在明显正相关。",
            "metadata": {
                "file_name": "demo_paper.pdf",
                "source_path": "data/raw_papers/demo_paper.pdf",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 35,
                "page_numbers": [1],
            },
        },
        {
            "chunk_id": "demo_chunk_0002",
            "document_id": "demo_paper",
            "text": "研究指出，步行十五分钟生活圈能够改善日常服务覆盖，但需要结合道路安全治理。",
            "metadata": {
                "file_name": "demo_paper.pdf",
                "source_path": "data/raw_papers/demo_paper.pdf",
                "chunk_index": 1,
                "start_char": 36,
                "end_char": 76,
                "page_numbers": [2],
            },
        },
    ]

    results = retrieve_relevant_chunks("城市更新如何影响公共服务可达性？", demo_records)
    print("Retriever Demo")
    print(f"召回数量：{len(results)}")
    for index, item in enumerate(results, start=1):
        print(f"{index}. 分数：{item.score:.2f} | 片段：{item.snippet}")


if __name__ == "__main__":
    run_retriever_demo()
