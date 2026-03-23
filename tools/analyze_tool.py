"""本模块作用：在整个智能体中负责对单篇论文全文或若干片段进行最小结构化学术分析，为教学讲解与论文理解提供统一结果。"""

import re
from dataclasses import dataclass, field


FIELD_RULES = {
    "research_question": {
        "label": "研究问题",
        "keywords": ["研究问题", "问题", "目的", "目标", "旨在", "关注", "探讨", "分析"],
        "default": "文本中未明确识别出研究问题，建议查看摘要、引言或研究设计部分。",
    },
    "research_object": {
        "label": "研究对象",
        "keywords": ["研究对象", "对象", "样本", "案例", "城市", "社区", "居民", "区域", "街区"],
        "default": "文本中未明确识别出研究对象，建议补充研究区、样本或案例描述。",
    },
    "methods": {
        "label": "方法",
        "keywords": ["方法", "采用", "构建", "模型", "访谈", "问卷", "实证", "回归", "评价", "分析方法"],
        "default": "文本中未明确识别出研究方法，建议查看方法或研究设计部分。",
    },
    "data_source": {
        "label": "数据来源",
        "keywords": ["数据", "数据来源", "样本", "年鉴", "问卷", "访谈", "遥感", "POI", "统计", "文本数据"],
        "default": "文本中未明确识别出数据来源，建议查看数据说明或样本来源部分。",
    },
    "key_findings": {
        "label": "主要结论",
        "keywords": ["结论", "发现", "结果", "表明", "说明", "影响", "显著", "提升", "降低"],
        "default": "文本中未明确识别出主要结论，建议查看结果与结论部分。",
    },
    "limitations": {
        "label": "局限性",
        "keywords": ["局限", "不足", "受限", "限制", "未来", "仍需", "但", "然而"],
        "default": "文本中未明确识别出局限性，可后续结合作者讨论部分人工补充。",
    },
    "implications": {
        "label": "对城市治理/规划/安全的启示",
        "keywords": ["启示", "建议", "治理", "规划", "安全", "政策", "管理", "优化", "决策"],
        "default": "文本中未明确识别出针对城市治理、规划或安全的启示，建议结合结论做进一步人工总结。",
    },
}


@dataclass
class StructuredPaperAnalysis:
    """本数据结构作用：保存单篇论文结构化提取结果，便于命令行展示或后续模块复用。"""

    file_name: str
    document_id: str
    research_question: str
    research_object: str
    methods: str
    data_source: str
    key_findings: str
    limitations: str
    implications: str
    evidence_map: dict[str, list[str]] = field(default_factory=dict)


def normalize_analysis_text(text: str) -> str:
    """对输入文本做基础清洗。

    输入：
        text: 原始全文或片段文本。
    输出：
        适合进一步分析的清洗后文本。
    异常：
        无。
    """

    cleaned_text = text.replace("\r", "\n")
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    return cleaned_text.strip()


def combine_input_text(text_or_segments: str | list[str]) -> str:
    """将单段全文或多段片段合并为统一分析文本。

    输入：
        text_or_segments: 单个字符串或字符串列表。
    输出：
        合并后的完整文本。
    异常：
        当输入既不是字符串也不是字符串列表时，抛出 TypeError。
        当输入为空时，抛出 ValueError。
    """

    if isinstance(text_or_segments, str):
        merged_text = text_or_segments
    elif isinstance(text_or_segments, list):
        merged_text = "\n".join(str(item).strip() for item in text_or_segments if str(item).strip())
    else:
        raise TypeError("text_or_segments 必须是字符串或字符串列表。")

    merged_text = normalize_analysis_text(merged_text)
    if not merged_text:
        raise ValueError("待分析文本不能为空。")
    return merged_text


def split_into_sentences(text: str) -> list[str]:
    """将分析文本切分为句子列表。

    输入：
        text: 已清洗的文本。
    输出：
        去除空白后的句子列表。
    异常：
        无。
    """

    raw_sentences = re.split(r"[。！？!?；;\n]+", text)
    sentences: list[str] = []
    for sentence in raw_sentences:
        cleaned_sentence = re.sub(r"\s+", " ", sentence).strip()
        if cleaned_sentence:
            sentences.append(cleaned_sentence)
    return sentences


def truncate_sentence(text: str, max_length: int = 120) -> str:
    """将过长句子裁剪到适合展示的长度。

    输入：
        text: 原始句子。
        max_length: 最大保留字符数。
    输出：
        裁剪后的句子。
    异常：
        无。
    """

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if len(cleaned_text) <= max_length:
        return cleaned_text
    return cleaned_text[:max_length].rstrip() + "..."


def deduplicate_sentences(sentences: list[str]) -> list[str]:
    """按原始顺序去除重复句子。

    输入：
        sentences: 原始句子列表。
    输出：
        去重后的句子列表。
    异常：
        无。
    """

    unique_sentences: list[str] = []
    seen_sentences: set[str] = set()
    for sentence in sentences:
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        unique_sentences.append(sentence)
    return unique_sentences


def score_sentence(sentence: str, keywords: list[str]) -> float:
    """根据关键词为句子打基础分。

    输入：
        sentence: 候选句子文本。
        keywords: 当前字段的关键词列表。
    输出：
        句子的相关性分数。
    异常：
        无。
    """

    normalized_sentence = sentence.lower()
    score = 0.0

    for keyword in keywords:
        if keyword.lower() in normalized_sentence:
            if len(keyword) >= 4:
                score += 2.0
            else:
                score += 1.2

    if sentence.startswith("本文") or sentence.startswith("研究"):
        score += 0.8
    if 12 <= len(sentence) <= 80:
        score += 0.5
    if len(sentence) > 150:
        score -= 0.6

    return score


def select_field_evidence(
    sentences: list[str],
    keywords: list[str],
    top_k: int = 1,
) -> list[str]:
    """为某个分析字段选择最相关的依据句子。

    输入：
        sentences: 论文句子列表。
        keywords: 当前字段的关键词列表。
        top_k: 最多返回多少条依据句子。
    输出：
        去重并排序后的依据句子列表。
    异常：
        无。
    """

    scored_sentences: list[tuple[float, str]] = []
    for sentence in sentences:
        score = score_sentence(sentence, keywords)
        if score <= 0:
            continue
        scored_sentences.append((score, truncate_sentence(sentence)))

    scored_sentences.sort(key=lambda item: (-item[0], item[1]))

    selected_sentences: list[str] = []
    seen_sentences: set[str] = set()
    for _, sentence in scored_sentences:
        if sentence in seen_sentences:
            continue
        selected_sentences.append(sentence)
        seen_sentences.add(sentence)
        if len(selected_sentences) >= top_k:
            break

    return selected_sentences


def build_field_summary(evidence_sentences: list[str], default_text: str) -> str:
    """将依据句子整理为字段摘要文本。

    输入：
        evidence_sentences: 当前字段的依据句子列表。
        default_text: 没有识别结果时的默认提示。
    输出：
        结构化字段摘要。
    异常：
        无。
    """

    if not evidence_sentences:
        return default_text
    return "；".join(deduplicate_sentences(evidence_sentences))


def analyze_single_paper(
    text_or_segments: str | list[str],
    file_name: str = "未知论文",
    document_id: str = "unknown_document",
) -> StructuredPaperAnalysis:
    """对单篇论文全文或若干片段执行结构化提取。

    输入：
        text_or_segments: 论文全文字符串，或若干论文片段组成的列表。
        file_name: 论文文件名。
        document_id: 论文文档编号。
    输出：
        单篇论文结构化分析结果对象。
    异常：
        当输入文本为空或类型错误时，抛出对应异常。
    """

    merged_text = combine_input_text(text_or_segments)
    sentences = split_into_sentences(merged_text)
    if not sentences:
        raise ValueError("文本过短，无法切分出有效句子进行分析。")

    evidence_map: dict[str, list[str]] = {}
    field_values: dict[str, str] = {}

    for field_name, rule in FIELD_RULES.items():
        evidence_sentences = select_field_evidence(sentences, rule["keywords"])
        evidence_map[field_name] = evidence_sentences
        field_values[field_name] = build_field_summary(evidence_sentences, rule["default"])

    return StructuredPaperAnalysis(
        file_name=file_name,
        document_id=document_id,
        research_question=field_values["research_question"],
        research_object=field_values["research_object"],
        methods=field_values["methods"],
        data_source=field_values["data_source"],
        key_findings=field_values["key_findings"],
        limitations=field_values["limitations"],
        implications=field_values["implications"],
        evidence_map=evidence_map,
    )


def format_analysis_result(result: StructuredPaperAnalysis) -> str:
    """将结构化分析结果整理为适合命令行展示的文本。

    输入：
        result: 单篇论文结构化分析结果对象。
    输出：
        带标题与分项字段的展示文本。
    异常：
        无。
    """

    lines = [
        "结构化学术分析结果：",
        f"论文名称：{result.file_name}",
        f"文档编号：{result.document_id}",
        f"1. 研究问题：{result.research_question}",
        f"2. 研究对象：{result.research_object}",
        f"3. 方法：{result.methods}",
        f"4. 数据来源：{result.data_source}",
        f"5. 主要结论：{result.key_findings}",
        f"6. 局限性：{result.limitations}",
        f"7. 对城市治理/规划/安全的启示：{result.implications}",
        "依据片段：",
    ]

    for field_name, rule in FIELD_RULES.items():
        evidence_sentences = result.evidence_map.get(field_name, [])
        evidence_text = "；".join(evidence_sentences) if evidence_sentences else "未识别到明确依据片段。"
        lines.append(f"- {rule['label']}：{evidence_text}")

    return "\n".join(lines)


def run_analysis_demo() -> None:
    """执行分析工具的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印结构化分析结果。
    异常：
        无。
    """

    demo_text = (
        "本文以某沿海超大城市的社区更新项目为研究对象，关注公共服务设施可达性变化对居民满意度的影响。"
        "研究采用问卷调查、POI 数据分析与多元回归方法。"
        "数据来源包括 2023 年社区问卷、城市开放 POI 数据和统计年鉴。"
        "结果表明，设施步行可达性提升能够显著改善居民对社区更新的评价。"
        "然而，样本主要集中于中心城区，外部可推广性仍然受限。"
        "研究建议在城市治理与规划中，将十五分钟生活圈与道路安全整治协同推进。"
    )

    result = analyze_single_paper(
        text_or_segments=demo_text,
        file_name="demo_paper.pdf",
        document_id="demo_paper",
    )
    print(format_analysis_result(result))


if __name__ == "__main__":
    run_analysis_demo()

