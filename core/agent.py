"""本模块作用：在整个智能体中负责串联论文加载、检索、问答与最小学术分析，形成可运行的本地论文助手闭环。"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from core.prompts import (
    build_answer_suffix,
    build_answer_system_prompt,
    build_answer_task_prompt,
    build_empty_library_message,
    build_no_result_message,
)
from rag.loader import list_pdf_files
from rag.parser import ParsedDocument, parse_pdf_files
from rag.retriever import RetrievedChunk, extract_search_terms, retrieve_relevant_chunks
from rag.splitter import build_knowledge_base_records
from tools.analyze_tool import StructuredPaperAnalysis, analyze_single_paper, format_analysis_result


@dataclass
class KnowledgeBaseState:
    """本数据结构作用：保存当前本地论文知识库的最小运行状态。"""

    raw_papers_dir: str
    pdf_files: list[str] = field(default_factory=list)
    documents: list[ParsedDocument] = field(default_factory=list)
    chunk_records: list[dict[str, object]] = field(default_factory=list)
    parse_errors: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AgentAnswer:
    """本数据结构作用：保存一次问答的最终输出，供命令行或后续界面展示。"""

    question: str
    model_answer: str
    sources: list[dict[str, object]]
    retrieved_count: int
    used_prompt: str


@dataclass
class PaperAnalysisResponse:
    """本数据结构作用：保存一次单篇论文结构化分析结果，供主程序展示。"""

    target: str
    status_message: str
    analysis: StructuredPaperAnalysis | None
    formatted_output: str


class CityScholarAgent:
    """本类作用：封装最小论文助手闭环，统一管理建库、检索、问答与单篇论文分析逻辑。"""

    def __init__(
        self,
        raw_papers_dir: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 3,
    ) -> None:
        """初始化最小论文代理。

        输入：
            raw_papers_dir: 本地 PDF 论文目录。
            chunk_size: 单个文本块的最大字符数。
            chunk_overlap: 相邻文本块之间的重叠字符数。
            top_k: 单次检索最多返回多少条结果。
        输出：
            无。
        异常：
            当切块参数不合法时，相关异常会在后续建库阶段抛出。
        """

        self.raw_papers_dir = Path(raw_papers_dir).expanduser().resolve()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.knowledge_base: KnowledgeBaseState | None = None

    def build_knowledge_base(self) -> KnowledgeBaseState:
        """根据本地 PDF 构建最小知识库。

        输入：
            无。
        输出：
            当前知识库状态对象。
        异常：
            当 PDF 解析依赖缺失时，抛出 ImportError。
            当目录访问失败或切块参数不合法时，抛出对应异常。
        """

        pdf_paths = list_pdf_files(self.raw_papers_dir)
        documents, parse_errors = parse_pdf_files(pdf_paths)
        chunk_records = build_knowledge_base_records(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self.knowledge_base = KnowledgeBaseState(
            raw_papers_dir=str(self.raw_papers_dir),
            pdf_files=[str(path) for path in pdf_paths],
            documents=documents,
            chunk_records=chunk_records,
            parse_errors=parse_errors,
        )
        return self.knowledge_base

    def ensure_knowledge_base_ready(self) -> KnowledgeBaseState:
        """确保知识库已准备完成。

        输入：
            无。
        输出：
            当前知识库状态对象。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        if self.knowledge_base is None:
            return self.build_knowledge_base()
        return self.knowledge_base

    def list_available_papers(self) -> list[dict[str, object]]:
        """列出当前知识库中可分析的论文信息。

        输入：
            无。
        输出：
            论文信息列表，每项包含序号、文件名、文档编号与页数统计。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        paper_items: list[dict[str, object]] = []
        for index, document in enumerate(knowledge_base.documents, start=1):
            paper_items.append(
                {
                    "index": index,
                    "file_name": document.file_name,
                    "document_id": document.document_id,
                    "total_pages": document.total_pages,
                    "total_characters": document.total_characters,
                }
            )
        return paper_items

    def find_document(self, target: str | None = None) -> ParsedDocument | None:
        """根据序号、文件名或文档编号定位目标论文。

        输入：
            target: 论文序号、文件名、文档编号或其片段。
        输出：
            命中的论文对象；若未找到则返回 None。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return None

        if target is None or not target.strip():
            return knowledge_base.documents[0]

        normalized_target = target.strip().lower()
        if normalized_target.isdigit():
            document_index = int(normalized_target) - 1
            if 0 <= document_index < len(knowledge_base.documents):
                return knowledge_base.documents[document_index]
            return None

        for document in knowledge_base.documents:
            if normalized_target == document.file_name.lower() or normalized_target == document.document_id.lower():
                return document

        for document in knowledge_base.documents:
            if normalized_target in document.file_name.lower() or normalized_target in document.document_id.lower():
                return document

        return None

    def answer(self, question: str) -> AgentAnswer:
        """根据用户问题生成最小回答结果。

        输入：
            question: 用户问题文本。
        输出：
            包含模型回答与来源依据的结果对象。
        异常：
            当问题为空时，抛出 ValueError。
            当知识库尚未构建且自动构建失败时，抛出对应异常。
        """

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("问题不能为空。")

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.chunk_records:
            return AgentAnswer(
                question=cleaned_question,
                model_answer=build_empty_library_message(),
                sources=[],
                retrieved_count=0,
                used_prompt=build_answer_system_prompt(),
            )

        retrieved_chunks = retrieve_relevant_chunks(
            question=cleaned_question,
            chunk_records=knowledge_base.chunk_records,
            top_k=self.top_k,
        )

        context_blocks = build_context_blocks(retrieved_chunks)
        used_prompt = (
            build_answer_system_prompt()
            + "\n\n"
            + build_answer_task_prompt(cleaned_question, context_blocks)
        )

        if not retrieved_chunks:
            return AgentAnswer(
                question=cleaned_question,
                model_answer=build_no_result_message(cleaned_question),
                sources=[],
                retrieved_count=0,
                used_prompt=used_prompt,
            )

        model_answer = synthesize_answer(cleaned_question, retrieved_chunks)
        sources = build_source_entries(retrieved_chunks)
        return AgentAnswer(
            question=cleaned_question,
            model_answer=model_answer,
            sources=sources,
            retrieved_count=len(retrieved_chunks),
            used_prompt=used_prompt,
        )

    def analyze_paper(self, target: str | None = None) -> PaperAnalysisResponse:
        """对指定论文执行单篇结构化学术分析。

        输入：
            target: 论文序号、文件名、文档编号或其片段；为空时默认分析第一篇。
        输出：
            包含结构化分析结果与展示文本的响应对象。
        异常：
            当自动建库失败时，抛出对应异常。
        """

        knowledge_base = self.ensure_knowledge_base_ready()
        if not knowledge_base.documents:
            return PaperAnalysisResponse(
                target=target or "默认第一篇论文",
                status_message="当前没有可分析的论文。请先放入 PDF 并确认其能够被成功解析。",
                analysis=None,
                formatted_output="当前没有可分析的论文。",
            )

        document = self.find_document(target)
        if document is None:
            return PaperAnalysisResponse(
                target=target or "默认第一篇论文",
                status_message="未找到匹配的论文，请先使用 papers 查看可用论文列表。",
                analysis=None,
                formatted_output="未找到匹配的论文，请先使用 papers 查看可用论文列表。",
            )

        analysis = analyze_single_paper(
            text_or_segments=document.full_text,
            file_name=document.file_name,
            document_id=document.document_id,
        )
        return PaperAnalysisResponse(
            target=target or document.file_name,
            status_message=f"已完成对《{document.file_name}》的结构化提取。",
            analysis=analysis,
            formatted_output=format_analysis_result(analysis),
        )


def split_text_into_sentences(text: str) -> list[str]:
    """将文本按中文与英文常见分隔符切分为句子。

    输入：
        text: 原始文本。
    输出：
        清洗后的句子列表。
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


def truncate_text(text: str, max_length: int = 120) -> str:
    """将文本裁剪到适合回答展示的长度。

    输入：
        text: 原始文本。
        max_length: 最大保留字符数。
    输出：
        适合展示的短文本。
    异常：
        无。
    """

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if len(cleaned_text) <= max_length:
        return cleaned_text
    return cleaned_text[:max_length].rstrip() + "..."


def score_supporting_sentence(
    question: str,
    sentence: str,
    retrieval_score: float,
) -> float:
    """计算候选支撑句与问题的相关程度。

    输入：
        question: 用户问题文本。
        sentence: 候选句子文本。
        retrieval_score: 句子所属文本块的检索分数。
    输出：
        用于排序的句子分数。
    异常：
        无。
    """

    question_terms = extract_search_terms(question)
    sentence_text = sentence.lower()
    score = retrieval_score * 0.5

    if question.strip().lower() in sentence_text:
        score += 6.0

    for term in question_terms:
        if term in sentence_text:
            if len(term) >= 6:
                score += 2.2
            elif len(term) >= 4:
                score += 1.7
            else:
                score += 1.0

    if len(sentence.strip()) < 8:
        score -= 1.0

    return score


def select_supporting_sentences(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    max_sentences: int = 3,
) -> list[str]:
    """从召回片段中选出最能支撑回答的句子。

    输入：
        question: 用户问题文本。
        retrieved_chunks: 已召回的文本块列表。
        max_sentences: 最多保留多少句。
    输出：
        适合直接拼装为回答的句子列表。
    异常：
        无。
    """

    sentence_candidates: list[tuple[float, str]] = []
    for chunk in retrieved_chunks:
        for sentence in split_text_into_sentences(chunk.text):
            sentence_score = score_supporting_sentence(question, sentence, chunk.score)
            if sentence_score <= 0:
                continue
            sentence_candidates.append((sentence_score, truncate_text(sentence)))

    sentence_candidates.sort(key=lambda item: (-item[0], item[1]))

    selected_sentences: list[str] = []
    seen_sentences: set[str] = set()
    for _, sentence in sentence_candidates:
        if sentence in seen_sentences:
            continue
        selected_sentences.append(sentence)
        seen_sentences.add(sentence)
        if len(selected_sentences) >= max_sentences:
            break

    return selected_sentences


def build_context_blocks(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    """将召回结果转换为后续回答模块可使用的上下文块。

    输入：
        retrieved_chunks: 已召回的文本块列表。
    输出：
        带有论文名、页码和片段的文本块列表。
    异常：
        无。
    """

    context_blocks: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        page_numbers = chunk.metadata.get("page_numbers", [])
        page_text = "、".join(str(number) for number in page_numbers) if page_numbers else "未知页码"
        file_name = str(chunk.metadata.get("file_name", "未知论文"))
        context_blocks.append(
            f"[来源 {index}] 论文：{file_name} | 页码：{page_text} | 片段：{chunk.snippet}"
        )
    return context_blocks


def synthesize_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """基于召回片段生成最小回答文本。

    输入：
        question: 用户问题文本。
        retrieved_chunks: 已召回的文本块列表。
    输出：
        适合直接展示给用户的回答文本。
    异常：
        无。
    """

    supporting_sentences = select_supporting_sentences(question, retrieved_chunks)
    answer_lines = ["根据当前召回的论文片段，可以整理出以下回答："]

    if supporting_sentences:
        for index, sentence in enumerate(supporting_sentences, start=1):
            answer_lines.append(f"{index}. {sentence}")
    else:
        answer_lines.append("当前能够召回相关片段，但还不足以抽取出稳定的支撑句。")
        for index, chunk in enumerate(retrieved_chunks[:2], start=1):
            answer_lines.append(f"{index}. {truncate_text(chunk.snippet)}")

    answer_lines.append(build_answer_suffix())
    return "\n".join(answer_lines)


def build_source_entries(
    retrieved_chunks: list[RetrievedChunk],
    max_sources: int = 3,
) -> list[dict[str, object]]:
    """将召回结果转换为可展示的来源依据列表。

    输入：
        retrieved_chunks: 已召回的文本块列表。
        max_sources: 最多保留多少条来源信息。
    输出：
        适合界面或命令行展示的来源依据列表。
    异常：
        无。
    """

    sources: list[dict[str, object]] = []
    for chunk in retrieved_chunks[:max_sources]:
        sources.append(
            {
                "file_name": str(chunk.metadata.get("file_name", "未知论文")),
                "source_path": str(chunk.metadata.get("source_path", "")),
                "page_numbers": chunk.metadata.get("page_numbers", []),
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "score": round(chunk.score, 2),
                "snippet": chunk.snippet,
                "matched_terms": chunk.matched_terms,
            }
        )
    return sources


def run_agent_demo() -> None:
    """执行 agent 模块的最小演示。

    输入：
        无。
    输出：
        无。函数会直接打印问答与分析示例。
    异常：
        无。
    """

    demo_agent = CityScholarAgent(raw_papers_dir="data/raw_papers")
    demo_agent.knowledge_base = KnowledgeBaseState(
        raw_papers_dir="data/raw_papers",
        pdf_files=["data/raw_papers/demo_paper.pdf"],
        documents=[
            ParsedDocument(
                document_id="demo_paper",
                file_name="demo_paper.pdf",
                source_path="data/raw_papers/demo_paper.pdf",
                total_pages=2,
                total_characters=88,
                full_text=(
                    "本文以某沿海超大城市的社区更新项目为研究对象，关注公共服务设施可达性变化对居民满意度的影响。"
                    "研究采用问卷调查、POI 数据分析与多元回归方法。"
                    "数据来源包括社区问卷、城市开放 POI 数据和统计年鉴。"
                    "结果表明，设施步行可达性提升能够显著改善居民对社区更新的评价。"
                    "然而，样本主要集中于中心城区，外部可推广性仍然受限。"
                    "研究建议在城市治理与规划中，将十五分钟生活圈与道路安全整治协同推进。"
                ),
                pages=[],
            )
        ],
        chunk_records=[
            {
                "chunk_id": "demo_chunk_0001",
                "document_id": "demo_paper",
                "text": "城市更新项目中，公共服务设施的可达性提升与居民满意度存在明显正相关。",
                "metadata": {
                    "file_name": "demo_paper.pdf",
                    "source_path": "data/raw_papers/demo_paper.pdf",
                    "chunk_index": 0,
                    "page_numbers": [1],
                },
            },
            {
                "chunk_id": "demo_chunk_0002",
                "document_id": "demo_paper",
                "text": "步行十五分钟生活圈能够改善日常服务覆盖，但需要结合道路安全治理。",
                "metadata": {
                    "file_name": "demo_paper.pdf",
                    "source_path": "data/raw_papers/demo_paper.pdf",
                    "chunk_index": 1,
                    "page_numbers": [2],
                },
            },
        ],
        parse_errors=[],
    )

    answer_result = demo_agent.answer("城市更新如何影响公共服务可达性？")
    print("Agent Demo - 问答")
    print("模型回答：")
    print(answer_result.model_answer)
    print("来源依据：")
    for index, source in enumerate(answer_result.sources, start=1):
        print(f"{index}. {source['file_name']} | 页码：{source['page_numbers']} | 片段：{source['snippet']}")

    analysis_result = demo_agent.analyze_paper("1")
    print("\nAgent Demo - 结构化分析")
    print(analysis_result.formatted_output)


if __name__ == "__main__":
    run_agent_demo()

