"""本模块作用：提供 CityScholar-Agent 的命令行主流程，完成本地建库、检索问答与单篇论文分析，并支持 DashScope 大模型增强。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import get_app_config
from core.agent import AgentAnswer, CityScholarAgent, KnowledgeBaseState, PaperAnalysisResponse
from llm_dashscope import DashScopeClient, parse_first_json_object
from tools.analyze_tool import StructuredPaperAnalysis, format_analysis_result


def ensure_directories(paths: list[Path]) -> None:
    """创建运行所需目录。"""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def initialize_llm_client(config: dict[str, Path | str | int | bool]) -> DashScopeClient | None:
    """根据配置初始化 DashScope 客户端。"""

    api_key = str(config.get("dashscope_api_key", "")).strip()
    base_url = str(config.get("dashscope_base_url", "")).strip()
    timeout_sec = int(config.get("dashscope_timeout_sec", 45))
    if not api_key:
        return None
    try:
        return DashScopeClient(api_key=api_key, base_url=base_url, timeout_sec=timeout_sec)
    except Exception:
        return None


def show_startup_info(
    config: dict[str, Path | str | int | bool],
    knowledge_base: KnowledgeBaseState,
    llm_client: DashScopeClient | None,
) -> None:
    """输出启动信息。"""

    llm_enabled = llm_client is not None
    answer_model = str(config.get("dashscope_answer_model", "qwen-plus"))
    analysis_model = str(config.get("dashscope_analysis_model", "qwen-max"))

    print("=" * 60)
    print(f"启动项目：{config['project_name']}")
    print(f"项目目录：{config['base_dir']}")
    print(f"论文目录：{config['raw_papers_dir']}")
    print(f"发现 PDF 数量：{len(knowledge_base.pdf_files)}")
    print(f"解析成功论文数：{len(knowledge_base.documents)}")
    print(f"可检索片段数：{len(knowledge_base.chunk_records)}")
    print(f"解析失败论文数：{len(knowledge_base.parse_errors)}")
    print(f"大模型增强：{'已启用' if llm_enabled else '未启用'}")
    if llm_enabled:
        print(f"问答模型：{answer_model}")
        print(f"分析模型：{analysis_model}")
    print("当前模式：命令行最小检索、问答与单篇论文结构化分析。")
    print("可用命令：")
    print("- 直接输入问题：执行检索问答（启用大模型时自动增强）")
    print("- papers：查看当前可分析论文列表")
    print("- analyze：分析第一篇论文（启用大模型时自动增强）")
    print("- analyze 1：按序号分析论文")
    print("- analyze 文件名关键词：按文件名或文档编号分析论文")
    print("- help：再次查看命令说明")
    print("- exit：退出程序")
    print("=" * 60)


def show_parse_error_summary(parse_errors: list[dict[str, str]], max_items: int = 3) -> None:
    """输出 PDF 解析失败摘要信息。"""

    if not parse_errors:
        return

    print("以下 PDF 在建库时解析失败，已自动跳过：")
    for error_item in parse_errors[:max_items]:
        print(f"- 文件：{error_item['file_name']}")
        print(f"  原因：{error_item['error_message']}")


def format_page_numbers(page_numbers: list[int]) -> str:
    """将页码列表转换为可展示文本。"""

    if not page_numbers:
        return "未知页码"
    return "、".join(str(number) for number in page_numbers)


def build_answer_context(result: AgentAnswer) -> str:
    """将召回来源整理为模型可读上下文。"""

    context_lines: list[str] = []
    for index, source in enumerate(result.sources, start=1):
        page_text = format_page_numbers(source.get("page_numbers", []))
        file_name = str(source.get("file_name", "未知论文"))
        snippet = str(source.get("snippet", ""))
        context_lines.append(f"[来源{index}] 论文：{file_name} | 页码：{page_text} | 片段：{snippet}")
    return "\n".join(context_lines)


def enhance_answer_with_llm(
    result: AgentAnswer,
    client: DashScopeClient | None,
    model_name: str,
) -> tuple[AgentAnswer, str | None]:
    """在已有检索结果基础上，使用大模型增强回答文本。"""

    if client is None:
        return result, None
    if not result.sources:
        return result, None

    system_prompt = (
        "你是 CityScholar-Agent 的学术问答模块。"
        "你只能基于给定来源回答，不要编造来源中不存在的结论。"
        "回答必须包含简洁结论，并尽量保留来源编号引用。"
    )
    user_prompt = (
        f"用户问题：{result.question}\n\n"
        f"来源片段：\n{build_answer_context(result)}\n\n"
        "请输出中文回答，结构为：\n"
        "1) 直接回答\n"
        "2) 关键依据（用 [来源1] 这样的编号表示）\n"
        "3) 不确定性提示（如果有）"
    )

    try:
        llm_answer = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=900,
        )
    except Exception as exc:
        return result, f"问答大模型调用失败，已回退最小规则回答：{exc}"

    result.model_answer = llm_answer
    return result, None


def build_analysis_input_for_llm(full_text: str, max_chars: int = 18000) -> str:
    """构建用于结构化分析的大模型输入文本。"""

    text = full_text.strip()
    if not text:
        return text

    lowered = text.lower()
    reference_markers = ["\nreferences", "\nreference", "\nbibliography", "\nacknowledgments"]
    cut_positions: list[int] = []
    for marker in reference_markers:
        index = lowered.find(marker)
        if index != -1:
            cut_positions.append(index)
    if cut_positions:
        text = text[: min(cut_positions)].strip()

    if len(text) <= max_chars:
        return text

    head = text[: int(max_chars * 0.65)]
    tail = text[-int(max_chars * 0.35) :]
    return f"{head}\n\n[...中间内容已省略...]\n\n{tail}"


def pick_text_field(data: dict[str, Any], key: str, fallback: str) -> str:
    """从 JSON 结果中读取文本字段，不合法时回退。"""

    value = data.get(key, "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def normalize_evidence_map(data: dict[str, Any]) -> dict[str, list[str]]:
    """将模型输出的 evidence_map 规范为固定结构。"""

    field_keys = [
        "research_question",
        "research_object",
        "methods",
        "data_source",
        "key_findings",
        "limitations",
        "implications",
    ]
    evidence_map: dict[str, list[str]] = {}
    raw_map = data.get("evidence_map", {})
    if not isinstance(raw_map, dict):
        raw_map = {}

    for field_key in field_keys:
        raw_value = raw_map.get(field_key, [])
        if isinstance(raw_value, list):
            normalized_list = [str(item).strip() for item in raw_value if str(item).strip()]
        elif isinstance(raw_value, str):
            normalized_list = [raw_value.strip()] if raw_value.strip() else []
        else:
            normalized_list = []
        evidence_map[field_key] = normalized_list
    return evidence_map


def enhance_analysis_with_llm(
    agent: CityScholarAgent,
    target: str | None,
    client: DashScopeClient | None,
    model_name: str,
) -> tuple[PaperAnalysisResponse, str | None]:
    """执行单篇论文分析，并在可用时使用大模型增强结构化结果。"""

    base_result = agent.analyze_paper(target)
    if client is None or base_result.analysis is None:
        return base_result, None

    document = agent.find_document(target)
    if document is None:
        return base_result, None

    llm_input = build_analysis_input_for_llm(document.full_text)
    if not llm_input:
        return base_result, None

    system_prompt = (
        "你是城市研究领域的论文分析助手。"
        "请仅根据给定论文内容提取结构化结果，不要编造。"
        "输出必须是 JSON 对象。"
    )
    user_prompt = (
        "请对以下论文内容进行结构化提取，并严格输出 JSON。"
        "字段必须包含：\n"
        "research_question, research_object, methods, data_source, key_findings, limitations, implications, evidence_map\n"
        "其中 evidence_map 是对象，键为上述七个字段名，值为字符串数组（每项是依据片段）。\n\n"
        f"论文文件名：{document.file_name}\n"
        f"文档编号：{document.document_id}\n\n"
        f"论文内容：\n{llm_input}"
    )

    try:
        raw_text = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        return base_result, f"学术分析大模型调用失败，已回退规则分析：{exc}"

    json_data = parse_first_json_object(raw_text)
    if json_data is None:
        return base_result, "学术分析大模型输出非 JSON，已回退规则分析。"

    fallback_analysis = base_result.analysis
    merged_analysis = StructuredPaperAnalysis(
        file_name=fallback_analysis.file_name,
        document_id=fallback_analysis.document_id,
        research_question=pick_text_field(json_data, "research_question", fallback_analysis.research_question),
        research_object=pick_text_field(json_data, "research_object", fallback_analysis.research_object),
        methods=pick_text_field(json_data, "methods", fallback_analysis.methods),
        data_source=pick_text_field(json_data, "data_source", fallback_analysis.data_source),
        key_findings=pick_text_field(json_data, "key_findings", fallback_analysis.key_findings),
        limitations=pick_text_field(json_data, "limitations", fallback_analysis.limitations),
        implications=pick_text_field(json_data, "implications", fallback_analysis.implications),
        evidence_map=normalize_evidence_map(json_data),
    )

    enhanced_result = PaperAnalysisResponse(
        target=base_result.target,
        status_message=f"{base_result.status_message}（已使用大模型增强：{model_name}）",
        analysis=merged_analysis,
        formatted_output=format_analysis_result(merged_analysis),
    )
    return enhanced_result, None


def display_answer(result: AgentAnswer) -> None:
    """以命令行方式展示一次问答结果。"""

    print("\n模型回答：")
    print(result.model_answer)

    print("\n来源依据：")
    if not result.sources:
        print("当前没有可展示的来源依据。")
        return

    for index, source in enumerate(result.sources, start=1):
        print(f"{index}. 论文：{source['file_name']}")
        print(f"   页码：{format_page_numbers(source['page_numbers'])}")
        print(f"   片段编号：{source['chunk_id']}")
        print(f"   匹配词：{', '.join(source['matched_terms']) if source['matched_terms'] else '无'}")
        print(f"   相关分数：{source['score']}")
        print(f"   片段内容：{source['snippet']}")
        print(f"   文件路径：{source['source_path']}")


def display_analysis(result: PaperAnalysisResponse) -> None:
    """以命令行方式展示单篇论文结构化分析结果。"""

    print("\n学术分析：")
    print(result.status_message)
    print(result.formatted_output)


def display_paper_list(agent: CityScholarAgent) -> None:
    """展示当前可用论文列表。"""

    paper_items = agent.list_available_papers()
    print("\n当前可用论文：")
    if not paper_items:
        print("当前没有可用论文。")
        return

    for item in paper_items:
        print(
            f"{item['index']}. {item['file_name']} | 文档编号：{item['document_id']} | "
            f"页数：{item['total_pages']} | 字符数：{item['total_characters']}"
        )


def show_cli_help() -> None:
    """输出命令行帮助信息。"""

    print("\n命令说明：")
    print("- 直接输入问题：执行论文检索问答")
    print("- papers：查看论文列表")
    print("- analyze：分析第一篇论文")
    print("- analyze 1：分析第 1 篇论文")
    print("- analyze 关键词：按文件名或文档编号模糊匹配分析")
    print("- help：查看帮助")
    print("- exit：退出程序")


def parse_analyze_target(user_input: str) -> str | None:
    """从命令行输入中解析分析目标。"""

    parts = user_input.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None
    return parts[1].strip() or None


def run_cli_chat(
    agent: CityScholarAgent,
    llm_client: DashScopeClient | None,
    answer_model: str,
    analysis_model: str,
) -> None:
    """启动命令行交互循环。"""

    while True:
        try:
            user_input = input("\n请输入你的问题或命令：").strip()
        except EOFError:
            print("\n检测到输入结束，程序已退出。")
            break
        except KeyboardInterrupt:
            print("\n检测到手动中断，程序已退出。")
            break

        normalized_input = user_input.lower()
        if normalized_input in {"exit", "quit", "q", "退出", "结束"}:
            print("程序已退出，欢迎下次继续使用。")
            break

        if not user_input:
            print("请输入有效问题或命令，输入 help 查看说明。")
            continue

        if normalized_input in {"help", "帮助"}:
            show_cli_help()
            continue

        if normalized_input in {"papers", "list", "论文", "论文列表"}:
            try:
                display_paper_list(agent)
            except Exception as exc:
                print(f"读取论文列表失败：{exc}")
            continue

        if normalized_input.startswith("analyze") or user_input.startswith("分析"):
            analyze_target = parse_analyze_target(user_input)
            try:
                analysis_result, warning = enhance_analysis_with_llm(
                    agent=agent,
                    target=analyze_target,
                    client=llm_client,
                    model_name=analysis_model,
                )
            except Exception as exc:
                print(f"学术分析过程出现异常：{exc}")
                continue

            if warning:
                print(f"提示：{warning}")
            display_analysis(analysis_result)
            continue

        try:
            answer_result = agent.answer(user_input)
            answer_result, warning = enhance_answer_with_llm(answer_result, llm_client, answer_model)
        except ValueError as exc:
            print(f"输入有误：{exc}")
            continue
        except Exception as exc:
            print(f"问答过程出现异常：{exc}")
            continue

        if warning:
            print(f"提示：{warning}")
        display_answer(answer_result)


def main() -> None:
    """运行命令行主入口。"""

    config = get_app_config()
    ensure_directories(
        [
            config["data_dir"],
            config["raw_papers_dir"],
            config["processed_data_dir"],
            config["output_dir"],
        ]
    )

    llm_client = initialize_llm_client(config)
    answer_model = str(config.get("dashscope_answer_model", "qwen-plus"))
    analysis_model = str(config.get("dashscope_analysis_model", "qwen-max"))

    agent = CityScholarAgent(raw_papers_dir=config["raw_papers_dir"])
    try:
        knowledge_base = agent.build_knowledge_base()
    except ImportError as exc:
        print(f"依赖缺失：{exc}")
        return
    except Exception as exc:
        print(f"知识库构建失败：{exc}")
        return

    show_startup_info(config, knowledge_base, llm_client)
    show_parse_error_summary(knowledge_base.parse_errors)

    if not knowledge_base.pdf_files:
        print("当前未在 data/raw_papers/ 发现 PDF，暂时无法进入交互。")
        print("请先放入论文文件后重新运行。")
        return

    if not knowledge_base.documents:
        print("当前没有可用论文被成功解析，暂时无法进行问答或分析。")
        print("请检查 PDF 是否可被正常解析，或补充新的论文后再试。")
        return

    if not knowledge_base.chunk_records:
        print("提示：当前没有可检索片段，问答可能无法返回有效结果。")

    run_cli_chat(
        agent=agent,
        llm_client=llm_client,
        answer_model=answer_model,
        analysis_model=analysis_model,
    )


if __name__ == "__main__":
    main()
