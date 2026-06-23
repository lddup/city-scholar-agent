"""本模块作用：提供最小智能体安全检查，用于拦截明显的提示注入与敏感信息请求。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SafetyCheckResult:
    """本数据结构作用：保存一次用户输入安全检查结果。"""

    allowed: bool
    risk_level: str
    categories: list[str] = field(default_factory=list)
    matched_rules: list[str] = field(default_factory=list)
    message: str = ""
    suggestion: str = "请围绕本地论文内容提出科研问题，或使用 analyze、compare、outline、workflow 等科研任务命令。"


SAFETY_RULES: list[dict[str, object]] = [
    {
        "category": "prompt_injection",
        "label": "提示注入",
        "patterns": [
            r"忽略.*(规则|指令|提示|限制)",
            r"无视.*(规则|指令|提示|限制)",
            r"绕过.*(规则|限制|安全)",
            r"ignore\s+(all\s+)?(previous|prior).*(instruction|prompt|rule)",
            r"disregard\s+(all\s+)?(previous|prior).*(instruction|prompt|rule)",
            r"you\s+are\s+now\s+(not|no\s+longer)",
            r"developer\s+mode|jailbreak",
        ],
    },
    {
        "category": "secret_extraction",
        "label": "敏感密钥读取",
        "patterns": [
            r"(输出|打印|显示|告诉我|泄露).*(api\s*key|apikey|密钥|token|令牌)",
            r"(das[h]?scope_api_key|dashscope_api_key|api_key|secret|access[_-]?key)",
            r"(环境变量|env).*(api|key|token|密钥)",
        ],
    },
    {
        "category": "system_prompt_leak",
        "label": "系统提示词窃取",
        "patterns": [
            r"(输出|打印|显示|告诉我|泄露).*(系统提示词|system\s*prompt|prompt)",
            r"(你的|当前).*(系统提示词|system\s*prompt|内部提示)",
            r"(show|print|reveal).*(system\s*prompt|hidden\s*prompt|developer\s*message)",
        ],
    },
    {
        "category": "unauthorized_file_access",
        "label": "越权文件读取",
        "patterns": [
            r"(读取|打开|显示).*(\.env|\.git|ssh|id_rsa|private\s*key)",
            r"(读取|打开|显示).*(c:\\users|/etc/passwd|用户目录)",
            r"(read|open|show).*(\.env|\.git|id_rsa|private\s*key|/etc/passwd)",
        ],
    },
    {
        "category": "dangerous_command",
        "label": "危险命令诱导",
        "patterns": [
            r"(删除|清空|格式化).*(文件|目录|磁盘|仓库)",
            r"(执行|运行).*(rm\s+-rf|del\s+/|format\s+|git\s+reset\s+--hard)",
            r"(run|execute).*(rm\s+-rf|format\s+|git\s+reset\s+--hard)",
        ],
    },
]


def normalize_safety_text(text: str) -> str:
    """归一化待检测文本。"""

    return re.sub(r"\s+", " ", text.strip().lower())


def check_user_input_safety(user_input: str) -> SafetyCheckResult:
    """检查用户输入是否包含明显的高风险攻击意图。"""

    normalized_text = normalize_safety_text(user_input)
    if not normalized_text:
        return SafetyCheckResult(
            allowed=True,
            risk_level="low",
            message="未检测到风险。",
        )

    categories: list[str] = []
    matched_rules: list[str] = []
    for rule in SAFETY_RULES:
        category = str(rule["category"])
        label = str(rule["label"])
        patterns = rule["patterns"]
        if not isinstance(patterns, list):
            continue

        for pattern in patterns:
            if re.search(str(pattern), normalized_text, flags=re.IGNORECASE):
                if category not in categories:
                    categories.append(category)
                if label not in matched_rules:
                    matched_rules.append(label)
                break

    if not categories:
        return SafetyCheckResult(
            allowed=True,
            risk_level="low",
            message="未检测到明显安全风险。",
        )

    return SafetyCheckResult(
        allowed=False,
        risk_level="high",
        categories=categories,
        matched_rules=matched_rules,
        message="检测到高风险请求，已在 Agent 安全层拦截，未进入论文检索或大模型问答。",
    )


def format_safety_result(result: SafetyCheckResult) -> str:
    """将安全检查结果整理为命令行展示文本。"""

    lines = [
        "安全检查：",
        f"是否允许继续：{'是' if result.allowed else '否'}",
        f"风险等级：{result.risk_level}",
        f"命中类型：{', '.join(result.categories) if result.categories else '无'}",
        f"命中规则：{', '.join(result.matched_rules) if result.matched_rules else '无'}",
        f"说明：{result.message}",
        f"建议：{result.suggestion}",
    ]
    return "\n".join(lines)
