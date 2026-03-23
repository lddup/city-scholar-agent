# 第 1 周：最小科研助教智能体

## 0. 本讲义作用

本讲义用于第 1 周课堂教学，目标是帮助学生理解什么是“围绕科研任务工作的最小智能体”，并结合 `CityScholar-Agent` 的当前代码跑通一个最小闭环示例。

---

## 1. 本周目标

### 1.1 教学目标

- 理解科研助教智能体与普通聊天机器人的区别。
- 理解最小智能体闭环的基本组成。
- 能够读懂当前项目中的核心入口代码。
- 能够在本地运行命令行程序并完成一次问答或分析。

### 1.2 本周完成后学生应该掌握什么

- 知道什么是“任务导向”的智能体。
- 知道最小系统为什么需要“输入、工具、知识、输出”四层。
- 能解释 `app.py`、`core/agent.py`、`core/prompts.py` 在系统中的作用。
- 能运行 `python app.py`，并完成 `papers`、`analyze`、提问三种基本操作。

---

## 2. 本周要回答的核心问题

1. 什么是科研助教智能体？
2. 为什么它不是一个普通 PDF 聊天器？
3. 最小智能体闭环包含哪些部分？
4. 当前项目是如何把论文库和问答流程串起来的？

---

## 3. 科研助教智能体的最小定义

在本课程中，我们把“最小科研助教智能体”定义为：

> 一个能够围绕本地论文知识库完成基础任务识别、论文片段检索、来源支撑回答与结构化分析的最小系统。

这里的关键点不是“会聊天”，而是：

- 它有任务目标。
- 它能调工具。
- 它依赖外部知识。
- 它输出的不只是自由文本，而是能服务科研任务的结果。

---

## 4. 第 1 周最小闭环

### 4.1 当前最小闭环流程

```mermaid
flowchart LR
    A[用户输入问题] --> B[app.py 接收输入]
    B --> C[core.agent 判定任务]
    C --> D[调用 rag 检索链路]
    D --> E[返回相关论文片段]
    E --> F[组织回答]
    F --> G[输出模型回答与来源依据]
```

### 4.2 对应到当前项目的真实流程

- 用户从命令行输入问题或命令。
- `app.py` 把输入交给 `CityScholarAgent`。
- `CityScholarAgent` 会先确认本地知识库是否已经构建。
- 如果问题是问答任务，就走检索与回答逻辑。
- 如果命令是 `analyze`，就走单篇结构化分析逻辑。
- 最后系统把结果打印到命令行。

---

## 5. 项目结构与本周相关文件

```text
CityScholar-Agent/
├─ app.py
├─ config.py
├─ core/
│  ├─ agent.py
│  └─ prompts.py
├─ rag/
│  ├─ loader.py
│  ├─ parser.py
│  ├─ splitter.py
│  └─ retriever.py
├─ tools/
│  └─ analyze_tool.py
└─ data/
   └─ raw_papers/
```

### 5.1 本周重点关注的文件

- `app.py`
- `core/agent.py`
- `core/prompts.py`

### 5.2 本周辅助理解的文件

- `rag/loader.py`
- `rag/parser.py`
- `rag/splitter.py`
- `rag/retriever.py`
- `tools/analyze_tool.py`

---

## 6. 核心代码讲解

### 6.1 `app.py` 的职责

`app.py` 是系统入口，它主要负责：

- 准备目录。
- 构建知识库。
- 打印启动信息。
- 接收用户输入。
- 把用户输入分发为问答任务或分析任务。

课堂上可以重点看这几个函数：

- `main()`
- `run_cli_chat()`
- `display_answer()`
- `display_analysis()`

示意代码：

```python
from config import get_app_config
from core.agent import CityScholarAgent


def main() -> None:
    config = get_app_config()
    agent = CityScholarAgent(raw_papers_dir=config["raw_papers_dir"])
    knowledge_base = agent.build_knowledge_base()
    run_cli_chat(agent)
```

讲解重点：

- `app.py` 不直接做复杂推理。
- 它只做“接收输入 -> 交给智能体 -> 展示输出”。
- 这体现了入口层和逻辑层分离。

### 6.2 `core/agent.py` 的职责

`core/agent.py` 是当前系统的核心协调者，负责：

- 构建知识库。
- 列出论文。
- 定位目标论文。
- 执行问答。
- 执行单篇结构化分析。

课堂上可以重点看：

- `build_knowledge_base()`
- `answer()`
- `analyze_paper()`

示意代码：

```python
class CityScholarAgent:
    def build_knowledge_base(self):
        pdf_paths = list_pdf_files(self.raw_papers_dir)
        documents, parse_errors = parse_pdf_files(pdf_paths)
        chunk_records = build_knowledge_base_records(documents)
        ...

    def answer(self, question: str):
        retrieved_chunks = retrieve_relevant_chunks(question, self.knowledge_base.chunk_records)
        model_answer = synthesize_answer(question, retrieved_chunks)
        ...
```

讲解重点：

- 这里体现了“智能体不是一个大函数”，而是多个模块的协调器。
- 它把数据层、检索层、输出层串联起来。
- 这就是最小智能体中的“任务调度”角色。

### 6.3 `core/prompts.py` 的职责

虽然当前版本还没有正式接入真实大模型，但 `core/prompts.py` 已经保留了提示词接口。

它的作用是：

- 集中管理回答原则。
- 统一兜底提示语。
- 为后续接入 LLM 保持接口稳定。

示意代码：

```python
def build_answer_system_prompt() -> str:
    return (
        "你是 CityScholar-Agent 的最小问答模块。"
        "请优先依据召回到的论文片段作答。"
    )
```

讲解重点：

- 即使当前没有真正调用大模型，也应该先把提示词层独立出来。
- 这样后续升级模型时，不需要重写整个系统结构。

---

## 7. 辅助模块如何支持最小智能体

### 7.1 `rag/loader.py`

作用：找到本地 `PDF` 文件。

### 7.2 `rag/parser.py`

作用：把 `PDF` 解析成统一文档对象。

### 7.3 `rag/splitter.py`

作用：把长文本切成更适合检索的片段。

### 7.4 `rag/retriever.py`

作用：根据用户问题召回最相关的文本块。

### 7.5 `tools/analyze_tool.py`

作用：对单篇论文做结构化分析。

课堂上要强调：

- `rag` 模块是知识处理链路。
- `tools` 模块是任务工具。
- `core/agent.py` 是把它们调起来的人。

---

## 8. 本周运行示例

### 8.1 启动程序

```bash
python app.py
```

### 8.2 查看当前论文列表

```text
papers
```

### 8.3 默认分析第一篇论文

```text
analyze
```

### 8.4 执行一个最小问答

```text
哪些论文关注城市安全或韧性？
```

### 8.5 退出程序

```text
exit
```

---

## 9. 课堂演示建议顺序

### 第一步：说明系统不是聊天机器人

可以先展示一句定义：

> CityScholar-Agent 不是“随便聊天”的系统，而是“围绕科研任务工作的最小智能体”。

### 第二步：展示最小架构图

用第 4 节的闭环图讲清楚系统最小组成。

### 第三步：现场运行 `python app.py`

让学生看到：

- 系统会扫描多少篇论文
- 能否成功建库
- 能否进入交互命令行

### 第四步：演示 `papers`

让学生看到系统确实掌握了论文对象，而不是空对话。

### 第五步：演示 `analyze`

强调：

- 智能体不仅能回答问题
- 还能调用工具完成结构化任务

### 第六步：演示问答

输入一个方向问题，让学生看到“模型回答 + 来源依据”的双层输出。

---

## 10. 本周课堂可讨论的局限性

为了让学生理解“最小系统不等于完美系统”，本周可以明确指出这些局限：

- 当前检索主要是关键词匹配，不是语义检索。
- 当前问答还不是正式大模型生成，而是基于片段整理。
- 当前单篇分析是规则式抽取，准确率还需要优化。
- 当前还没有多篇论文对比。
- 当前还没有综述提纲生成。
- 当前还没有 embedding 检索。

这一部分很重要，因为它自然引出后面几周的迭代主线。

---

## 11. 本周与后续周次的连接

### 第 1 周结束时，系统已经有了什么

- 最小命令行智能体入口。
- 本地论文建库能力。
- 最小检索问答能力。
- 单篇论文结构化分析原型。

### 第 2 周自然要补什么

- 工具化更明确。
- 多篇论文对比。
- 更规范的结构化输出。

### 第 4 周自然要补什么

- embedding。
- 向量索引。
- 更稳健的 RAG 检索。

---

## 12. 本周练习任务

### 练习 1

自己运行 `python app.py`，记录：

- 发现 PDF 数量
- 解析成功数量
- 片段总数
- 是否存在解析失败论文

### 练习 2

尝试输入 3 个不同问题，观察：

- 哪些问题能召回结果
- 哪些问题召回失败
- 中文问题和英文问题的效果是否一致

### 练习 3

运行 `analyze`，观察单篇分析结果中：

- 哪些字段比较可信
- 哪些字段明显不稳定
- 为什么规则抽取会误抓引文或参考文献

### 练习 4

阅读以下文件，并用自己的话总结每个文件的职责：

- `app.py`
- `core/agent.py`
- `core/prompts.py`
- `rag/retriever.py`

---

## 13. 本周小结

第 1 周的重点不是做出一个“很强”的系统，而是做出一个“能解释、能运行、能迭代”的最小智能体原型。

如果学生在本周真正理解了下面这句话，那么这一周的目标就达到了：

> 一个最小智能体，至少要能围绕任务目标，把输入、工具、知识和输出串成闭环。

---

## 14. 课后延伸思考

1. 为什么只做问答还不能算完整智能体？
2. 为什么“工具调用”是后续周次必须补上的能力？
3. 为什么当前系统虽然能跑，但检索质量还不够？
4. 为什么 embedding 检索属于后续迭代，而不是第一周就必须完成？

---

## 15. 建议的下一份讲义文件

下一周建议继续编写：

`notebooks/02_工具调用与模块化.md`

核心内容可以围绕：

- 什么是工具
- 工具与智能体的边界
- 多篇论文对比工具怎么设计
- 结构化输出如何服务科研任务
