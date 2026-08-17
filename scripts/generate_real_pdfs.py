"""生成真实中文技术文档 PDF 用于测试 — 3 份文档，共 23 页"""

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

pdfmetrics.registerFont(TTFont("CN", "C:/Windows/Fonts/msyh.ttc"))
pdfmetrics.registerFont(TTFont("CNB", "C:/Windows/Fonts/msyhbd.ttc"))

OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/tests/fixtures"

styles = getSampleStyleSheet()
for name, font, size, leading, color, space_after, space_before in [
    ("CNTitle", "CNB", 18, 26, "#1a1a2e", 12, 20),
    ("CNH1", "CNB", 14, 22, "#0f3460", 10, 16),
    ("CNH2", "CNB", 12, 18, "#2d3436", 8, 12),
    ("CNBody", "CN", 10, 16, "#2d3436", 6, 0),
    ("CNCode", "CN", 9, 14, "#636e72", 4, 4),
]:
    styles.add(
        ParagraphStyle(
            name,
            fontName=font,
            fontSize=size,
            leading=leading,
            textColor=color,
            spaceAfter=space_after,
            spaceBefore=space_before,
        )
    )


def H1(text):
    return Paragraph(text, styles["CNH1"])


def H2(text):
    return Paragraph(text, styles["CNH2"])


def P(text):
    return Paragraph(text, styles["CNBody"])


def C(text):
    return Paragraph(f'<font face="Courier" size="9">{text}</font>', styles["CNCode"])


def SP(mm_val=4):
    return Spacer(1, mm_val * mm)


def make_doc(filename, title, content_blocks):
    doc = SimpleDocTemplate(
        f"{OUT_DIR}/{filename}",
        pagesize=A4,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )
    story = [Paragraph(title, styles["CNTitle"]), SP(6)]
    for block in content_blocks:
        story.append(block)
        story.append(SP(2))
    doc.build(story)
    print(f"Generated: {filename}")


# ══════════════════════════════════════════════════════════════
# 文档 1: 企业知识库系统技术白皮书 (8 页)
# ══════════════════════════════════════════════════════════════
doc1 = [
    H1("一、系统概述"),
    P(
        "本系统是一套面向企业的智能知识库管理平台，采用检索增强生成（RAG）架构，结合语义搜索与知识图谱技术，实现对海量非结构化文档的智能解析、存储与精准问答。系统支持 PDF、Word、Markdown 等多种格式的文档导入，能够自动识别文档结构并构建语义索引。"
    ),
    P(
        "系统的核心设计目标包括：第一，提供亚秒级的文档检索响应时间；第二，支持百万级文档的横向扩展能力；第三，确保数据全链路不出企业内网，满足合规要求；第四，通过多轮自适应检索机制，将问答准确率维持在 85% 以上。"
    ),
    H1("二、技术架构"),
    H2("2.1 整体架构"),
    P(
        "系统采用分层微服务架构，自上而下分为接入层、业务层、引擎层和数据层四个逻辑层次。接入层负责 HTTP 请求的接收、认证、限流和路由分发，基于 FastAPI 框架构建，部署在 Uvicorn ASGI 服务器之上。业务层编排文档处理的完整流水线——从文档上传、格式检测、解析路由到分块策略的选择和知识图谱的增量更新。"
    ),
    P(
        "引擎层是系统的核心，包含六个独立的功能模块：文档解析引擎负责将不同格式的源文件转换为统一的中间表示（UIR）；语义分块引擎基于句子嵌入模型的相似度计算，在文档的主题边界处进行动态切分；向量嵌入引擎使用 BGE 中文预训练模型将文本片段编码为 512 维浮点向量；向量检索引擎基于余弦相似度实现高效的近似最近邻搜索；知识图谱引擎通过规则匹配与大语言模型协作完成实体识别和关系抽取；重排序引擎利用 Cross-Encoder 模型对初步检索结果进行精细排序。"
    ),
    P(
        "数据层采用异构存储架构：向量数据存储在 LanceDB 嵌入式数据库中（生产环境可切换为 Milvus 集群），知识图谱数据存储在 Neo4j 图数据库中，文档元数据存储在 PostgreSQL 关系型数据库中，缓存与会话数据存储在 Redis 内存数据库中。"
    ),
    H2("2.2 技术选型原则"),
    P(
        "每一项技术选型均遵循以下原则：第一，优先选择国产或开源方案，避免供应商锁定；第二，开发环境追求零配置与快速启动，生产环境追求高可用与水平扩展；第三，所有外部依赖必须具备至少一级降级策略；第四，优先选择在中文场景有充分验证的模型和工具。"
    ),
    PageBreak(),
    H1("三、文档解析引擎"),
    H2("3.1 多引擎协同解析架构"),
    P(
        "文档解析是知识库系统的基础环节，其质量直接影响后续分块、检索和问答的效果。系统设计了分层的三引擎解析架构，根据输入文档的类型自动路由到最优的解析器组合。"
    ),
    P(
        "对于原生 PDF 文档（即由 Word、LaTeX 等软件直接导出的 PDF），系统优先使用 PyMuPDF 引擎进行文本提取。PyMuPDF 基于 MuPDF C 语言库构建，在文本提取速度上比纯 Python 方案快 5 至 10 倍，且对中文字符的兼容性经过了充分验证。对于包含复杂表格的 PDF 文档，系统会同时启动 PDFPlumber 引擎进行表格区域的坐标级精确解析。PDFPlumber 通过读取 PDF 的底层绘图指令流，能够准确还原单元格的跨行跨列关系，其表格提取精度在标准测试集上达到 97% 以上。"
    ),
    P(
        "对于扫描件或图片型 PDF，系统调用 PaddleOCR 引擎进行光学字符识别。PaddleOCR 是百度飞桨团队开发的中文 OCR 引擎，在简体中文识别准确率上达到 98.7%，优于开源的 Tesseract 引擎（中文准确率约 88%），且支持离线部署。"
    ),
    H2("3.2 统一中间表示（UIR）"),
    P(
        "为了解决不同解析器输出格式不一致的问题，系统定义了一套统一中间表示（Unified Intermediate Representation, UIR）。无论底层使用哪种解析引擎，最终都会将解析结果标准化为 UIR 格式。UIR 的核心数据结构包括页面列表（pages）、文本块列表（text_blocks）、表格列表（tables）和图片列表（images）。每个文本块记录其所在的页面坐标、字体大小、是否为标题等元信息，为后续的语义分块提供结构线索。"
    ),
    PageBreak(),
    H1("四、语义分块策略"),
    H2("4.1 基于标题树与语义相似度的动态分块"),
    P(
        "传统的文档分块方法——如 LangChain 的 RecursiveCharacterTextSplitter——按照固定的字符数量进行切分，完全忽略了文档的语义结构。这种方式经常导致段落在中间被截断，甚至将一句话切成两半，严重影响后续的检索质量。"
    ),
    P(
        "本系统采用了一种结合文档结构感知与语义相似度的动态分块策略。首先，利用 PDF 解析阶段提取的字体大小和位置信息构建标题树，识别出文档的层级结构（一级标题、二级标题等）。每个文本块继承其所在的标题链，例如「第一章 > 1.1 系统概述 > 1.1.1 架构设计」。标题链提供了重要的上下文信息，使得检索时不仅能够匹配文本内容，还能利用文档结构进行精确导航。"
    ),
    P(
        "其次，系统使用 BGE 嵌入模型将每个候选句子编码为向量，计算相邻句子之间的余弦相似度。当相似度低于自适应阈值（当前窗口相似度均值减去 0.5 倍标准差）时，系统判定此处为主题切换点并进行切分。这种自适应阈值的设计使得分块策略能够适应不同密度和风格的文档——技术白皮书通常段落紧密、相似度高，而产品手册则频繁切换主题。"
    ),
    P(
        "分块的参数经过实验调优：最小令牌数为 100（低于此值的片段缺乏足够的上下文信息），最大令牌数为 800（接近 BGE 模型的最优输入窗口 512 令牌），硬切重叠率为 10%（避免关键信息出现在两个 chunk 的边缘而丢失）。"
    ),
    H2("4.2 分块质量评估"),
    P(
        "在包含 50 份中文技术文档的测试集上，语义分块策略相比固定大小分块策略在检索召回率（Recall@10）上提升约 12 个百分点，在答案准确率上提升约 8 个百分点。特别是在包含多级标题和混合内容的复杂文档中，结构感知带来的提升更为显著。"
    ),
    PageBreak(),
    H1("五、向量嵌入与检索"),
    H2("5.1 嵌入模型选择"),
    P(
        "系统选用北京智源人工智能研究院（BAAI）开发的 BGE-small-zh-v1.5 作为文本嵌入模型。BGE 系列模型在 C-MTEB（中文大规模文本嵌入基准评测）中综合排名前列，其中文语义区分度经过了大规模中文语料的充分训练。"
    ),
    P(
        "选择 512 维的 small 版本而非 768 维的 base 版本或 1024 维的 large 版本，基于以下工程考量：第一，精度差异在中文场景下通常不超过 3%，但 small 版本的模型体积仅为 100MB，base 版本为 400MB，large 版本超过 1GB；第二，CPU 推理延迟分别为 20ms、45ms 和 90ms 每条文 本，在无 GPU 的生产环境中，延迟差异将成倍放大；第三，512 维向量在 LanceDB 中存储和检索的开销显著低于更高维度。"
    ),
    P(
        "需要注意的是，BGE 模型在训练时使用了 passage: 和 query: 前缀来区分段落编码和查询编码。实测表明，如果不加此前缀，检索精度会下降约 5 个百分点。系统在 EmbeddingService 中自动为段落文本添加 passage: 前缀，为查询文本添加 query: 前缀。"
    ),
    H2("5.2 向量存储方案"),
    P(
        "在开发环境中，系统使用 LanceDB 作为嵌入式向量数据库。LanceDB 采用与 Apache Arrow 深度集成的列式存储格式，支持多版本并发控制（MVCC），读写操作不互锁。与广泛使用的 Chroma 相比，LanceDB 在 10 万级向量的检索场景中延迟低约 40%，且内存占用更少。"
    ),
    P(
        "生产环境中推荐将向量存储升级为 Milvus 分布式向量数据库。Milvus 支持 IVF（倒排文件索引）和 HNSW（分层可导航小世界图）等多种近似最近邻索引算法，具备水平扩展能力和 GPU 加速推理，并且原生集成了 Prometheus 监控端点。切换方案通过 VectorStore 的抽象接口实现，业务代码无需修改。"
    ),
    PageBreak(),
    H1("六、知识图谱构建"),
    H2("6.1 规则与 LLM 混合抽取"),
    P(
        "知识图谱是提升检索深度和广度的关键组件。系统采用规则匹配与大语言模型相结合的混合抽取策略。对于常见的计算机技术实体——包括编程语言、框架、数据库、协议、文件格式等——系统使用预定义的 TECH_PATTERNS 正则表达式集合进行抽取。规则抽取的准确率接近 100%，且零延迟、零 API 调用成本。"
    ),
    P(
        "对于非技术领域的实体——如公司名称、产品名称、法规编号、行业术语——系统调用大语言模型从上下文中进行抽取。LLM 同时也负责识别实体之间的关系。系统定义了 8 种标准关系类型：uses（使用）、depends_on（依赖）、contains（包含）、supplies（提供）、references（引用）、employs（雇佣）、owns（拥有）以及 signs（签署）。通过将关系类型限定在这 8 种以内，既覆盖了企业文档的常见关联模式，又避免了大语言模型生成不一致或不可控的关系类型。"
    ),
    H2("6.2 图谱检索与多跳推理"),
    P(
        "在问答环节，系统对用户问题中的实体进行识别，然后在知识图谱中执行多跳遍历查询。例如，用户提问「FastAPI 依赖哪些底层组件」，系统会首先定位 FastAPI 实体节点，然后沿 depends_on 关系进行多跳扩展，发现 FastAPI 依赖 Starlette 和 Pydantic，Starlette 又依赖 Uvicorn 和 anyio，最终形成完整的依赖链路返回给用户。"
    ),
    P(
        "多跳推理的深度限制为 3 跳，这是工程实践中在检索广度和响应延迟之间的平衡点。超过 3 跳的图遍历不仅导致延迟显著增加，而且引入的实体与原始问题的语义相关性也大幅下降。"
    ),
    PageBreak(),
    H1("七、混合检索与 Self-Retrieval"),
    H2("7.1 三路召回与精排"),
    P(
        "系统的检索流程分为召回和排序两个阶段。在召回阶段，系统同时启动三路检索：第一路为稠密向量检索，通过计算查询嵌入与文档嵌入的余弦相似度，从 LanceDB 中召回 Top-40 的候选文档；第二路为知识图谱检索，识别查询中的实体并在图谱中执行多跳遍历，召回相关的文档节点；第三路为关键词检索（基于 jieba 分词），用于补充向量检索可能遗漏的精确术语匹配。"
    ),
    P(
        "三路召回的结果按照 chunk_id 进行合并去重，然后进入排序阶段。排序阶段使用 BGE-Reranker-base 模型（Cross-Encoder 架构）对候选项进行精细排序。Cross-Encoder 与 Bi-Encoder 的核心区别在于：Bi-Encoder（如 BGE Embedding）分别编码查询和文档，然后计算余弦相似度；Cross-Encoder 则将查询和文档拼接在一起输入模型，让模型直接对两者之间的语义交互进行建模。实测表明，Cross-Encoder 相比 Bi-Encoder 的 MRR 指标提升约 5 至 15 个百分点。"
    ),
    H2("7.2 Self-Retrieval 自适应检索"),
    P(
        "传统 RAG 系统的一次性检索策略存在明显缺陷：如果首次检索的结果质量不高，系统没有纠正的机会。Self-Retrieval 机制通过在检索之后增加评估、改写和重检的闭环来解决这个问题。"
    ),
    P(
        "评估器从三个维度衡量检索质量：相关性（relevance，检索结果与查询的语义匹配度）、覆盖率（coverage，检索结果是否覆盖了问题的不同方面）和置信度（confidence，评分分布的集中程度）。当评估器判定检索质量不足时——例如 relevance < 0.5 或 coverage < 0.5——系统会根据评估反馈选择相应的改写策略：关键词扩展（keyword_expand）、同义词替换（synonym）、问题分解（decompose）或摘要调整（abstract_adjust）。改写后的查询重新进入检索流水线，形成多轮闭环。"
    ),
    P(
        "系统将最大重试轮数限制为 3 轮。工程实践表明，超过 3 轮的改写已难以带来实质性的检索质量提升，反而导致用户感知的响应延迟成倍增长。在典型的业务场景中，超过 80% 的查询在第一轮即可获得满意的检索结果，约 15% 的查询需要第二轮改写，仅约 5% 的查询需要进入第三轮。"
    ),
    PageBreak(),
    H1("八、部署与运维"),
    H2("8.1 部署拓扑"),
    P(
        "生产环境的推荐部署拓扑包含以下组件：至少 2 个 API 服务实例（通过 Nginx 反向代理进行负载均衡）、至少 2 个 Celery Worker 实例（负责异步文档处理）、1 个 Milvus 集群（3 节点，负责向量存储与检索）、1 个 Neo4j 单机实例（或集群，负责知识图谱存储）、1 个 Redis 实例（负责限流计数、缓存和 Celery 消息代理）、1 个 PostgreSQL 实例（负责文档元数据持久化）以及 1 个 MinIO 对象存储服务（负责原始文档持久化）。"
    ),
    H2("8.2 监控与告警"),
    P(
        "系统通过 Prometheus 采集关键性能指标，通过 Grafana 进行可视化展示。核心监控指标包括：QA 请求速率（按检索模式和成功/失败状态分组）、QA 响应延迟分布（P50/P95/P99）、LLM Token 消耗速率（按 prompt/completion 分组）、检索轮数分布（反映 Self-Retrieval 的触发频率）、向量库规模、知识图谱规模以及 LLM 调用错误分布（按错误类型分组）。"
    ),
    P(
        "告警规则覆盖以下场景：QA 错误率超过 5% 持续 5 分钟、QA P99 延迟超过 10 秒持续 5 分钟、LLM API 调用成功率低于 95% 持续 5 分钟、向量库可用存储低于 80% 以及 Celery Worker 存活数低于 2。"
    ),
    PageBreak(),
    H1("九、性能基准"),
    P(
        "以下数据基于 16GB RAM、8 核 CPU、无 GPU 的开发环境实测（测试文档集：100 份中文技术 PDF，平均每份 8 页，生成约 4500 个 chunks）："
    ),
    P(
        "文档解析吞吐量为每页 0.3 秒（PyMuPDF 原生 PDF）至 2.1 秒（PaddleOCR 扫描件）；语义分块速度为每 1000 字符 0.05 秒；向量嵌入速度为每 chunk 0.02 秒（CPU 推理）；向量检索延迟为 12ms（千级向量）至 180ms（十万级向量）；QA 端到端延迟中位数为 1.2 秒，P99 为 3.8 秒；单次 QA 的 LLM Token 消耗均值约 1200 tokens（其中 prompt 约 900 tokens，completion 约 300 tokens）。"
    ),
]

# ══════════════════════════════════════════════════════════════
# 文档 2: Python Web 框架技术选型报告 (8 页)
# ══════════════════════════════════════════════════════════════
doc2 = [
    H1("一、背景与目标"),
    P(
        "本文档旨在为技术团队提供 Python Web 框架的选型参考。随着团队业务从单体应用向微服务架构演进，现有的 Django 单体架构在开发效率、异步性能和部署灵活性方面逐渐暴露出瓶颈。本次技术选型的目标是选择一套能够支撑未来 3 至 5 年业务增长的 Web 框架，同时兼顾团队现有的 Python 技术栈优势。"
    ),
    P(
        "评估维度涵盖以下六个方面：性能基准（吞吐量、响应延迟、并发承载能力）、开发效率（代码量、文档质量、生态丰富度）、异步支持（原生异步能力、WebSocket 支持、后台任务处理）、可维护性（代码组织方式、测试友好度、类型安全）、社区活跃度（GitHub Star 数、Issue 响应速度、版本发布频率）以及部署运维（容器化友好度、监控集成、配置管理）。"
    ),
    H1("二、候选框架概览"),
    H2("2.1 FastAPI"),
    P(
        "FastAPI 是由 Sebastián Ramírez 于 2018 年开发的现代 Python Web 框架，基于 Starlette 作为 ASGI 底层引擎，使用 Pydantic 进行数据校验和序列化。FastAPI 的核心卖点是其极致的开发体验——通过 Python 类型注解自动生成 OpenAPI 文档、请求参数校验和编辑器自动补全。自发布以来，FastAPI 在 GitHub 上的 Star 数增长迅速，截至 2026 年已超过 80,000 Star，成为 Python Web 框架中增长最快的项目。"
    ),
    P(
        "FastAPI 的性能表现优秀。根据 TechEmpower 的独立基准测试，在 JSON 序列化场景下，FastAPI 的吞吐量约为 Django 的 3.5 倍、Flask 的 2.8 倍。这主要归功于其异步优先的设计理念和 Starlette 引擎的高效实现。在实际业务中，一个典型的 CRUD 接口在 FastAPI 上可以达到 5000 至 8000 QPS（单进程，Uvicorn Worker），而同等硬件条件下的 Django + Gunicorn 仅为 1500 至 2500 QPS。"
    ),
    H2("2.2 Django"),
    P(
        "Django 是 Python 生态中最成熟的全栈 Web 框架，自 2005 年发布以来已积累了近 20 年的社区经验。Django 采用「包含一切」（Batteries Included）的设计哲学，内置了 ORM、模板引擎、表单处理、认证系统、管理后台、国际化和安全防护等几乎所有的 Web 开发组件。"
    ),
    P(
        "Django 的主要优势在于其成熟度和完整性。对于标准的 CRUD 业务系统，Django 可以在一周内完成从零到上线的全流程开发。Django Admin 自动生成的后台管理界面更是被许多项目直接用作内部运营系统。然而，Django 的异步支持直到 3.0 版本才开始引入，至今仍不够成熟。Django ORM 的异步查询接口在 4.2 版本才达到基本可用状态，生态中的大量第三方包仍然只支持同步模式。"
    ),
    PageBreak(),
    H2("2.3 Flask"),
    P(
        "Flask 是一个轻量级的 Web 微框架，由 Armin Ronacher 于 2010 年创建。Flask 的设计哲学是「最小核心 + 丰富插件」——框架本身只提供路由、请求上下文和模板渲染等最基础的能力，其他功能通过社区扩展实现。"
    ),
    P(
        "Flask 的优势在于灵活性和简洁性。对于小型项目、API 网关和微服务原型，Flask 的轻量级特征使其启动极快（通常不到 0.1 秒），代码量少（一个典型的 REST API 端点仅需 5 至 10 行代码），学习曲线平缓。然而，Flask 的灵活性也带来了明显的代价：项目规模增长后，插件选择和组合成为技术债务的主要来源。不同的 Flask 项目可能使用完全不同的 ORM（SQLAlchemy、Peewee、PonyORM）、序列化方案（Marshmallow、Pydantic、Cerberus）和项目结构，这给团队协作和代码交接带来持续的认知成本。"
    ),
    H2("2.4 Sanic 与 Tornado"),
    P(
        "Sanic 是一个类 Flask 语法的异步 Web 框架，主打极高性能。Sanic 在 TechEmpower 基准测试中的吞吐量略高于 FastAPI（约高 10% 至 15%），但生态丰富度和社区规模远小于 FastAPI。对于大多数业务场景，15% 的吞吐量差异并非瓶颈，而较弱的生态意味着需要自行实现更多的业务组件。"
    ),
    P(
        "Tornado 是最早的 Python 异步 Web 框架之一，由 Facebook（现 Meta）于 2009 年开源。Tornado 的长连接和 WebSocket 支持在实时通信场景中经过了充分的验证。然而，Tornado 的异步模型基于回调而非 async/await 协程，与现代 Python 异步编程范式存在较大差异，学习曲线较陡。此外，Tornado 的社区活跃度已明显下降。"
    ),
    PageBreak(),
    H1("三、综合对比"),
    H2("3.1 性能基准测试"),
    P(
        "测试环境：Intel Core i7-13700H（14 核 20 线程），32GB DDR5 RAM，Ubuntu 22.04 LTS，Python 3.12。各框架均使用推荐的 ASGI/WSGI 服务器，使用 wrk 工具以 100 并发连接持续 30 秒压测。"
    ),
    P(
        "JSON 序列化场景（返回 1KB JSON）：FastAPI + Uvicorn 为 7200 QPS，Sanic 为 8100 QPS，Flask + Gunicorn（gevent）为 2600 QPS，Django + Gunicorn（sync）为 1900 QPS。"
    ),
    P(
        "数据库读取场景（单表查询，50KB 结果集）：FastAPI + SQLAlchemy async 为 3200 QPS，Django ORM sync 为 1100 QPS，Django ORM async 为 2400 QPS。"
    ),
    P(
        "WebSocket 并发连接（消息广播）：FastAPI + Uvicorn 稳定支持 15,000 并发连接，Django Channels 约 8,000，Tornado 约 20,000。"
    ),
    H2("3.2 开发效率对比"),
    P(
        "以开发一个包含 10 个 REST 端点的标准 CRUD 微服务为例：FastAPI 约需 150 至 200 行代码（含 Pydantic Schema 定义），Django + DRF 约需 300 至 400 行代码（含 Serializer 和 ViewSet），Flask 约需 250 至 350 行代码（取决于使用的扩展组合）。FastAPI 在代码量上具有明显优势，这主要得益于 Pydantic 的类型驱动序列化——同一个 Model 定义同时服务于数据校验、序列化、反序列化和 API 文档生成。"
    ),
    H2("3.3 生态与社区"),
    P(
        "Django 拥有 Python Web 框架中最庞大的第三方包生态（PyPI 上约 5000 个 django-* 包），这意味着几乎任何常见的业务需求都有现成的解决方案。FastAPI 的生态虽然规模较小（约 800 个 fastapi-* 或专用于 FastAPI 的包），但其增长速度快，并且由于 FastAPI 的设计与 Starlette 和 Pydantic 深度整合，许多通用的 ASGI 中间件和 Pydantic 扩展可以直接使用。"
    ),
    PageBreak(),
    H1("四、选型建议"),
    H2("4.1 推荐方案"),
    P(
        "综合以上评估，技术团队的推荐方案为：以 FastAPI 作为主力 Web 框架，搭配 SQLAlchemy 2.0 异步 ORM 和 Alembic 数据库迁移工具，使用 Pydantic Settings 进行配置管理，使用 Uvicorn 作为生产级 ASGI 服务器，通过 Docker 进行容器化部署，通过 Prometheus + Grafana 进行性能监控。"
    ),
    P(
        "该方案适用于以下场景：新启动的微服务项目、需要高性能异步处理的数据接口、前后端分离架构下的 REST API 服务、需要 WebSocket 实时通信的应用、以及需要自动生成 OpenAPI 文档的团队协作场景。"
    ),
    H2("4.2 迁移策略"),
    P(
        "对于现有的 Django 单体应用，建议采取「绞杀者模式」（Strangler Fig Pattern）进行渐进式迁移。具体步骤为：第一步，在 Nginx 层面将新功能的路由指向 FastAPI 服务，旧功能继续由 Django 处理；第二步，逐个模块地将 Django 的逻辑抽取为独立的 FastAPI 微服务，同时保持 API 契约的向后兼容；第三步，当所有业务逻辑完成迁移后，将 Django 实例下线。整个迁移过程预计持续 3 至 6 个月，期间新旧服务通过 HTTP 协议进行通信，通过 API 网关统一对外暴露。"
    ),
    PageBreak(),
    H1("五、深度学习框架技术选型"),
    H2("5.1 PyTorch"),
    P(
        "PyTorch 由 Meta（原 Facebook）的人工智能研究团队于 2016 年发布，是目前学术界和工业界使用最广泛的深度学习框架。PyTorch 的核心优势在于其动态计算图（Define-by-Run）的设计——计算图在运行时动态构建，这使得调试体验与普通的 Python 程序完全一致。开发者可以随时使用 Python 的 print 语句或 pdb 调试器检查中间张量的值，而不需要在静态图中插入额外的调试节点。"
    ),
    P(
        "自 2.0 版本起，PyTorch 引入了 torch.compile 功能，通过 JIT（即时编译）将动态图优化为静态图，在保持开发体验的同时大幅提升了推理性能。此外，PyTorch 的 torch.distributed 模块提供了完善的数据并行和模型并行训练能力，支持从单机单卡到数千张 GPU 集群的无缝扩展。"
    ),
    H2("5.2 TensorFlow 与 PaddlePaddle"),
    P(
        "TensorFlow 由 Google 于 2015 年发布，曾是深度学习框架的绝对霸主。但自 2020 年以来，随着 PyTorch 在学术界的全面超越和工业界的快速追赶，TensorFlow 的市场份额持续下滑。不过，TensorFlow 在移动端推理（TensorFlow Lite）和生产级模型服务（TensorFlow Serving）方面仍保持优势。"
    ),
    P(
        "PaddlePaddle（飞桨）是百度开发的中国首个开源深度学习平台，在中文 NLP 和 OCR 领域具有明显的本地化优势。PaddleOCR 是目前中文 OCR 识别准确率最高的开源方案之一，预训练模型覆盖了简体中文、繁体中文、英文、日文、韩文等 80 多种语言。PaddleNLP 则提供了完整的中文预训练模型家族，包括 ERNIE 系列模型。"
    ),
    H2("5.3 嵌入模型生态"),
    P(
        "在中文文本嵌入领域，BAAI 的 BGE 系列和 text2vec 系列是应用最广泛的两套预训练模型。BGE 系列由北京智源人工智能研究院开发，在 C-MTEB 基准上综合排名领先。BGE-large-zh-v1.5（1024 维）是精度最高的中文嵌入模型之一，BGE-small-zh-v1.5（512 维）则在精度和效率之间取得了良好的平衡。text2vec-large-chinese 提供 1024 维嵌入，在特定领域（如金融、法律）的微调效果优于 BGE，但模型体积大、推理速度慢。两项工程实践建议：第一，嵌入模型应加 passage: 和 query: 前缀以匹配训练时的输入格式；第二，向量必须进行 L2 归一化以保证余弦相似度的正确计算。"
    ),
]

# ══════════════════════════════════════════════════════════════
# 文档 3: 知识库系统部署与运维手册 (7 页)
# ══════════════════════════════════════════════════════════════
doc3 = [
    H1("一、环境要求"),
    P(
        "生产环境的最低硬件配置：CPU 不少于 8 核（推荐 16 核以上），内存不少于 32GB（推荐 64GB 以上），可用磁盘空间不少于 500GB SSD（推荐 NVMe SSD），网络带宽不少于 100Mbps。软件环境要求：操作系统为 Ubuntu 22.04 LTS 或 Rocky Linux 9.x；容器运行时为 Docker Engine 24.x+ 及 Docker Compose v2.x+；Python 版本为 3.12+（仅在本地开发部署时需要）；数据库外部依赖包括 PostgreSQL 16+、Redis 7+ 和 Neo4j 5.x Community Edition。"
    ),
    H1("二、Docker Compose 部署"),
    H2("2.1 服务拓扑"),
    P(
        "项目根目录下的 docker-compose.yml 文件定义了完整的生产级服务拓扑，包含 8 个核心服务组件：api 服务负责接收和处理所有 HTTP 请求，对外暴露 8000 端口；worker 服务基于 Celery 实现异步任务队列，专门处理文档解析等耗时操作，配置了 4 个并发 Worker 进程；milvus-standalone 服务使用 Milvus 2.4.0 镜像作为分布式向量数据库，对外暴露 19530（gRPC）和 9091（Metrics）端口；etcd 服务作为 Milvus 的元数据协调中心，使用 3.5.5 版本；minio 服务提供 S3 兼容的对象存储，同时为 Milvus 和文档存储提供后端，对外暴露 9000（API）和 9001（Console）端口；neo4j 服务使用 5.x Community 版本作为知识图谱数据库，对外暴露 7474（HTTP）和 7687（Bolt）端口；redis 服务使用 7.x Alpine 轻量版本，同时充当 Celery 消息代理、限流计数器和 QA 缓存后端；postgres 服务使用 pgvector/pg16 镜像，集成了向量扩展的 PostgreSQL 16 数据库。"
    ),
    PageBreak(),
    H2("2.2 启动与验证"),
    P(
        "首次部署的完整步骤：第一步，克隆项目代码仓库并切换到目标版本标签；第二步，从 .env.example 复制环境配置文件并填写必要的 API Key 和访问密钥；第三步，执行 docker-compose up -d 命令启动所有服务，首次启动会自动拉取所需的 Docker 镜像；第四步，等待约 60 秒后执行 docker-compose ps 确认所有服务状态为 healthy；第五步，访问 http://localhost:8000/health 验证 API 服务是否正常响应；第六步，访问 http://localhost:8000/docs 查看自动生成的 OpenAPI 文档。"
    ),
    H2("2.3 已知问题与解决方案"),
    P(
        "问题一：Docker Desktop for Windows 版本 v29.5.3 存在引擎响应超时的 Bug，表现为 docker-compose up 命令执行后长时间无响应，超时后报出 context deadline exceeded 错误。解决方案为卸载当前版本并降级安装 Docker Desktop v27.x，或者在 Windows 上改用 Podman Desktop 作为替代的容器运行时。"
    ),
    P(
        "问题二：Milvus 依赖的 etcd 容器在首次启动时因缺少显式的 listen-client-urls 配置而 CrashLoopBackoff。解决方案已在 docker-compose.yml 中应用——为 etcd 服务添加了 command 指令，显式设置 advertise-client-urls 和 listen-client-urls 参数。"
    ),
    P(
        "问题三：MinIO 在较新版本中强制要求设置 root user 和 root password 环境变量，旧版 docker-compose 配置仅设置了 access key 和 secret key 而遗漏了这两个必要变量，导致 MinIO 容器启动后立即退出。解决方案已在 docker-compose.yml 中补全 MINIO_ROOT_USER 和 MINIO_ROOT_PASSWORD 环境变量。"
    ),
    PageBreak(),
    H1("三、性能调优"),
    H2("3.1 Uvicorn Worker 配置"),
    P(
        "Uvicorn 的生产部署建议使用多 Worker 模式：对于纯 CPU 密集型负载（如嵌入推理），Worker 数不应超过 CPU 物理核心数；对于 IO 密集型负载（如 API 请求处理），Worker 数可以设置为 CPU 核心数的 2 至 4 倍。推荐的启动命令为：uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop --http httptools。其中 uvloop 是 libuv 的 Python 绑定，相比默认的 asyncio 事件循环吞吐量提升约 20% 至 30%；httptools 是 C 语言实现的 HTTP 解析器，减少了解析开销。"
    ),
    H2("3.2 LanceDB 索引优化"),
    P(
        "当向量数量超过 10,000 条时，建议为 LanceDB 创建 IVF_PQ 索引以替代默认的暴力搜索。索引参数建议为 num_partitions=sqrt(N)（其中 N 为向量总数）、num_sub_vectors=32（PQ 编码的子向量数，512/32=16 维每个子向量）。索引构建操作较为耗时，建议在离线时段批量执行，或在新文档处理流水线中增量更新。"
    ),
    H2("3.3 LLM 调用优化"),
    P(
        "减少 LLM Token 消耗的几项实用策略：第一，启用 QA 缓存——对于重复出现的高频问题，将 LLM 生成的答案缓存到 Redis 中，Key 为问题的 MD5 哈希与 top_k 参数的组合，TTL 设为 1 小时；第二，优化 Prompt 设计——将系统提示词从当前的约 200 个中文字符精简到约 80 个，通过减少 System Prompt 的长度降低每次调用的 Token 开销；第三，使用 LLM 熔断器——当连续 3 次调用失败时自动打开熔断器，后续请求在 30 秒内直接返回降级答案而不经过 LLM，避免在故障期间持续消耗 API 配额。"
    ),
    PageBreak(),
    H1("四、备份与恢复"),
    H2("4.1 数据备份策略"),
    P(
        "生产环境需要定期备份以下四类数据：PostgreSQL 中的文档元数据和 QA 日志，建议使用 pg_dump 进行每日全量备份，保留最近 30 天的历史版本；LanceDB 中的向量数据，建议通过文件系统层面的目录复制进行每周全量备份（因为 LanceDB 的列式存储格式天然支持文件级别的快照）；Neo4j 中的知识图谱数据，建议使用 neo4j-admin dump 命令进行每日增量备份；MinIO 中的原始文档文件，建议配置 Bucket 级别的版本控制和跨区域复制。"
    ),
    H2("4.2 灾难恢复流程"),
    P(
        "恢复流程按优先级排序：第一步，恢复 PostgreSQL 数据库——通过 pg_restore 将最近的备份文件还原到新的 PostgreSQL 实例，验证文档元数据的完整性（总数、时间范围）；第二步，恢复 MinIO 对象存储——将备份的 Bucket 数据同步到新的 MinIO 实例，使用 mc mirror 命令；第三步，恢复 Neo4j 图数据库——使用 neo4j-admin load 命令从备份文件重建图数据库；第四步，重建 LanceDB 向量索引——由于向量可以从源文档重新生成，LanceDB 数据的恢复优先级最低。如果原始文档完好，只需重新运行文档处理流水线即可重新生成所有向量。"
    ),
    PageBreak(),
    H1("五、安全配置"),
    H2("5.1 网络安全"),
    P(
        "生产环境必须限制各服务的网络暴露面。仅 API 服务的 8000 端口需要对用户网络可达；Milvus、Redis、PostgreSQL、Neo4j 和 MinIO 的内部端口不应直接暴露在公网上，仅允许 Docker 内部网络访问。如果必须从外部访问这些管理端口，应通过 SSH 隧道或 VPN 进行访问。"
    ),
    H2("5.2 认证与授权"),
    P(
        "API 层应启用 API Key 认证中间件。设置环境变量 RAG_API_KEY_ENABLED=true 激活认证，设置 RAG_API_KEY 为不少于 32 个字符的随机字符串。API Key 支持两种传递方式：通过 X-API-Key 自定义请求头传递，或通过 Authorization: Bearer <key> 标准头传递。健康检查端点 /health 和 API 文档端点 /docs 自动豁免认证检查。"
    ),
    P(
        "敏感信息管理规则：LLM API Key、数据库密码、Redis 密码等必须通过环境变量或 Docker Secrets 注入，严禁硬编码在源代码或配置文件中提交到版本控制系统。所有密钥应定期轮换，最少每季度轮换一次。"
    ),
    PageBreak(),
    H1("六、监控指标参考"),
    P(
        "以下是生产环境正常运行时的各项关键指标的参考范围：QA 请求错误率应低于 1%（若超过 5% 持续 5 分钟触发告警）；QA 接口 P50 延迟应低于 2 秒，P99 延迟应低于 10 秒；LLM API 调用成功率应高于 98%；单次 QA 请求的平均 Token 消耗应控制在 1500 tokens 以下（Prompt + Completion 合计）；Celery Worker 的队列积压任务数不应超过 50；文档处理（含解析、分块、嵌入和图谱构建）的端到端延迟对于 10 页 PDF 应在 30 秒以内完成；向量库的可用磁盘空间应始终保留 20% 以上的余量；知识图谱的节点数和关系数应随文档上传量近似线性增长（异常偏离可能说明实体抽取出现了问题）。"
    ),
]

# ── 生成 ──
os.makedirs(OUT_DIR, exist_ok=True)
make_doc("企业知识库系统技术白皮书.pdf", "企业知识库系统技术白皮书", doc1)
make_doc("Python_Web框架技术选型报告.pdf", "Python Web框架技术选型报告", doc2)
make_doc("知识库系统部署与运维手册.pdf", "知识库系统部署与运维手册", doc3)
print(f"\nDone — 3 PDFs generated in {OUT_DIR}")
