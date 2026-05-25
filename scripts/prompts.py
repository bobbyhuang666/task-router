"""
提示模板 — 13+ 种任务类型的优化 prompt 模板
"""

TOOL_PREFIX = "只输出结果，不要解释。\n\n"

# ─── 提示模板 ──────────────────────────────────────────────────

PROMPT_TEMPLATES: dict[str, dict] = {
    "general_classify": {
        "detect": ["分类", "归类", "分组", "产品类别", "按类别", "分为"],
        "template": TOOL_PREFIX + """将以下内容按类别分组，保持数据完整。每种类别一行，列出属于该类别的所有项：

示例：
iPhone15, 华为Mate60, 小米14, MacBookPro, 联想小新
→ 手机：iPhone15, 华为Mate60, 小米14
电脑：MacBookPro, 联想小新

现在分类：
{text}
→""",
        "max_tokens": 256,
    },

    "file_classify": {
        "detect": ["扩展名", "按类型", "按格式"],
        "template": TOOL_PREFIX + """把以下每个文件按扩展名分到对应类别，**必须列出所有文件**，每个一行。

类别对应关系：
.pdf → 文档  .txt → 文档  .docx → 文档
.jpg → 图片  .png → 图片  .gif → 图片
.py → 代码  .js → 代码  .css → 代码  .html → 代码
.csv → 数据  .xlsx → 数据  .json → 数据
.mp3 → 音频  .mp4 → 视频

示例：
report.pdf, photo.jpg, main.py, data.csv, song.mp3
→ 文档：report.pdf
图片：photo.jpg
代码：main.py
数据：data.csv
音频：song.mp3

现在分类：
{text}
→""",
        "max_tokens": 256,
    },

    "sentiment": {
        "detect": ["情感", "情绪", "正面", "负面", "好评", "差评", "正负面"],
        "template": TOOL_PREFIX + """判断情感倾向，只输出"正面"或"负面"：

示例：
商品很好 → 正面
快递太慢 → 负面
服务不错 → 正面
质量很差 → 负面

现在判断：
{text}
→""",
        "max_tokens": 32,
    },

    "translate_en2zh": {
        "detect": ["翻译成中文", "译成中文", "转成中文", "翻译为中文"],
        "template": TOOL_PREFIX + """将英文翻译成中文，只输出翻译结果：

示例：
Hello world → 你好世界
Good morning → 早上好

现在翻译：
{text}
→""",
        "max_tokens": 256,
    },

    "translate_zh2en": {
        "detect": ["翻译成英文", "译成英文", "转成英文", "翻译为英文"],
        "template": TOOL_PREFIX + """将中文翻译成英文，只输出翻译结果：

示例：
你好世界 → Hello world
早上好 → Good morning

现在翻译：
{text}
→""",
        "max_tokens": 256,
    },

    "extract_keywords": {
        "detect": ["关键词", "keyword", "关键字"],
        "template": TOOL_PREFIX + """从以下文本中提取最重要的3-5个关键词，用逗号分隔：

示例：
"今天天气真好，适合出去散步" → 天气,散步,户外
"这个手机续航长、拍照清晰、性价比高" → 手机,续航,拍照,性价比

现在提取：
{text}
→""",
        "max_tokens": 64,
    },

    "extract_info": {
        "detect": ["提取", "摘录", "找出", "抽取出"],
        "template": TOOL_PREFIX + """从文本中提取指定信息，每行一个：

示例：
文本：请联系张三(13800138000)或发邮件到 admin@test.com
提取：姓名、电话、邮箱
→ 张三
13800138000
admin@test.com

现在提取：
文本：{text}
提取：{target}
→""",
        "max_tokens": 128,
    },

    "summarize_short": {
        "detect": ["总结", "概括", "摘要", "要点"],
        "template": TOOL_PREFIX + """用1-3句话概括以下内容：

示例：
"苹果今天发布了新款iPhone，搭载A18芯片，起售价799美元。"
→ 苹果发布新款iPhone，搭载A18芯片，起售价799美元。

现在概括：
{text}
→""",
        "max_tokens": 256,
    },

    "format_json": {
        "detect": ["转成json", "格式化成json", "json格式"],
        "template": TOOL_PREFIX + """将以下内容转成JSON格式：

示例：
姓名:张三,年龄:28,城市:北京
→ {"name": "张三", "age": 28, "city": "北京"}

现在转换：
{text}
→""",
        "max_tokens": 256,
    },

    "dedup": {
        "detect": ["去重", "去重复", "重复"],
        "template": TOOL_PREFIX + """去重后输出，保留原始顺序。每行一个：

示例：
apple, banana, apple, orange, banana
→ apple
banana
orange

现在去重：
{text}
→""",
        "max_tokens": 256,
    },

    "sort_numbers": {
        "detect": ["排序", "排列", "按销量", "按数量", "从高到低", "从大到小"],
        "template": "",
        "rule_based": True,
        "max_tokens": 8,
    },

    "sort_alpha": {
        "detect": ["按字母", "按名称", "按名字", "A到Z", "Z到A"],
        "template": TOOL_PREFIX + """按字母顺序排序，每行一个：

示例：
orange, apple, banana
→ apple
banana
orange

现在排序：
{text}
→""",
        "max_tokens": 256,
    },

    "clean_data": {
        "detect": ["清洗", "清理", "格式统一", "规范化"],
        "template": TOOL_PREFIX + """统一数据格式为标准CSV：每行一个记录，逗号分隔字段。

示例：
商品A,100;商品B,50;商品C,200
→ 商品A,100
商品B,50
商品C,200

现在格式化：
{text}
→""",
        "max_tokens": 256,
    },

    "rename_suggest": {
        "detect": ["重命名", "改名为"],
        "template": TOOL_PREFIX + """建议新文件名，保持扩展名不变。格式：原名 → 新名

示例：
IMG_001.jpg → vacation_beach.jpg
Document1.docx → project_report.docx

现在处理：
{text}
→""",
        "max_tokens": 512,
    },

    "tag": {
        "detect": ["打标签", "标记为"],
        "template": TOOL_PREFIX + """给以下内容打一个最合适的类别标签。可选：技术、生活、工作、学习、娱乐、其他
只输出一个类别名：

示例：
"Python列表推导式用法" → 技术
"周末去爬山攻略" → 生活
"季度工作总结" → 工作

现在打标签：
{text}
→""",
        "max_tokens": 16,
    },

    "qa_short": {
        "detect": ["是什么", "什么意思", "解释"],
        "template": TOOL_PREFIX + """用一句话回答：

示例：
问：API是什么意思？
答：应用程序编程接口，用于不同软件之间通信。

现在回答：
问：{text}
答：""",
        "max_tokens": 128,
    },

    # ── 中文企业场景 ──

    "contract_clause": {
        "detect": ["合同条款", "合同提取", "协议条款", "合同约定"],
        "template": TOOL_PREFIX + """从以下合同文本中提取关键条款：

条款类型：内容

示例：
"甲方应在合同签订后30日内支付全部款项，金额为人民币50万元。"
→ 付款期限：合同签订后30日内
付款金额：人民币50万元

现在提取：
{text}
→""",
        "max_tokens": 256,
    },

    "invoice_parse": {
        "detect": ["发票", "票据", "报销", "开票"],
        "template": TOOL_PREFIX + """从以下发票信息中提取结构化数据，每行一个字段：

现在提取：
{text}
→""",
        "max_tokens": 128,
    },

    "meeting_minutes": {
        "detect": ["会议纪要", "会议记录", "会议总结", "会议摘要"],
        "template": TOOL_PREFIX + """将以下会议内容整理为结构化纪要，包含：议题、决议、待办事项

现在整理：
{text}
→""",
        "max_tokens": 512,
    },

    "feedback_classify": {
        "detect": ["客户反馈", "用户反馈", "投诉", "建议分类"],
        "template": TOOL_PREFIX + """将以下客户反馈分类并提取关键问题。

类别可选：产品质量、物流配送、售后服务、价格、功能需求、其他

现在分类：
{text}
→""",
        "max_tokens": 128,
    },

    "data_report": {
        "detect": ["数据分析", "报表", "报告生成", "数据总结"],
        "template": TOOL_PREFIX + """根据以下数据生成简要分析报告，包含：数据概览、关键发现、建议

现在分析：
{text}
→""",
        "max_tokens": 256,
    },
}


def build_optimized_prompt(task_type: str, action: str, text: str, files: list[str]) -> str:
    """根据任务类型选择最佳 prompt 模板"""
    if task_type in PROMPT_TEMPLATES:
        tpl = PROMPT_TEMPLATES[task_type]
        if tpl.get("rule_based"):
            return ""
        content = text or action
        target = ""
        for kw in ["提取", "找出", "抽取出"]:
            if kw in action:
                parts = action.split(kw, 1)
                if len(parts) > 1:
                    target = parts[1].strip().split("从")[0].split("：")[0].strip()
                break
        prompt = tpl["template"].format(text=content, target=target)
        return prompt

    # 泛化 prompt
    prompt = TOOL_PREFIX + action + "\n"
    if text:
        prompt += "\n内容：\n" + text + "\n"
    if files:
        prompt += "\n文件：\n" + "\n".join(f"  {f}" for f in files) + "\n"
    prompt += "\n只输出结果，不要解释。"
    return prompt


def get_max_tokens(task_type: str) -> int:
    """根据任务类型获取合适的 max_tokens"""
    if task_type in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[task_type].get("max_tokens", 256)
    return 512


def compress_prompt_tokens(prompt: str, task_type: str) -> str:
    """压缩长 prompt（去冗余、截断示例）"""
    if len(prompt) < 500:
        return prompt

    # 压缩策略：截断中间的示例部分
    lines = prompt.split("\n")
    if len(lines) > 20:
        # 保留前5行和后5行
        compressed = lines[:5] + ["..."] + lines[-5:]
        return "\n".join(compressed)
    return prompt
