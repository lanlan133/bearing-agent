import sys
import re
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# ============================================================
# 2. 项目模块
# ============================================================

from data_loader import (
    load_mat_dataset,
    load_single_sample,
)

from diagnosis_agent import (
    run_diagnosis_agent,
)


# ============================================================
# 3. Streamlit 页面
# ============================================================

st.set_page_config(
    page_title="轴承润滑诊断智能助手",
    page_icon="🤖",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17243b;
        --muted: #66758a;
        --line: #afc1d5;
        --accent: #1d5f91;
        --soft: #f3f7fb;
        --user: #eef6ff;
    }

    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 4%, #eaf3fb 0, transparent 28%),
            linear-gradient(180deg, #f7fafe 0%, #edf3f9 100%);
        color: var(--ink);
        isolation: isolate;
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        right: max(16px, calc((100vw - 1180px) / 2));
        bottom: 0;
        left: max(16px, calc((100vw - 1180px) / 2));
        z-index: 0;
        pointer-events: none;
        background: #ffffff;
        border-right: 1px solid var(--line);
        border-left: 1px solid var(--line);
        box-shadow: 0 0 50px rgba(35, 69, 103, 0.08);
    }

    [data-testid="stAppViewContainer"] {
        position: relative;
        z-index: 1;
        background: transparent;
        scroll-padding-bottom: 10rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        min-height: calc(100vh - 32px);
        margin: 16px auto;
        padding: 1.7rem 3.2rem 8rem;
        background: transparent;
        border: 0;
        border-radius: 0;
        box-shadow: none;
    }

    [data-testid="stMainBlockContainer"]::after {
        content: "";
        display: block;
        width: 100%;
        height: 8rem;
    }

    .chat-hero {
        text-align: center;
        margin: 0.15rem 0 1.7rem;
    }

    .chat-hero h1 {
        margin: 0;
        color: var(--ink);
        font-size: clamp(1.65rem, 3vw, 2.25rem);
        font-weight: 720;
        letter-spacing: 0.04em;
    }

    .chat-hero p {
        margin: 0.55rem 0 0;
        color: var(--muted);
        font-size: 0.92rem;
    }

    [data-testid="stChatMessage"] {
        padding: 0.8rem 0.25rem;
        margin: 0.15rem 0;
        background: transparent;
        border: 0;
    }

    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none;
    }

    [data-testid="stChatMessageContent"] {
        width: 100%;
        min-width: 0;
        color: var(--ink);
        font-size: 1rem;
        line-height: 1.85;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
        width: 90%;
        max-width: 90%;
        margin-right: 0;
        margin-left: auto;
        padding: 0.85rem 1.1rem;
        box-sizing: border-box;
        background: var(--user);
        border: 1px solid #d8e7f5;
        border-radius: 18px;
    }

    .role-label {
        display: inline-block;
        margin-bottom: 0.3rem;
        color: var(--accent);
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    .role-label.user {
        color: #315b7d;
    }

    [data-testid="stChatMessageContent"] h2 {
        margin-top: 0.1rem;
        font-size: 1.6rem;
    }

    [data-testid="stChatMessageContent"] h3 {
        margin-top: 1.15rem;
        color: #203e5d;
        font-size: 1.18rem;
    }

    [data-testid="stChatMessageContent"] code {
        color: #146b55;
        background: #f2f8f6;
        border-radius: 6px;
        padding: 0.12rem 0.32rem;
    }

    [data-testid="stBottomBlockContainer"] {
        position: fixed !important;
        right: auto;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999;
        width: min(1180px, calc(100% - 32px));
        max-width: 1180px;
        margin: 0;
        padding: 1.35rem 3.2rem 1.2rem;
        box-sizing: border-box;
        background: #ffffff;
        border-right: 1px solid var(--line);
        border-left: 1px solid var(--line);
    }

    [data-testid="stChatInput"] {
        width: 100%;
        background: rgba(255, 255, 255, 0.98);
        border: 1.5px solid #426887;
        border-radius: 22px;
        box-shadow: 0 10px 30px rgba(35, 69, 103, 0.12);
    }

    [data-testid="stChatInput"] textarea {
        min-height: 3.25rem;
        padding-top: 0.95rem;
        font-size: 1rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px;
    }

    @media (max-width: 760px) {
        .stApp::before {
            top: 0;
            right: 6px;
            bottom: 0;
            left: 6px;
        }

        [data-testid="stMainBlockContainer"] {
            min-height: calc(100vh - 12px);
            margin: 6px;
            padding: 1.2rem 1rem 7.5rem;
            border-radius: 24px;
        }

        [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ) {
            width: 100%;
            max-width: 100%;
        }

        .chat-hero p {
            display: none;
        }

        [data-testid="stBottomBlockContainer"] {
            width: calc(100% - 12px);
            padding: 1rem 1rem 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="chat-hero">
        <h1>轴承润滑诊断智能助手</h1>
        <p>MSM-17D · MSM-ADASYN-Stacking · Review Policy · 智能诊断对话</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. Session State
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "你好，我是轴承润滑诊断助手。\n\n"
                "请上传一个 `.mat` 振动数据文件。"
            ),
        }
    ]


if "mat_file_path" not in st.session_state:
    st.session_state.mat_file_path = None


if "mat_file_name" not in st.session_state:
    st.session_state.mat_file_name = None


if "total_samples" not in st.session_state:
    st.session_state.total_samples = 0


if "last_sample_index" not in st.session_state:
    st.session_state.last_sample_index = None


if "last_report" not in st.session_state:
    st.session_state.last_report = None


if "last_sample" not in st.session_state:
    st.session_state.last_sample = None


# ============================================================
# 5. 中文数字转换
# ============================================================

CN_NUM = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def chinese_number_to_int(text):
    """
    支持：
    五
    十
    十五
    二十
    二十五
    一百零五
    """

    if not text:
        return None

    # 纯数字
    if text.isdigit():
        return int(text)

    total = 0
    current = 0

    units = {
        "十": 10,
        "百": 100,
        "千": 1000,
    }

    for char in text:

        if char in CN_NUM:

            current = CN_NUM[char]

        elif char in units:

            unit = units[char]

            if current == 0:
                current = 1

            total += current * unit
            current = 0

        else:
            return None

    total += current

    return total


# ============================================================
# 6. 从用户文本提取样本编号
# ============================================================

def parse_sample_number(text):
    """
    用户使用 1-based 编号：
    5      -> 第5条
    第5条  -> 第5条
    第五条 -> 第5条

    返回 Python 0-based index。
    """

    text = text.strip()

    # --------------------------------------------
    # 阿拉伯数字
    # --------------------------------------------

    patterns = [
        r"第\s*(\d+)\s*条",
        r"第\s*(\d+)\s*个",
        r"处理\s*(\d+)",
        r"分析\s*(\d+)",
        r"^\s*(\d+)\s*$",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            number = int(
                match.group(1)
            )

            return number - 1


    # --------------------------------------------
    # 中文数字
    # --------------------------------------------

    match = re.search(
        r"第([零一二两三四五六七八九十百千]+)[条个]",
        text
    )

    if match:

        number = chinese_number_to_int(
            match.group(1)
        )

        if number is not None:
            return number - 1


    # 例如：
    # 分析第五条
    match = re.search(
        r"([零一二两三四五六七八九十百千]+)",
        text
    )

    if match:

        number = chinese_number_to_int(
            match.group(1)
        )

        if number is not None:
            return number - 1

    return None


# ============================================================
# 7. 保存上传的 MAT 文件
# ============================================================

def save_uploaded_mat(uploaded_file):

    temp_dir = (
        PROJECT_ROOT
        / "data"
        / "uploaded"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        temp_dir
        / uploaded_file.name
    )

    output_path.write_bytes(
        uploaded_file.getvalue()
    )

    return output_path


# ============================================================
# 8. 读取数据集数量
# ============================================================

def load_uploaded_dataset(file_path):

    dataset = load_mat_dataset(
        file_path
    )

    try:
        total = len(dataset)

    except Exception:
        raise RuntimeError(
            "无法获取 MAT 数据集样本数量，"
            "需要根据 data_loader.py 的返回结构调整。"
        )

    return dataset, total


# ============================================================
# 9. 诊断结果格式化
# ============================================================

def format_diagnosis_result(
    sample_number,
    sample,
    report
):

    diagnosis = report.get(
        "diagnosis",
        "未知"
    )

    probability = float(
        report.get(
            "stacking_probability",
            report.get(
                "risk_probability",
                0.0
            )
        )
    )

    model_diagnosis = report.get(
        "model_diagnosis",
        "未知"
    )

    base_probabilities = report.get(
        "base_probabilities",
        {}
    )

    rf_probability = float(
        base_probabilities.get(
            "rf",
            report.get("rf_probability", 0.0)
        )
    )
    xgb_probability = float(
        base_probabilities.get(
            "xgb",
            report.get("xgb_probability", 0.0)
        )
    )
    svm_probability = float(
        base_probabilities.get(
            "svm",
            report.get("svm_probability", 0.0)
        )
    )

    review_policy = report.get(
        "review_policy",
        {}
    )
    lower_threshold = float(
        review_policy.get("lower_threshold", 0.01)
    )
    upper_threshold = float(
        review_policy.get("upper_threshold", 0.96)
    )

    risk_level = report.get(
        "risk_level",
        "未知"
    )

    decision_status = report.get(
        "decision_status",
        ""
    )

    lines = []

    lines.append(
        f"## 第 {sample_number} 条数据诊断结果"
    )

    lines.append("")

    # --------------------------------------------
    # 样本信息
    # --------------------------------------------

    equipment_id = sample.get(
        "equipment_id",
        "-"
    )

    measurement_point = sample.get(
        "measurement_point",
        "-"
    )

    sampling_rate = sample.get(
        "sampling_rate",
        "-"
    )

    speed = sample.get(
        "speed",
        "-"
    )

    lines.append(
        f"**设备编号：** {equipment_id}"
    )

    lines.append(
        f"**测点：** {measurement_point}"
    )

    lines.append(
        f"**采样频率：** {sampling_rate} Hz"
    )

    lines.append(
        f"**转速：** {speed} rpm"
    )

    lines.append("")

    # --------------------------------------------
    # 诊断
    # --------------------------------------------

    lines.append(
        f"### Agent 最终决策：{diagnosis}"
    )

    lines.append(
        f"**MSM-ADASYN-Stacking 综合诊断：{model_diagnosis}**"
    )

    lines.append(
        f"**Stacking 最终概率：{probability:.2%}**"
    )

    lines.append(
        "**Level-0 基学习器概率："
        f"RF={rf_probability:.2%}，"
        f"XGBoost={xgb_probability:.2%}，"
        f"SVM={svm_probability:.2%}**"
    )

    lines.append(
        "**Review Policy："
        f"{lower_threshold:.2f}～{upper_threshold:.2f}**"
    )

    lines.append(
        f"**风险等级：{risk_level}**"
    )

    if decision_status == "review":

        lines.append("")

        lines.append(
            "⚠️ **该样本位于 Review 区间，"
            "系统建议进行人工复核。**"
        )

        review_reason = report.get(
            "review_reason"
        )

        if review_reason:

            lines.append(
                f"\n复核原因：{review_reason}"
            )

    # --------------------------------------------
    # XGBoost 基学习器 SHAP
    # --------------------------------------------

    lines.append("")
    lines.append("### 判断依据")

    evidence = report.get(
        "top_evidence",
        []
    )

    if evidence:

        for rank, item in enumerate(
            evidence[:5],
            start=1
        ):

            feature = item.get(
                "feature",
                "-"
            )

            name = item.get(
                "feature_name",
                "-"
            )

            value = item.get(
                "feature_value",
                0
            )

            shap_value = item.get(
                "shap_value",
                0
            )

            mechanism = item.get(
                "mechanism",
                ""
            )

            lines.append(
                (
                    f"{rank}. **{feature} {name}**  \n"
                    f"   特征值：`{value}`  \n"
                    f"   SHAP：`{float(shap_value):+.4f}`  \n"
                    f"   {mechanism}"
                )
            )

    else:

        lines.append(
            "当前报告未返回 SHAP 证据。"
        )

    # --------------------------------------------
    # 建议
    # --------------------------------------------

    advice = report.get(
        "maintenance_advice",
        []
    )

    if advice:

        lines.append("")
        lines.append("### 建议")

        for item in advice:
            lines.append(
                f"- {item}"
            )

    lines.append("")

    lines.append(
        "> SHAP 表示 XGBoost 基学习器的判别贡献，"
        "不代表 Stacking 整体贡献，"
        "也不应单独解释为已经证明的物理因果关系。"
    )

    return "\n".join(
        lines
    )


# ============================================================
# 10. 显示聊天历史
# ============================================================

def render_chat_message(
    role,
    content
):
    """统一渲染助手和用户消息。"""

    role_name = (
        "助手"
        if role == "assistant"
        else "用户"
    )

    role_class = (
        "assistant"
        if role == "assistant"
        else "user"
    )

    with st.chat_message(role):
        st.markdown(
            (
                f'<span class="role-label {role_class}">'
                f'{role_name}</span>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown(content)


for message in st.session_state.messages:
    render_chat_message(
        message["role"],
        message["content"],
    )


# ============================================================
# 11. GPT 风格输入框
# ============================================================

prompt = st.chat_input(
    "输入样本编号或问题，也可以上传 .mat 文件…",
    accept_file=True,
    file_type=["mat"],
)


# ============================================================
# 12. 处理用户输入
# ============================================================

if prompt:

    # --------------------------------------------
    # 新版 Streamlit ChatInputValue
    # --------------------------------------------

    user_text = ""

    uploaded_files = []

    if hasattr(prompt, "text"):

        user_text = (
            prompt.text
            or ""
        )

        uploaded_files = list(
            prompt.files
            or []
        )

    else:

        user_text = str(prompt)


    # ========================================================
    # A. 文件上传
    # ========================================================

    if uploaded_files:

        uploaded_file = uploaded_files[0]

        st.session_state.messages.append(
            {
                "role": "user",
                "content": (
                    f"📎 已上传：`{uploaded_file.name}`"
                ),
            }
        )

        try:

            file_path = save_uploaded_mat(
                uploaded_file
            )

            with st.spinner(
                "正在读取 MAT 文件..."
            ):

                dataset, total_samples = (
                    load_uploaded_dataset(
                        file_path
                    )
                )

            st.session_state.mat_file_path = (
                str(file_path)
            )

            st.session_state.mat_file_name = (
                uploaded_file.name
            )

            st.session_state.total_samples = (
                total_samples
            )

            # 上传新文件，清空旧诊断
            st.session_state.last_report = None
            st.session_state.last_sample = None
            st.session_state.last_sample_index = None

            assistant_text = (
                f"✅ 文件 `{uploaded_file.name}` "
                f"读取完毕。\n\n"
                f"一共加载 **{total_samples} 条数据**。\n\n"
                f"你想处理哪一条？\n\n"
                f"例如可以输入：`5`、`第5条` 或 `第五条`。"
            )

        except Exception as exc:

            assistant_text = (
                "❌ MAT 文件读取失败：\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )

        st.rerun()


    # ========================================================
    # B. 普通文本
    # ========================================================

    if user_text.strip():

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )


        # ----------------------------------------------------
        # 尚未上传 MAT
        # ----------------------------------------------------

        if not st.session_state.mat_file_path:

            assistant_text = (
                "请先上传一个 `.mat` 数据文件，"
                "读取完成后我再帮你选择并诊断样本。"
            )


        else:

            # ------------------------------------------------
            # 判断用户是否在选择样本
            # ------------------------------------------------

            sample_index = parse_sample_number(
                user_text
            )

            if sample_index is not None:

                total_samples = (
                    st.session_state.total_samples
                )

                if (
                    sample_index < 0
                    or
                    sample_index >= total_samples
                ):

                    assistant_text = (
                        f"当前数据集共有 "
                        f"**{total_samples} 条数据**。\n\n"
                        f"请输入 1 ～ {total_samples} "
                        f"之间的样本编号。"
                    )

                else:

                    human_number = (
                        sample_index + 1
                    )

                    with st.spinner(
                        f"正在诊断第 {human_number} 条数据..."
                    ):

                        sample = load_single_sample(
                            Path(
                                st.session_state.mat_file_path
                            ),
                            sample_index=sample_index,
                        )

                        report = run_diagnosis_agent(
                            signal_data=sample[
                                "signal"
                            ],
                            sampling_rate=sample[
                                "sampling_rate"
                            ],
                            top_n=5,
                        )

                    st.session_state.last_sample_index = (
                        sample_index
                    )

                    st.session_state.last_sample = (
                        sample
                    )

                    st.session_state.last_report = (
                        report
                    )

                    assistant_text = (
                        format_diagnosis_result(
                            human_number,
                            sample,
                            report,
                        )
                    )

            # ------------------------------------------------
            # 已诊断后用户继续聊天
            # ------------------------------------------------

            elif (
                st.session_state.last_report
                is not None
            ):

                assistant_text = (
                    "我已经保留了上一条样本的诊断结果。\n\n"
                    "下一步我们可以把这里接入 DeepSeek，"
                    "这样你就可以继续问：\n\n"
                    "- 为什么判定为轴承润滑不良？\n"
                    "- 哪个特征影响最大？\n"
                    "- 应该怎么维护？\n"
                    "- 这个概率可靠吗？\n"
                    "- 给我生成一份完整报告。\n"
                )

            else:

                assistant_text = (
                    "我没有识别出你要处理的样本编号。\n\n"
                    "例如可以输入：`5`、`第5条`、"
                    "`第五条`。"
                )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )

        st.rerun()
