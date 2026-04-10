# -*- coding: utf-8 -*-
import logging
import os
import sys
from logging.handlers import QueueHandler, QueueListener
import queue

# 解决Windows中文乱码问题
sys.stdout.reconfigure(encoding='utf-8')

log_path = os.path.join("logs", "run.log")
log_dir = os.path.dirname(log_path)
os.makedirs(log_dir, exist_ok=True)

# 1. 定义 Handler（先创建好，但不直接加到 root logger）
file_handler = logging.FileHandler(log_path, encoding="utf-8")
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 2. 启动异步监听线程
log_queue = queue.Queue(-1)  # 无界队列
queue_listener = QueueListener(log_queue, file_handler, console_handler)
queue_listener.start()

# 3. 配置 root logger 使用 QueueHandler
logging.basicConfig(
    level=logging.INFO,
    handlers=[QueueHandler(log_queue)]  # 只加这一个 Handler
)
logger = logging.getLogger(__name__)

# 【可选】程序退出时记得停止监听（防止丢最后几条日志）
# import atexit
# atexit.register(queue_listener.stop)

# ===================== 系统编码修复（Windows专属） =====================
import sys
import io
# 强制标准输出使用 UTF-8 编码，避免 Windows GBK 编码报错
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ===================== 1. Python 标准库 =====================
import os
import re
import json
import uuid
import asyncio
import operator
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, Optional

# ===================== 2. 数据科学与可视化库 =====================
import numpy as np
import matplotlib.pyplot as plt

# ===================== 3. 数据库 ORM 库 (SQLAlchemy) =====================
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ===================== 4. 环境变量加载 =====================
from dotenv import load_dotenv

# ===================== 5. LangChain 核心（LLM/消息/工具） =====================
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

# ===================== 6. LangGraph 工作流框架 =====================
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# ===================== 7. MCP 服务框架（模型上下文协议） =====================
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
# ======== 固定图片生成路径 =======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================================================一.配置层：环境变量加载，大模型初始化 ==================================
load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("ENDPOINT_ID"),
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_URL"),
    temperature=0.1,
    timeout=60,
)

# =================================================== 二.数据层 ============================================================
# ================== 1.数据库的配置和初始化 =============================
# 数据库配置：本地SQLite，开箱即用，无需额外服务
DB_URL = "sqlite:///./agent_data.db"
# 数据库引擎初始化（SQLite线程兼容配置）
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
# 数据库会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# ORM基类
Base = declarative_base()




# 1. 对话消息表：全量存储对话链路的每一条消息，可完整复现对话过程
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), comment="消息唯一ID")
    session_id = Column(String(64), index=True, nullable=False, comment="关联的会话ID")
    message_type = Column(String(16), nullable=False, comment="消息类型：human/ai/tool/system")
    content = Column(Text, nullable=False, comment="消息内容")
    sequence = Column(Integer, nullable=False, comment="消息在对话中的顺序号")
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment="消息创建时间")


# 2. 工具调用记录表：全量存储每一次工具调用的入参、输出、状态，可追溯可复现
class ToolCallRecord(Base):
    __tablename__ = "tool_call_records"
    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()), comment="调用记录ID")
    session_id = Column(String(64), index=True, nullable=False, comment="关联的会话ID")
    tool_name = Column(String(64), nullable=False, comment="调用的工具名称")
    tool_args = Column(JSON, nullable=False, comment="工具入参（JSON格式）")
    tool_output = Column(Text, nullable=False, comment="工具返回结果")
    status = Column(String(16), default="success", nullable=False, comment="执行状态：success/failed")
    call_time = Column(DateTime, default=datetime.now, nullable=False, comment="调用时间")


# 首次运行自动创建所有表（已存在则跳过）
Base.metadata.create_all(bind=engine)


# ================== 2.数据写入层 =====================
def get_db_session():
    """获取数据库会话（普通函数版）"""
    return SessionLocal()


def save_message_to_db(session_id: str, message_type: str, content: str, sequence: int):
    """单条消息落库，全链路记录对话过程"""
    db = get_db_session()
    try:
        message_record = ChatMessage(
            session_id=session_id,
            message_type=message_type,
            content=content,
            sequence=sequence
        )
        db.add(message_record)
        db.commit()
    except Exception as e:
        print(f"⚠️ 消息写入数据库失败：{str(e)}")
        db.rollback()
    finally:
        db.close()


def save_tool_call_to_db(session_id: str, tool_name: str, tool_args: dict, tool_output: str, status: str = "success"):
    """工具调用记录落库，完整记录入参和输出，可追溯可复现"""
    db = get_db_session()
    try:
        tool_record = ToolCallRecord(
            session_id=session_id,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_output=tool_output,
            status=status
        )
        db.add(tool_record)
        db.commit()
    except Exception as e:
        print(f"⚠️ 工具调用记录写入数据库失败：{str(e)}")
        db.rollback()
    finally:
        db.close()


# ================================================= 三.工具层 =================================================

# ===================== （1） RAG知识库检索工具 (外挂TXT版本) =====================
def load_knowledge_base(file_path: str = r"E:\PythonProject\doubao2\knowledge.txt"):

    """
    从外部txt文件加载知识库。
    文件格式：使用 --- 分隔不同条目，Keywords: 定义关键词，Content: 定义内容。
    """
    knowledge_base = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 按分隔符切分不同的知识条目
        entries = content.split('---')

        for entry in entries:
            entry = entry.strip()
            if not entry or entry.startswith('#'):
                continue  # 跳过空条目和注释

            # 提取关键词和内容
            keywords_line = ""
            content_line = ""

            lines = entry.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Keywords:"):
                    keywords_line = line.replace("Keywords:", "").strip()
                elif line.startswith("Content:"):
                    content_line = line.replace("Content:", "").strip()

            if keywords_line and content_line:
                # 解析关键词列表
                keywords = [kw.strip() for kw in keywords_line.split(',')]
                knowledge_base.append({
                    "keywords": keywords,
                    "content": content_line
                })

    except FileNotFoundError:
        print(f"⚠️ 警告：未找到知识库文件 {file_path}，将使用空知识库。")
    except Exception as e:
        print(f"⚠️ 警告：读取知识库失败：{str(e)}")

    return knowledge_base


# 全局加载一次知识库（启动时加载，避免每次读盘）
_knowledge_base = load_knowledge_base()


@tool
def retrieve_knowledge(query: str):
    """
    检索线性回归、最小二乘法相关的专业知识，用于回答用户的专业问题。
    【使用场景】：用户询问线性回归、最小二乘法的原理、定义、相关知识时，必须调用这个工具。
    【入参要求】：query为用户的问题原文。
    【返回值】：检索到的专业知识内容。
    """

    if not _knowledge_base:
        return "知识库未加载。"

    for item in _knowledge_base:
        for kw in item["keywords"]:
            if kw in query:
                return item["content"]
    return "未检索到相关专业知识。"


# ===================== （2）Calculate工具（最小二乘法线性回归拟合）===================================
@tool
def calculate(data_points: str | list):
    """
    对输入的(x,y)数据点执行最小二乘法线性回归拟合，预测下一个x对应的y值，生成本地拟合图片。
    【使用场景】：用户提供序列数据、询问分数/数值趋势、预测下一个数值时，必须调用这个工具。
    【入参要求】：data_points可以是[(x1,y1), (x2,y2), ...]格式的列表/JSON字符串，也可以是“第一次xx，第二次xx”的自然语言描述。
    【返回值】：拟合结果、预测值、拟合方程、图片保存路径
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # ========== 【兼容解析，支持自然语言】 ==========
        # 情况1：入参是字符串，先尝试提取所有数字
        if isinstance(data_points, str):
            # 先提取字符串里所有的数字，自动适配自然语言描述
            numbers = re.findall(r"\d+\.?\d*", data_points)
            numbers = [float(num) for num in numbers]
            # 如果是成对的数字，自动转成(x,y)格式，x从1开始
            if len(numbers) >= 2:
                # 如果是“第一次450，第二次470”这种，只有y值，自动补x
                if len(numbers) % 2 != 0:
                    data_points = [(i + 1, num) for i, num in enumerate(numbers)]
                # 如果是成对的x,y，直接转成列表
                else:
                    data_points = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
            else:
                return "错误：无法从输入中提取到有效数据点，至少需要2个数值"

        # 情况2：入参是列表，统一格式
        if isinstance(data_points, list):
            data = np.array(data_points)
            # 如果是一维列表，自动补x从1开始
            if data.ndim == 1:
                data = np.array([(i + 1, num) for i, num in enumerate(data)])
            # 如果是二维列表，只取前两列
            elif data.ndim >= 2:
                data = data[:, :2]
        else:
            return "错误：输入格式无法解析，请检查数据格式"

        x = data[:, 0]
        y = data[:, 1]
        n = len(x)
        if n < 2:
            return "错误：至少需要2个数据点才能完成线性拟合"
    except Exception as e:
        return f"输入解析失败：{str(e)}，请检查输入格式是否正确"

    # 后面的拟合、画图代码完全保留，不用动
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_square = np.sum(x ** 2)
    denominator = n * sum_x_square - sum_x ** 2

    if denominator == 0:
        return "错误：所有x值相同，无法完成线性拟合"

    w = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - w * sum_x) / n
    next_x = x[-1] + 1
    predict_y = w * next_x + b

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="steelblue", s=80, label="原始数据点")
    x_fit = np.linspace(x.min(), next_x, 100)
    y_fit = w * x_fit + b
    plt.plot(x_fit, y_fit, color="coral", linewidth=2, label=f"拟合直线: y = {w:.2f}x + {b:.2f}")
    plt.scatter(next_x, predict_y, color="red", s=100, marker="*", label=f"预测点(x={next_x}, y={predict_y:.2f})")
    plt.xlabel("X序列")
    plt.ylabel("Y值")
    plt.title("线性回归趋势拟合与预测")
    plt.legend()
    plt.grid(True, alpha=0.3)
    image_path = os.path.join(SCRIPT_DIR, "linear_fit_result.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()

    return f"""🎉 恭喜你成功调用MCP工具calculate！

    拟合完成：
    - 拟合直线方程：y = {w:.4f}x + {b:.4f}
    - 预测值：第{next_x}次的预测分数为 {predict_y:.2f} 分
    - 图片保存路径：{image_path}

    【重要提示】请不要尝试执行任何系统命令（如 start、Invoke-Item、explorer 等）来打开图片，图片路径已提供给用户手动查看。最终结果请用中文输出"""

#=================== StateGragh的初始化和工作流构建 ============================
# 1. 定义工作流全局State（兼容后续所有需求）定义各项参数内容
class AgentState(TypedDict):
    # 对话历史：operator.add实现消息累加，核心循环的基础
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 会话唯一ID，用于数据库存储、状态持久化
    session_id: str
    # 用户原始输入
    user_input: str
    # 结构化结果，完全兼容原有逻辑
    structured_result: dict
    # 工具调用记录，用于前端展示、数据库存储
    tool_calls: Annotated[list, operator.add]
    # 错误信息，用于异常兜底
    error_msg: Optional[str]

    final_json: Optional[str]

# =========================================== 四. Agent层：State定义、节点函数、路由、工作流构建 ==============================
# 2. 工作流节点1：Agent思考环节（对应循环的「思考」步骤）
def agent_think_node(state: AgentState) -> dict:
    try:
        # 系统提示词完全兼容原有逻辑，仅新增异常兜底提示
        # 重写后的强规则System Prompt，100%触发工具调用
        system_prompt = SystemMessage(content="""你是专业的线性回归数据分析助手，必须严格遵守以下铁则，禁止任何违反：
        1. 【强制工具调用规则】只要用户提供任何序列数据、分数趋势、数值预测需求，无论用户用什么格式描述，你必须：
           a. 先把用户的自然语言描述转换成[(x1,y1), (x2,y2), ...]格式：x是从1开始的序号（第一次=1，第二次=2，以此类推），y是对应的数值；
           b. 必须调用calculate工具，把转换后的列表作为data_points入参传入，禁止自己计算、禁止直接回答用户问题、禁止跳过工具调用。
        2. 【知识查询规则】用户询问线性回归、最小二乘法相关专业知识时，必须调用retrieve_knowledge工具，禁止编造任何信息。
        3. 【结果使用规则】只有工具返回结果后，你才能基于工具的返回内容回答用户，禁止脱离工具结果编造内容。
        4. 【异常处理规则】如果工具执行报错，直接把完整的错误信息告知用户，不要二次加工。
        """)
        # 拼接对话历史
        messages = [system_prompt] + state["messages"]
        # 绑定工具，和原有逻辑完全一致
        tools = [calculate, retrieve_knowledge]
        model_with_tools = llm.bind_tools(tools)
        # 大模型决策
        ai_response = model_with_tools.invoke(messages)

        # ========== 【需求3新增：AI思考消息落库】 ==========
        save_message_to_db(
            session_id=state["session_id"],
            message_type="ai",
            content=ai_response.content,
            sequence=len(state["messages"]) + 1
        )

        return {
            "messages": [ai_response],
            "tool_calls": ai_response.tool_calls if hasattr(ai_response, "tool_calls") else [],
            "error_msg": None
        }
    except Exception as e:
        error_content = f"思考环节异常：{str(e)}"
        # ========== 【需求3新增：异常消息落库】 ==========
        save_message_to_db(
            session_id=state["session_id"],
            message_type="system",
            content=error_content,
            sequence=len(state["messages"]) + 1
        )
        return {
            "error_msg": error_content,
            "messages": [SystemMessage(content=error_content)]
        }


# 3. 工作流节点2：工具执行环节（对应循环的「调工具」步骤）
def tool_execute_node(state: AgentState) -> dict:
    try:
        # 拿到上一步的工具调用指令
        ai_response = state["messages"][-1]
        tool_messages = []
        structured_result = state["structured_result"]
        tool_call_records = []
        # 消息顺序号
        current_sequence = len(state["messages"]) + 1

        # 遍历执行所有工具调用
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"].lower()
            # 匹配工具，兼容原有逻辑
            selected_tool = {
                "calculate": calculate,
                "retrieve_knowledge": retrieve_knowledge
            }.get(tool_name)

            if not selected_tool:
                tool_output = f"错误：未找到工具{tool_name}"
                call_status = "failed"
            else:
                # 执行工具
                tool_output = selected_tool.invoke(tool_call["args"])
                call_status = "success"

            print(f"✅ 工具执行完成：{tool_name}，结果：{tool_output[:100]}...")
            # 生成工具消息
            tool_msg = ToolMessage(tool_output, tool_call_id=tool_call["id"])
            tool_messages.append(tool_msg)
            # 记录工具调用
            tool_call_records.append({
                "tool_name": tool_name,
                "args": tool_call["args"],
                "output": tool_output
            })

            # ========== 【需求3新增：工具调用记录+工具消息落库】 ==========
            # 保存工具调用记录
            save_tool_call_to_db(
                session_id=state["session_id"],
                tool_name=tool_name,
                tool_args=tool_call["args"],
                tool_output=tool_output,
                status=call_status
            )
            # 保存工具返回消息
            save_message_to_db(
                session_id=state["session_id"],
                message_type="tool",
                content=tool_output,
                sequence=current_sequence
            )
            current_sequence += 1

            # 提取calculate工具的结构化信息，完全兼容原有逻辑
            if tool_name == "calculate":
                eq_match = re.search(r"y = [0-9\.\-\+x ]+", tool_output)
                pred_match = re.search(r"预测值为([0-9\.]+)", tool_output)
                path_match = re.search(r"保存到(.+)$", tool_output)
                if eq_match: structured_result["fitting_equation"] = eq_match.group()
                if pred_match: structured_result["prediction_result"] = pred_match.group(1)
                if path_match: structured_result["image_path"] = path_match.group(1)

        return {
            "messages": tool_messages,
            "structured_result": structured_result,
            "tool_calls": tool_call_records,
            "error_msg": None
        }
    except Exception as e:
        error_content = f"工具执行异常：{str(e)}"
        # ========== 【需求3新增：异常消息落库】 ==========
        save_message_to_db(
            session_id=state["session_id"],
            message_type="system",
            content=error_content,
            sequence=len(state["messages"]) + 1
        )
        return {
            "error_msg": error_content,
            "messages": [SystemMessage(content=error_content)]
        }


# 4. 工作流节点3：结果生成环节（对应循环的「响应」步骤）
def result_generate_node(state: AgentState) -> dict:
    try:
        structured_result = state["structured_result"]
        error_msg = state["error_msg"]

        # 异常兜底
        if error_msg:
            json_prompt = f"""
            请基于以下异常信息，生成严格合法的JSON格式回复，禁止输出任何JSON以外的内容。
            必须包含以下字段：
            - user_intent: "异常处理"
            - prediction_result: "无"
            - fitting_equation: "无"
            - image_path: "无"
            - final_reply: "执行出现异常：{error_msg}，请重试或检查输入"
            """
        else:
            # 原有JSON生成逻辑完全保留
            json_prompt = f"""
            请基于以上对话内容，生成严格合法的JSON格式回复，禁止输出任何JSON以外的内容，禁止输出Markdown代码块标记。
            必须包含以下字段：
            - user_intent: 字符串，用户的核心需求意图（如"数据分析与预测"、"知识咨询"、"闲聊"）
            - prediction_result: 字符串，预测的数值结果，无预测则填"无"
            - fitting_equation: 字符串，拟合的直线方程，无拟合则填"无"
            - image_path: 字符串，拟合图片的本地绝对路径，无图片则填"无"
            - final_reply: 字符串，给用户的通顺自然的中文回复

            【已知结构化信息】（如果有）：
            - 预测值：{structured_result["prediction_result"]}
            - 拟合方程：{structured_result["fitting_equation"]}
            - 图片路径：{structured_result["image_path"]}
            """
        # 拼接对话历史
        messages = state["messages"] + [HumanMessage(json_prompt)]
        # 调用大模型生成最终JSON
        final_response = llm.invoke(messages)
        # 清理JSON格式，完全兼容原有逻辑
        json_str = final_response.content.strip()
        if json_str.startswith("```json"): json_str = json_str[7:]
        if json_str.startswith("```"): json_str = json_str[3:]
        if json_str.endswith("```"): json_str = json_str[:-3]

        # ========== 【需求3新增：最终结果消息落库】 ==========
        save_message_to_db(
            session_id=state["session_id"],
            message_type="system",
            content=json_str,
            sequence=len(state["messages"]) + 1
        )

        # ===================== 【修复2：确保返回 final_json】 =====================
        return {
            "messages": [SystemMessage(content=json_str)],
            "final_json": json_str,  # 这里现在是合法的，因为 State 里已经定义了
            "error_msg": None
        }
    except Exception as e:
        error_content = f"结果生成异常：{str(e)}"
        json_str = json.dumps({
            "user_intent": "异常处理",
            "prediction_result": "无",
            "fitting_equation": "无",
            "image_path": "无",
            "final_reply": error_content
        }, ensure_ascii=False)
        # ========== 【需求3新增：异常消息落库】 ==========
        save_message_to_db(
            session_id=state["session_id"],
            message_type="system",
            content=json_str,
            sequence=len(state["messages"]) + 1
        )
        # ===================== 【修复2：异常时也返回 final_json】 =====================
        return {
            "error_msg": error_content,
            "final_json": json_str,
            "messages": [SystemMessage(content=json_str)]
        }

# 5. 工作流路由：实现「思考→调工具→思考」的循环核心逻辑
def route_should_call_tool(state: AgentState) -> str:
    # 异常直接结束流程
    if state.get("error_msg"):
        return "generate_result"
    # 拿到思考节点的输出
    ai_response = state["messages"][-1]
    # 有工具调用，走工具执行节点，完成循环
    if len(ai_response.tool_calls) > 0:
        return "call_tool"
    # 无工具调用，直接生成最终响应
    return "generate_result"


# 6. 构建StateGraph工作流
def build_agent_workflow():
    # ===================== 【修复2：使用MemorySaver替代SqliteSaver】 =====================
    memory = MemorySaver()  # 状态持久化（修复版）
    # 初始化StateGraph，传入定义的AgentState
    workflow = StateGraph(AgentState)

    # 注册3个核心节点，对应「思考→调工具→响应」全流程
    workflow.add_node("agent_think", agent_think_node)  # 思考节点
    workflow.add_node("tool_execute", tool_execute_node)  # 调工具节点
    workflow.add_node("result_generate", result_generate_node)  # 响应节点

    # 设置流程起点
    workflow.add_edge(START, "agent_think")

    # 设置条件路由：实现循环核心逻辑
    workflow.add_conditional_edges(
        "agent_think",
        route_should_call_tool,
        {
            "call_tool": "tool_execute",  # 需要调工具，进入工具执行节点
            "generate_result": "result_generate"  # 无需调工具，直接生成结果
        }
    )

    # 工具执行完成后，回到思考节点，完美实现循环闭环
    workflow.add_edge("tool_execute", "agent_think")

    # 结果生成完成后，结束流程
    workflow.add_edge("result_generate", END)

    # 编译工作流，绑定持久化存储器
    return workflow.compile(checkpointer=memory)


# 初始化工作流（全局单例，后续所有需求复用）
agent_graph = build_agent_workflow()

# ========== MCP Server实现层 适配 mcp>=1.0】 ==========
# 初始化 MCP 服务
mcp_server = Server("linear-regression-analysis-agent")


# 1. 注册工具列表（新版 API）
@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="retrieve_knowledge_mcp",
            description="检索线性回归、最小二乘法相关的专业知识。入参：query (用户的问题原文)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户的问题原文"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="calculate_mcp",
            description="对(x,y)数据点执行最小二乘法线性回归拟合，预测下一个值。入参：data_points (格式：[(x1,y1), (x2,y2), ...] 的字符串)",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_points": {"type": "string", "description": "[(x1,y1), (x2,y2), ...] 格式的字符串"}
                },
                "required": ["data_points"]
            }
        ),
        Tool(
            name="agent_session_mcp",
            description="调用完整智能体全流程会话，自动思考、调工具、数据持久化。入参：user_input (用户的完整问题)",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "用户的完整问题输入"}
                },
                "required": ["user_input"]
            }
        )
    ]


# 2. 处理工具调用（新版 API）
@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "retrieve_knowledge_mcp":
        query = arguments["query"]

        result = retrieve_knowledge.invoke(query)
        return [TextContent(type="text", text=result)]

    elif name == "calculate_mcp":
        data_points = arguments["data_points"]
        result = calculate.invoke(data_points)
        return [TextContent(type="text", text=result)]

    elif name == "agent_session_mcp":
        user_input = arguments["user_input"]
        session_id = str(uuid.uuid4())


        # ===================== 【修复3：初始化时也加上 final_json】 =====================
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "session_id": session_id,
            "user_input": user_input,
            "structured_result": {
                "fitting_equation": "无",
                "prediction_result": "无",
                "image_path": "无"
            },
            "tool_calls": [],
            "error_msg": None,
            "final_json": None  # 初始化时也加上这个字段
        }

        save_message_to_db(
            session_id=session_id,
            message_type="human",
            content=user_input,
            sequence=1
        )

        config = RunnableConfig(configurable={"thread_id": session_id})
        run_status = "success"
        final_json = ""

        try:
            final_state = agent_graph.invoke(initial_state, config=config)
            # ===================== 【修复3：用 get 安全取值，避免 KeyError】 =====================
            final_json = final_state.get("final_json", json.dumps({
                "user_intent": "异常处理",
                "prediction_result": "无",
                "fitting_equation": "无",
                "image_path": "无",
                "final_reply": "Agent执行完成，未生成最终结果"
            }, ensure_ascii=False))
        except Exception as e:
            run_status = "failed"
            # 打印完整异常堆栈，方便调试
            import traceback

            print("=== Agent执行异常堆栈 ===")
            traceback.print_exc()
            final_json = json.dumps({
                "user_intent": "异常处理",
                "prediction_result": "无",
                "fitting_equation": "无",
                "image_path": "无",
                "final_reply": f"工作流执行异常：{str(e)}"
            }, ensure_ascii=False)



        return [TextContent(type="text", text=final_json)]

    else:
        return [TextContent(type="text", text=f"未知工具：{name}")]


# 新版 MCP 服务启动函数
async def run_mcp_server():
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )

# ===================== 4. 主程序（原有逻辑100%保留，新增MCP启动入口） =====================
if __name__ == "__main__":
    # 启动 MCP Serve（带--mcp参数时启动，新版 API）
    if len(sys.argv) > 1 and sys.argv[1] == "--mcp":
        print("✅ MCP 服务已启动，等待客户端连接...")
        asyncio.run(run_mcp_server())
    else:
        # ...（这里保留你原有的交互式逻辑，完全不用动）...
        # 下面是你原有的代码，直接保留即可
        try:
            with open("agent_workflow.png", "wb") as f:
                f.write(agent_graph.get_graph().draw_mermaid_png())
            print("✅ 工作流可视化图已保存为 agent_workflow.png")
        except Exception as e:
            print(f"⚠️ 工作流可视化图生成失败：{str(e)}")

        print("=" * 60)
        print("📊 数据分析智能体已启动 (LangGraph+数据库持久化版，输入 exit 退出)")
        print("💾 所有会话、消息、工具调用已自动持久化到本地 agent_data.db 数据库")
        print("🚀 启动MCP服务请运行: python 脚本名.py --mcp")
        print("=" * 60)

        while True:
            user_input = input("\n👤 请输入你的问题：")
            if user_input.lower() == "exit":
                print("\n👋 对话结束，程序退出")
                break

            print("\n🔄 正在处理中...")
            session_id = str(uuid.uuid4())


            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "session_id": session_id,
                "user_input": user_input,
                "structured_result": {
                    "fitting_equation": "无",
                    "prediction_result": "无",
                    "image_path": "无"
                },
                "tool_calls": [],
                "error_msg": None
            }
            save_message_to_db(
                session_id=session_id,
                message_type="human",
                content=user_input,
                sequence=1
            )

            config = RunnableConfig(configurable={"thread_id": session_id})
            final_state = None
            run_status = "success"
            run_error = None
            try:
                final_state = agent_graph.invoke(initial_state, config=config)
                final_json = final_state["final_json"]
            except Exception as e:
                run_status = "failed"
                run_error = f"工作流执行异常：{str(e)}"
                final_json = json.dumps({
                    "user_intent": "异常处理",
                    "prediction_result": "无",
                    "fitting_equation": "无",
                    "image_path": "无",
                    "final_reply": run_error
                }, ensure_ascii=False)
                print(f"❌ {run_error}")


            print("\n" + "=" * 60)
            print("🎉 【最终结构化JSON输出】")
            print("=" * 60)
            print(final_json)
            print("=" * 60)