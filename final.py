from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
import os
from dotenv import load_dotenv
import json
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 基础配置与大模型初始化 =====================
load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("ENDPOINT_ID"),
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_URL"),
    temperature=0.1,  # 调低温度，让JSON输出更稳定
    timeout=60,
)


# ===================== 2. RAG知识库检索工具 =====================
@tool
def retrieve_knowledge(query: str):
    """
    检索线性回归、最小二乘法相关的专业知识，用于回答用户的专业问题。
    【使用场景】：用户询问线性回归、最小二乘法的原理、定义、相关知识时，必须调用这个工具。
    【入参要求】：query为用户的问题原文。
    【返回值】：检索到的专业知识内容。
    """
    knowledge_base = [
        {
            "keywords": ["线性回归", "拟合", "wx+b", "线性关系"],
            "content": "线性回归是一种用于预测数值型目标变量的统计方法，通过拟合自变量x和因变量y之间的线性关系y=wx+b来实现预测，其中w是斜率，b是截距。它是数据分析中最基础、最常用的趋势预测方法。"
        },
        {
            "keywords": ["最小二乘法", "最小二乘", "参数求解"],
            "content": "最小二乘法是线性回归最经典的参数求解方法，核心逻辑是最小化所有数据点的预测值与真实值的平方误差和，从而计算出全局最优的斜率w和截距b，不会陷入局部最优解。"
        }
    ]
    # 关键词匹配检索
    for item in knowledge_base:
        for kw in item["keywords"]:
            if kw in query:
                return item["content"]
    return "未检索到相关专业知识。"


# ===================== 3. 终极容错版Calculate工具 =====================
@tool
def calculate(data_points: str | list):
    """
    对输入的(x,y)数据点执行最小二乘法线性回归拟合，预测下一个x对应的y值，生成本地拟合图片。
    【使用场景】：用户提供序列数据、询问分数/数值趋势、预测下一个数值时，必须调用这个工具。
    【入参要求】：data_points必须是[(x1,y1), (x2,y2), ...]格式的列表，或该格式的JSON字符串，x为从1开始递增的序列数字，y为对应的数值。
    【返回值】：拟合结果、预测值、拟合方程、图片保存路径
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # 超强容错解析
        if isinstance(data_points, str):
            data_points = data_points.replace("(", "[").replace(")", "]").strip()
            data_points = json.loads(data_points)
        data = np.array(data_points)
        if data.ndim == 1:
            if len(data) % 2 != 0:
                return "错误：数据格式不对"
            data = data.reshape(-1, 2)
        x = data[:, 0]
        y = data[:, 1]
        n = len(x)
        if n < 2:
            return "错误：至少需要2个数据点才能完成线性拟合"
    except Exception as e:
        return f"输入解析失败：{str(e)}"

    # 最小二乘法计算
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

    # 生成拟合图
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
    image_path = os.path.abspath("linear_fit_result.png")
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

    return f"拟合完成：拟合直线方程为y = {w:.4f}x + {b:.4f}，下一个x={next_x}的预测值为{predict_y:.2f}，拟合图片已保存到{image_path}"


# ===================== 4. 整合所有功能的主程序 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("📊 数据分析智能体已启动 (输入 exit 退出)")
    print("=" * 60)

    # 初始化对话历史（实现记忆功能）
    messages = [
        SystemMessage(content="你是专业的数据分析助手，严格遵循以下规则："
                              "1. 用户询问趋势、预测数值时，必须调用calculate工具。"
                              "2. 用户询问专业知识（如线性回归、最小二乘法）时，必须调用retrieve_knowledge工具。"
                              "3. 基于工具返回的结果回答用户，不要编造信息。")
    ]

    # 绑定所有工具
    tools = [calculate, retrieve_knowledge]
    model_with_tools = llm.bind_tools(tools)

    while True:
        # 1. 获取用户输入
        user_input = input("\n👤 请输入你的问题：")
        if user_input.lower() == "exit":
            print("\n👋 对话结束，程序退出")
            break

        print("\n🔄 正在处理中...")
        messages.append(HumanMessage(user_input))

        # 2. 大模型决策：是否调用工具
        ai_response = model_with_tools.invoke(messages)
        messages.append(ai_response)

        # 3. 执行工具调用（如果有）
        tool_result_text = "无"
        fitting_eq = "无"
        pred_result = "无"
        img_path = "无"

        if ai_response.tool_calls:
            for tool_call in ai_response.tool_calls:
                selected_tool = {"calculate": calculate, "retrieve_knowledge": retrieve_knowledge}[
                    tool_call["name"].lower()]
                tool_output = selected_tool.invoke(tool_call["args"])
                print(f"✅ 工具执行完成：{tool_output}")
                messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
                tool_result_text = tool_output

                # 如果是calculate工具，提取结构化信息
                if tool_call["name"].lower() == "calculate":
                    if "拟合直线方程" in tool_output:
                        # 简单提取方程和预测值
                        import re

                        eq_match = re.search(r"y = [0-9\.\-\+x ]+", tool_output)
                        pred_match = re.search(r"预测值为([0-9\.]+)", tool_output)
                        path_match = re.search(r"保存到(.+)$", tool_output)
                        if eq_match: fitting_eq = eq_match.group()
                        if pred_match: pred_result = pred_match.group(1)
                        if path_match: img_path = path_match.group(1)

        # 4. 强制生成最终JSON格式输出
        json_prompt = f"""
        请基于以上对话内容，生成严格合法的JSON格式回复，禁止输出任何JSON以外的内容，禁止输出Markdown代码块标记。
        必须包含以下字段：
        - user_intent: 字符串，用户的核心需求意图（如"数据分析与预测"、"知识咨询"、"闲聊"）
        - prediction_result: 字符串，预测的数值结果，无预测则填"无"
        - fitting_equation: 字符串，拟合的直线方程，无拟合则填"无"
        - image_path: 字符串，拟合图片的本地绝对路径，无图片则填"无"
        - final_reply: 字符串，给用户的通顺自然的中文回复

        【已知结构化信息】（如果有）：
        - 预测值：{pred_result}
        - 拟合方程：{fitting_eq}
        - 图片路径：{img_path}
        """
        messages.append(HumanMessage(json_prompt))
        final_response = llm.invoke(messages)

        # 5. 清理并输出JSON
        json_str = final_response.content.strip()
        if json_str.startswith("```json"): json_str = json_str[7:]
        if json_str.startswith("```"): json_str = json_str[3:]
        if json_str.endswith("```"): json_str = json_str[:-3]

        print("\n" + "=" * 60)
        print("🎉 【最终结构化JSON输出】")
        print("=" * 60)
        print(json_str)
        print("=" * 60)

        # 把JSON里的final_reply也加入历史，方便下一轮对话
        try:
            parsed_json = json.loads(json_str)
            messages.append(SystemMessage(content=parsed_json["final_reply"]))
        except:
            pass