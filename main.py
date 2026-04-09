from langchain.tools import tool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 加载配置（你已经验证过连通的配置） =====================
load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("ENDPOINT_ID"),
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_URL"),
    temperature=0.3,
    timeout=60,
)


# ===================== 2. 封装calculate工具（使用你提供的微软雅黑字体配置） =====================
@tool
def calculate(data_points: str | list):
    """
    对输入的(x,y)数据点执行最小二乘法线性回归拟合，预测下一个x对应的y值，生成本地拟合图片。
    【使用场景】：用户提供序列数据、询问分数/数值趋势、预测下一个数值时，必须调用这个工具。
    【入参要求】：data_points必须是[(x1,y1), (x2,y2), ...]格式的列表，或该格式的JSON字符串，x为从1开始递增的序列数字，y为对应的数值。
    【返回值】：拟合结果、预测值、拟合方程、图片保存路径
    """

    # -------------------------- 你的字体配置（已放在最前面，全局生效） --------------------------
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # -------------------------- 输入格式统一处理 --------------------------
    try:
        if isinstance(data_points, str):
            data_points = json.loads(data_points.strip())
        data = np.array(data_points)
        x = data[:, 0]
        y = data[:, 1]
        n = len(x)
        if n < 2:
            return "错误：至少需要2个数据点才能完成线性拟合"
    except Exception as e:
        return f"输入解析失败：{str(e)}"

    # -------------------------- 纯numpy实现最小二乘法 --------------------------
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_square = np.sum(x ** 2)
    denominator = n * sum_x_square - sum_x ** 2

    if denominator == 0:
        return "错误：所有x值相同，无法完成线性拟合"

    w = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - w * sum_x) / n

    # -------------------------- 预测下一个x的值 --------------------------
    next_x = x[-1] + 1
    predict_y = w * next_x + b

    # -------------------------- 生成拟合图并保存到本地 --------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="steelblue", s=80, label="原始数据点")
    # 绘制拟合直线
    x_fit = np.linspace(x.min(), next_x, 100)
    y_fit = w * x_fit + b
    plt.plot(x_fit, y_fit, color="coral", linewidth=2, label=f"拟合直线: y = {w:.2f}x + {b:.2f}")
    # 标注预测点
    plt.scatter(next_x, predict_y, color="red", s=100, marker="*", label=f"预测点(x={next_x}, y={predict_y:.2f})")
    # 图表美化
    plt.xlabel("X序列")
    plt.ylabel("Y值")
    plt.title("线性回归趋势拟合与预测")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 保存图片
    plt.savefig("linear_fit_result.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------- 结构化返回结果 --------------------------
    return {
        "fitting_equation": f"y = {w:.4f}x + {b:.4f}",
        "next_x": int(next_x),
        "predict_y": round(float(predict_y), 2),
        "image_path": "linear_fit_result.png",
        "message": f"拟合完成，下一次的预测值为{predict_y:.2f}"
    }


# ===================== 3. 绑定工具并测试 =====================
# 把工具绑定到大模型
model_with_tools = llm.bind_tools([calculate])

# 直接用考核示例的问题测试
if __name__ == "__main__":
    user_question = "我这几次模拟考的分数分别是：第一次450，第二次470，第三次485，第四次500。帮我看看这个趋势，如果不发生意外，我第五次大概能考多少？"
    print("用户提问：", user_question)

    # 调用大模型，验证它能不能识别到要调用工具
    response = model_with_tools.invoke(user_question)
    print("\n大模型生成的工具调用指令：")
    print(response.tool_calls)