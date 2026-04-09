# ===================== 【一级模块：依赖库导入】 =====================
import streamlit as st
import json
import uuid
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
# 导入我们完善的 agent_graph
from finalplus1 import agent_graph

# ===================== 【一级模块：Streamlit页面基础配置】 =====================
st.set_page_config(
    page_title="线性回归分析智能体",
    page_icon="📊",
    layout="wide"
)

st.title("📊 线性回归分析智能体")
st.divider()

# ===================== 【一级模块：会话状态初始化】 =====================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "final_result" not in st.session_state:
    st.session_state.final_result = None
if "image_path" not in st.session_state:
    st.session_state.image_path = None

# ===================== 【一级模块：对话历史展示】 =====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===================== 【一级模块：用户输入交互】 =====================
# ===================== 【二级模块：用户输入框渲染】 =====================
user_input = st.chat_input("请输入你的问题，例如：我的数据是[(1,10),(2,20),(3,30)]，帮我预测下一个值")

# ===================== 【二级模块：用户输入处理逻辑】 =====================
if user_input:
    # 1. 添加用户消息到对话历史
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 执行Agent工作流
    with st.chat_message("assistant"):
        with st.status("🔄 智能体处理中...", expanded=True) as status:
            # 初始化状态（与 finalplus1.py 完全一致）
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "session_id": st.session_state.session_id,
                "user_input": user_input,
                "structured_result": {
                    "fitting_equation": "无",
                    "prediction_result": "无",
                    "image_path": "无"
                },
                "tool_calls": [],
                "error_msg": None
            }
            config = RunnableConfig(configurable={"thread_id": st.session_state.session_id})

            # 用invoke进行传入参数
            # 对于入门场景，invoke 比 stream 更不容易出错，能直接拿到最终 state
            final_state = agent_graph.invoke(initial_state, config=config)

            # 模拟展示执行过程（提升用户体验）
            st.write("✅ 【思考节点】完成")
            if len(final_state.get("tool_calls", [])) > 0:
                st.write(f"✅ 【工具执行节点】完成，共执行{len(final_state['tool_calls'])}个工具")
                for tool in final_state["tool_calls"]:
                    # 修复后的代码：用 get 方法安全取值，兼容 name 和 tool_name 两种键
                    tool_name = tool.get("tool_name") or tool.get("name") or "未知工具"
                    tool_args = tool.get("args", {})
                    st.write(f"   - 工具：{tool_name}，入参：{tool_args}")
            st.write("✅ 【结果生成节点】完成")

            status.update(label="🎉 处理完成！", state="complete", expanded=False)

        # 3. 解析最终结果
        if final_state:
            # ========== 【修复2：直接从 final_state 拿结构化结果】 ==========
            # 方案 A（推荐）：直接用 structured_result，它在 finalplus1.py 里已经更新好了
            structured_result = final_state["structured_result"]

            # 方案 B（兜底）：如果需要完整的 final_json，从 messages 里找（但尽量用方案 A）
            final_json = None
            for msg in reversed(final_state["messages"]):
                if msg.type == "system":
                    try:
                        final_json = json.loads(msg.content)
                        break
                    except:
                        continue

            # 合并结果（优先用 final_json，兜底用 structured_result）
            if final_json:
                st.session_state.final_result = final_json
                st.session_state.image_path = final_json["image_path"]
                reply_content = final_json["final_reply"]
            else:
                # 手动构建兜底结果
                st.session_state.final_result = structured_result
                st.session_state.image_path = structured_result["image_path"]
                reply_content = f"处理完成，拟合方程：{structured_result['fitting_equation']}，预测结果：{structured_result['prediction_result']}"

            # 展示最终回复
            st.markdown(reply_content)
            # 添加到对话历史
            st.session_state.messages.append({"role": "assistant", "content": reply_content})

# ===================== 【一级模块：侧边栏功能实现】 =====================
# ===================== 【二级模块：侧边栏结构化结果展示】 =====================
with st.sidebar:
    st.header("📋 线性回归预测与相关知识解释")
    if st.session_state.final_result:
        res = st.session_state.final_result
        st.metric("用户意图", res.get("user_intent", "未知"))
        st.metric("预测结果", res.get("prediction_result", "无"))
        st.text_input("拟合方程", res.get("fitting_equation", "无"), disabled=True)
        st.text_input("图片路径", res.get("image_path", "无"), disabled=True)

        # 拟合图片预览
        if st.session_state.image_path and st.session_state.image_path != "无":
            st.divider()
            st.subheader("📈 拟合结果图")
            try:
                st.image(st.session_state.image_path, width="stretch")
            except Exception as e:
                st.warning(f"图片加载失败：{str(e)}")

    st.divider()
    st.caption(f"会话ID：{st.session_state.session_id}")
    # ===================== 【二级模块：新建会话按钮逻辑】 =====================
    if st.button("🔄 新建会话"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.final_result = None
        st.session_state.image_path = None
        st.rerun()