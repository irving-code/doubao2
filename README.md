📊 线性回归数据分析智能体 (LangGraph + MCP) 基于 LangGraph 构建的专业线性回归分析智能体，支持自然语言输入自动完成数据拟合、趋势预测和专业知识检索，同时提供 MCP 服务接口，可无缝集成到各类 AI 客户端。  

✨ 核心功能 

* 自动线性回归拟合：支持自然语言描述数据，自动完成最小二乘法拟合与下一个值预测 

* 可视化生成：自动生成拟合趋势图并保存到本地 

* 专业知识库检索：内置线性回归、最小二乘法相关专业知识问答 

* 多轮对话支持：基于 LangGraph 状态持久化，支持上下文连续对话 

* 全链路日志记录：异步日志系统，同时输出到文件和控制台 

* MCP 标准接口：支持 MCP 协议，可在 Claude Desktop、Cursor 等客户端中直接调用 

* 数据持久化：自动保存所有对话消息和工具调用记录到本地 SQLite 数据库 

🚀 快速开始

1. 环境准备

- Python 版本：3.10 ~ 3.12（推荐 3.11，兼容性最好）
- 操作系统：Windows /macOS/ Linux 全平台支持

1. 安装依赖

**克隆项目到本地**

git clone https://github.com/你的用户名 / 你的仓库名.git

cd 你的仓库名

**（推荐）创建 Anaconda 虚拟环境（指定推荐的 Python 3.11 版本）**

conda create -n doubao python=3.11

**Windows /macOS/ Linux 通用：激活虚拟环境**

conda activate doubao

**安装所有项目依赖**

pip install -r requirements.txt

## 配置 API 密钥



本项目使用.env 文件管理敏感的 API 信息，无需修改代码，配置步骤如下：

① 在项目根目录中，找到名为.env.example 的模板文件

② 将该文件复制一份，并重命名为.env（注意文件名前面有个点，Windows 系统会自动隐藏，正常编辑即可）

③ 用记事本或 VS Code 打开.env 文件，填入你的火山引擎方舟 API 信息：

ENDPOINT_ID = 你的模型端点 ID

API_KEY = 你的 API 密钥

API_URL=https://ark.cn-beijing.volces.com/api/v3

⚠️ 重要安全提示

- .env 文件包含你的私密 API 密钥，绝对不要把这个文件分享给别人，也不要提交到 GitHub 等代码仓库
- 项目已经自动配置了.gitignore 文件，会忽略.env 文件，不用担心误提交
- 如果不小心泄露了 API 密钥，请立即去火山引擎控制台删除并重新生成新的密钥

### 补充

- 如果你使用其他支持 OpenAI 兼容接口的大模型，只需修改上面三个字段对应的值即可
- 验证配置是否成功：运行程序后，控制台没有出现 "API 密钥无效" 或 "连接失败" 的报错，就说明配置正确
- 模型端点 ID 可以在火山引擎方舟控制台的 "模型服务 - 端点管理" 中找到
