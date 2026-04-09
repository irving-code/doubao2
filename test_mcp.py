import subprocess
import json
import sys

def test_mcp_service():
    # ==================== 配置区域：这里改成你自己的脚本名 ====================
    YOUR_SCRIPT_NAME = ("finalplus1.py")  # 这里改成你的主脚本文件名
    # =========================================================================

    print("="*60)
    print(f" 正在启动 MCP 服务: {YOUR_SCRIPT_NAME} --mcp")
    print("="*60)

    try:
        # 启动你的 MCP 服务进程
        # 启动你的 MCP 服务进程
        process = subprocess.Popen(
            [sys.executable, YOUR_SCRIPT_NAME, "--mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',  # 【关键修改】强制用 UTF-8 读取输出
            bufsize=1
        )

        # 1. 发送初始化请求（这是 MCP 协议的标准第一步）
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }

        print("\n [1/4] 发送初始化请求...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()

        # 读取初始化响应
        init_response = process.stdout.readline()
        print(f" 初始化成功: {json.loads(init_response)['result']['serverInfo']['name']}\n")

        # 2. 发送初始化完成通知
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        process.stdin.write(json.dumps(initialized_notification) + "\n")
        process.stdin.flush()

        # 3. 请求工具列表
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }

        print("📡 [2/4] 获取工具列表...")
        process.stdin.write(json.dumps(list_tools_request) + "\n")
        process.stdin.flush()

        tools_response = process.stdout.readline()
        tools = json.loads(tools_response)['result']['tools']
        print(" 可用工具:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description'][:50]}...")

        # 4. 测试调用一个简单的工具（知识检索）
        call_tool_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "retrieve_knowledge_mcp",
                "arguments": {
                    "query": "什么是线性回归？"
                }
            }
        }

        print("\n📡 [3/4] 测试调用知识检索工具...")
        process.stdin.write(json.dumps(call_tool_request) + "\n")
        process.stdin.flush()

        call_response = process.stdout.readline()
        result = json.loads(call_response)['result']['content'][0]['text']
        print(f" 工具返回结果:\n{result}")

        # 5. 测试调用计算工具
        call_calc_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "calculate_mcp",
                "arguments": {
                    "data_points": "[(1,2), (2,4), (3,6)]"
                }
            }
        }

        print("\n📡 [4/4] 测试调用计算工具...")
        process.stdin.write(json.dumps(call_calc_request) + "\n")
        process.stdin.flush()

        calc_response = process.stdout.readline()
        calc_result = json.loads(calc_response)['result']['content'][0]['text']
        print(f" 计算工具返回结果:\n{calc_result}")

        print("\n" + "="*60)
        print(" 所有测试通过！你的 MCP 服务完全正常！")
        print("="*60)

        # 优雅关闭服务
        process.terminate()

    except Exception as e:
        print(f"\n 测试失败: {str(e)}")
        # 打印错误日志帮助排查
        if process.stderr:
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"\n 服务错误日志:\n{stderr_output}")
        process.terminate()

if __name__ == "__main__":
    test_mcp_service()