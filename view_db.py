import sqlite3

# 连接数据库（文件就在当前目录下）
DB_FILE = "agent_data.db"
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# 1. 先查看有哪些表
print("=" * 60)
print("📊 数据库中的所有表：")
print("=" * 60)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(f"- {table[0]}")

# 2. 遍历查看每张表的内容
for table_name in [t[0] for t in tables]:
    print("\n" + "=" * 60)
    print(f"📋 表名：{table_name}")
    print("=" * 60)

    # 查看表结构
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print("字段：", [col[1] for col in columns])

    # 查看前10条数据（避免数据太多刷屏）
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 10;")
    rows = cursor.fetchall()
    print(f"\n数据（前10条）：")
    for row in rows:
        print(row)

conn.close()
print("\n" + "=" * 60)
print("✅ 数据库查看完成")
print("=" * 60)