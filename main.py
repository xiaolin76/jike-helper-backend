from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_qianfan import QianfanChatEndpoint
import sqlite3
import os
from dotenv import load_dotenv

# 加载环境变量（存放大模型API密钥）
load_dotenv()

# 初始化FastAPI应用
app = FastAPI(title="计科学习助手API")

# 解决跨域问题（前端能访问后端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 上线后可指定前端域名，毕设阶段用*即可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化大模型（通义千问）
llm = QianfanChatEndpoint(
    model="ERNIE-3.5-8K-lite",  # 轻量化模型，免费额度足够
    api_key=os.getenv("QIANFAN_API_KEY"),
    secret_key=os.getenv("QIANFAN_SECRET_KEY"),
)

# 初始化数据库（创建表）
def init_db():
    conn = sqlite3.connect("jike_helper.db")
    c = conn.cursor()
    # 知识点问答记录
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge_qa
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    # 代码讲解记录
    c.execute('''CREATE TABLE IF NOT EXISTS code_explain
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, explain TEXT, create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    # 错题分析记录
    c.execute('''CREATE TABLE IF NOT EXISTS wrong_question
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, analysis TEXT, create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# 启动时初始化数据库
init_db()

# 数据模型（请求参数）
class KnowledgeQA(BaseModel):
    question: str  # 知识点问题（如：数据结构中链表的插入逻辑）

class CodeExplain(BaseModel):
    code: str  # 待讲解的代码（如：快速排序代码）
    course: str  # 所属课程（如：数据结构）

class WrongQuestion(BaseModel):
    question: str  # 错题内容
    course: str  # 所属课程

# 1. 知识点问答接口
@app.post("/api/knowledge/qa")
async def knowledge_qa(data: KnowledgeQA):
    try:
        # 构造大模型提示词（针对计科课程优化）
        prompt = f"""
        你是计科专业学习助手，针对{data.question}这个问题，按以下要求回答：
        1. 用通俗易懂的语言解释核心知识点，适配计科本科学生理解；
        2. 结合课程（数据结构/计组/操作系统）给出典型应用场景；
        3. 避免过于学术化的表述，重点突出考点和易错点。
        """
        # 调用大模型
        response = llm.invoke(prompt)
        answer = response.content

        # 保存到数据库
        conn = sqlite3.connect("jike_helper.db")
        c = conn.cursor()
        c.execute("INSERT INTO knowledge_qa (question, answer) VALUES (?, ?)", (data.question, answer))
        conn.commit()
        conn.close()

        return {"code": 200, "msg": "success", "data": {"answer": answer}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识点问答失败：{str(e)}")

# 2. 代码讲解接口
@app.post("/api/code/explain")
async def code_explain(data: CodeExplain):
    try:
        # 构造大模型提示词（代码讲解专属）
        prompt = f"""
        你是计科专业代码讲解助手，针对{data.course}课程的这段代码：
        {data.code}
        按以下要求讲解：
        1. 逐行解释代码逻辑；
        2. 分析时间/空间复杂度（如果适用）；
        3. 指出常见错误和优化方向；
        4. 用mermaid流程图语法输出代码执行流程（仅输出流程图代码，无需其他内容）。
        """
        # 调用大模型
        response = llm.invoke(prompt)
        explain = response.content

        # 保存到数据库
        conn = sqlite3.connect("jike_helper.db")
        c = conn.cursor()
        c.execute("INSERT INTO code_explain (code, explain) VALUES (?, ?)", (data.code, explain))
        conn.commit()
        conn.close()

        return {"code": 200, "msg": "success", "data": {"explain": explain}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"代码讲解失败：{str(e)}")

# 3. 错题分析接口
@app.post("/api/wrong/question")
async def wrong_question(data: WrongQuestion):
    try:
        # 构造大模型提示词（错题分析专属）
        prompt = f"""
        你是计科专业错题分析助手，针对{data.course}课程的这道错题：
        {data.question}
        按以下要求分析：
        1. 指出错误原因和对应的知识点漏洞；
        2. 给出正确解法/答案；
        3. 总结该知识点的考试高频考点；
        4. 输出该知识点所属的知识模块（用于可视化）。
        """
        # 调用大模型
        response = llm.invoke(prompt)
        analysis = response.content

        # 保存到数据库
        conn = sqlite3.connect("jike_helper.db")
        c = conn.cursor()
        c.execute("INSERT INTO wrong_question (question, analysis) VALUES (?, ?)", (data.question, analysis))
        conn.commit()
        conn.close()

        return {"code": 200, "msg": "success", "data": {"analysis": analysis}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"错题分析失败：{str(e)}")

# 测试接口（验证服务是否正常）
@app.get("/")
async def root():
    return {"msg": "计科学习助手后端服务运行正常"}

# 启动服务（本地测试）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)