shuyaming部分：
1.创建的文件
HKBUStudyCompanion/src/rag_engine.py

2.实现的功能
实现 HKBU Study Companion 项目的上下文工程部分，包括文档加载、索引构建、词法检索和神经检索功能。

3.调用
安装依赖
pip install ollama numpy
下载模型
ollama pull nomic-embed-text
调用
from src.rag_engine import RAGEngine

# 初始化（只需一次）
engine = RAGEngine(chunk_file="chunks_sliding_500_50.jsonl")

# 词法检索
context, chunks = engine.lexical_search("你的问题", top_k=3)

# 神经检索
context, chunks = engine.neural_search("你的问题", top_k=3)

# 对比两种检索器
result = engine.compare_retrievers("你的问题", top_k=3)

