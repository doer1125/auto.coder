from langchain.embeddings import HuggingFaceEmbeddings
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from typing import List
import numpy as np
import torch


# 转换为LangChain兼容格式
class ModelScopeEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_id: str):
        self.pipeline = pipeline(
            task=Tasks.text_embedding,
            model=model_id,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """处理文档列表"""
        max_length = 512  # 根据模型最大长度调整
        results = []
        for text in texts:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            embeddings = []
            for chunk in chunks:
                output = self.pipeline(chunk)
                emb = output["text_embedding"] if "text_embedding" in output else output[0]
                embeddings.append(emb.tolist() if isinstance(emb, np.ndarray) else emb)
            # 对分块结果取平均
            results.append(np.mean(embeddings, axis=0).tolist())
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """处理单条查询"""
        output = self.pipeline(text)
        emb = output["text_embedding"] if "text_embedding" in output else output[0]
        return emb.tolist() if isinstance(emb, np.ndarray) else emb

