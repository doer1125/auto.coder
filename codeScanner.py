import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置路径
base_dir = Path(r"D:\workspace\raising\lj.xc.ids")
module_paths = [
    str(base_dir / "api" / "service" / "src" / "main" / "java"),
    str(base_dir / "api" / "fast" / "src" / "main" / "java"),
    str(base_dir / "core" / "src" / "main" / "java"),
    str(base_dir / "fast" / "src" / "main" / "java"),
    str(base_dir / "pojo" / "src" / "main" / "java")
]

# 验证路径存在
valid_paths = [p for p in module_paths if os.path.exists(p)]


class LocalCodeEmbeddings:
    """本地代码嵌入模型 - 专门为代码优化"""

    def __init__(self, model_name="microsoft/codebert-base"):
        self.model_name = model_name
        self.dimension = 768  # 标准嵌入维度

        try:
            self._initialize_model()
            print(f"成功加载本地代码嵌入模型: {model_name}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("使用轻量级备选模型...")
            self._initialize_fallback()

    def _initialize_model(self):
        """初始化CodeBERT模型"""
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.use_transformers = True

    def _initialize_fallback(self):
        """初始化备选模型"""
        self.use_transformers = False
        print("使用基于特征的轻量级嵌入")

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        embeddings = []

        if self.use_transformers:
            # 使用transformers模型
            return self._embed_with_transformers(texts)
        else:
            # 使用基于特征的轻量级方法
            return self._embed_with_features(texts)

    def _embed_with_transformers(self, texts: List[str]) -> List[List[float]]:
        """使用transformers模型生成嵌入"""
        import torch

        embeddings = []
        batch_size = 8  # 减小批次大小避免内存问题

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用平均池化
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

                    for embedding in batch_embeddings:
                        embeddings.append(embedding.tolist())

                print(f"已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文档")

            except Exception as e:
                print(f"处理批次失败: {e}")
                # 为失败的批次生成fallback嵌入
                for text in batch:
                    embeddings.append(self._generate_feature_embedding(text))

        return embeddings

    def _embed_with_features(self, texts: List[str]) -> List[List[float]]:
        """使用基于特征的方法生成嵌入"""
        embeddings = []

        for i, text in enumerate(texts):
            embedding = self._generate_feature_embedding(text)
            embeddings.append(embedding)

            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{len(texts)} 个文档")

        return embeddings

    def _generate_feature_embedding(self, text: str) -> List[float]:
        """生成基于代码特征的嵌入向量"""
        vector = np.zeros(self.dimension)
        text_lower = text.lower()

        # Java关键字特征
        java_keywords = [
            'class', 'interface', 'void', 'public', 'private', 'protected',
            'static', 'final', 'return', 'import', 'package', 'extends',
            'implements', 'new', 'this', 'super', 'if', 'else', 'for',
            'while', 'do', 'switch', 'case', 'break', 'continue', 'try',
            'catch', 'finally', 'throw', 'throws'
        ]

        # 代码结构特征
        code_patterns = [
            ('{', 0.3), ('}', 0.3), ('(', 0.2), (')', 0.2),
            (';', 0.1), ('=', 0.1), ('==', 0.15), ('!=', 0.15),
            ('<', 0.1), ('>', 0.1), ('<=', 0.15), ('>=', 0.15)
        ]

        # 注解特征（Spring相关）
        annotations = [
            '@controller', '@service', '@repository', '@autowired',
            '@requestmapping', '@getmapping', '@postmapping', '@value'
        ]

        # 填充关键词特征
        for i, keyword in enumerate(java_keywords):
            if i < len(vector):
                count = text_lower.count(keyword)
                vector[i] = min(count / 5.0, 1.0)

        # 填充代码结构特征
        offset = len(java_keywords)
        for i, (pattern, weight) in enumerate(code_patterns):
            idx = offset + i
            if idx < len(vector):
                count = text.count(pattern)
                vector[idx] = min(count * weight, 1.0)

        # 填充注解特征
        offset += len(code_patterns)
        for i, annotation in enumerate(annotations):
            idx = offset + i
            if idx < len(vector):
                count = text_lower.count(annotation)
                vector[idx] = min(count / 2.0, 1.0)

        # 添加文本长度特征
        if len(vector) > offset + len(annotations):
            vector[offset + len(annotations)] = min(len(text) / 2000.0, 1.0)

        # 添加代码行数特征
        line_count = text.count('\n') + 1
        if len(vector) > offset + len(annotations) + 1:
            vector[offset + len(annotations) + 1] = min(line_count / 50.0, 1.0)

        return vector.tolist()


# 文档处理函数（保持不变）
def extract_java_metadata(file_path: str, content: str) -> dict:
    """从Java文件中提取元数据"""
    metadata = {
        'source': file_path,
        'module': Path(file_path).parent.name,
        'package': 'default',
        'type': 'unknown',
        'class_name': 'unknown'
    }

    package_match = re.search(r'^package\s+([\w.]+);', content, re.M)
    if package_match:
        metadata['package'] = package_match.group(1)

    class_match = re.search(r'public\s+(class|interface|enum)\s+(\w+)', content)
    if class_match:
        metadata['type'] = class_match.group(1)
        metadata['class_name'] = class_match.group(2)

    methods = re.findall(r'(public|protected|private)\s+([\w<>\[\]]+)\s+(\w+)\s*\([^)]*\)', content)
    if methods:
        method_names = [name for _, _, name in methods[:3]]
        metadata['methods'] = ', '.join(method_names)

    for key in list(metadata.keys()):
        if isinstance(metadata[key], list):
            metadata[key] = str(metadata[key])
        elif not isinstance(metadata[key], (str, int, float, bool, type(None))):
            metadata[key] = str(metadata[key])

    return metadata


def safe_load_module(path):
    """安全加载模块文档"""
    try:
        loader = DirectoryLoader(
            path,
            glob="**/*.java",
            exclude=["**/test/**", "**/mock/**", "**/target/**"],
            loader_cls=lambda f: TextLoader(f, encoding='utf-8')
        )
        loaded_docs = loader.load()

        valid_docs = []
        for doc in loaded_docs:
            if isinstance(doc, Document):
                doc.metadata.update(extract_java_metadata(doc.metadata['source'], doc.page_content))
                valid_docs.append(doc)

        return valid_docs
    except Exception as e:
        print(f"加载模块 {path} 失败: {e}")
        return []


def filter_complex_metadata_safe(document):
    """安全的元数据过滤函数"""
    if not hasattr(document, 'metadata'):
        return document

    filtered_metadata = {}
    for key, value in document.metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            filtered_metadata[key] = value
        else:
            filtered_metadata[key] = str(value)

    return Document(page_content=document.page_content, metadata=filtered_metadata)


# 主程序
def main():
    print("开始加载文档...")
    all_docs = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(safe_load_module, valid_paths)
        for result in results:
            if result:
                all_docs.extend(result)

    print(f"加载了 {len(all_docs)} 个Java文档")

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\npublic class ", "\nclass ", "\npublic interface ", "\ninterface ",
            "\npublic enum ", "\nenum ", "\n}\n\n", "\n    public ", "\n    protected ",
            "\n    private ", "\n// ========", "\n\n"
        ],
        keep_separator=True
    )

    valid_docs_for_split = [doc for doc in all_docs if isinstance(doc, Document)]
    split_docs = text_splitter.split_documents(valid_docs_for_split)
    print(f"分割为 {len(split_docs)} 个代码块")

    # 过滤复杂元数据
    print("过滤复杂元数据...")
    filtered_docs = []
    for doc in split_docs:
        if isinstance(doc, Document):
            filtered_doc = filter_complex_metadata_safe(doc)
            filtered_docs.append(filtered_doc)

    print(f"过滤后剩余 {len(filtered_docs)} 个文档")

    # 初始化本地嵌入模型
    print("初始化本地代码嵌入模型...")
    try:
        embeddings = LocalCodeEmbeddings("microsoft/codebert-base")
    except Exception as e:
        print(f"初始化失败: {e}")
        print("使用轻量级特征嵌入...")
        embeddings = LocalCodeEmbeddings()  # 使用fallback

    # 创建向量库
    print("开始创建向量库...")
    os.makedirs("./chroma_db_code", exist_ok=True)

    # 分批处理避免内存问题
    batch_size = 20
    db = None
    texts = [doc.page_content for doc in filtered_docs]
    metadatas = [doc.metadata for doc in filtered_docs]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        try:
            if db is None:
                db = Chroma.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_metadatas,
                    persist_directory="./chroma_db_code"
                )
            else:
                db.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )

            print(f"已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文档")

        except Exception as e:
            print(f"处理批次失败: {e}")
            continue

    if db:
        db.persist()
        print("向量库创建完成并已保存!")

        # 测试搜索
        print("\n测试搜索功能...")
        try:
            search_result = db.similarity_search("Controller", k=3)
            for i, result in enumerate(search_result):
                print(f"\n--- 结果 {i + 1} ---")
                print(f"来源: {result.metadata.get('source', '未知')}")
                print(f"类名: {result.metadata.get('class_name', '未知')}")
                print(f"类型: {result.metadata.get('type', '未知')}")
                print(f"内容预览: {result.page_content[:200]}...")
        except Exception as e:
            print(f"搜索测试失败: {e}")
    else:
        print("向量库创建失败")


if __name__ == "__main__":
    main()