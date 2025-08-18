from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 配置各模块加载路径（Windows路径示例）
# 自动补全为绝对路径
base_dir = Path(r"D:\workspace\ai\auto.coder\auto.coder")
module_paths = [
    str(base_dir / "api" / "service" / "src" / "main" / "java"),
    str(base_dir / "core" / "src" / "main" / "java"),
    str(base_dir / "pojo" / "src" / "main" / "java")
]
# 验证路径存在
valid_paths = [p for p in module_paths if os.path.exists(p)]

# 合并所有模块的文档
all_docs = []
for path in module_paths:
    loader = DirectoryLoader(
        path, 
        glob="**/*.java",
        # 排除测试代码
        exclude=["**/test/**", "**/mock/**"],
        loader_cls=lambda f: TextLoader(f, encoding='utf-8')
    )
    loaded_docs = loader.load()
    for doc in loaded_docs:
        doc.metadata.setdefault('source', str(Path(path)))
    all_docs.extend(loaded_docs)

# 加载项目构建配置
pom_loader = DirectoryLoader(
    r"D:\workspace\ai\auto.coder",
    glob="**/pom.xml",
    loader_cls=lambda f: TextLoader(f, encoding='utf-8')
)
pom_loaded_docs = pom_loader.load()
for doc in pom_loaded_docs:
    doc.metadata.setdefault('source', str(Path(path)))
all_docs.extend(pom_loaded_docs) 

# 添加重要配置文件
config_loader = DirectoryLoader(
    r"D:\workspace\ai\auto.coder\auto.coder\api\service\src\main\resources",
    glob="**/application*.yml",
    loader_cls=lambda f: TextLoader(f, encoding='utf-8')
)
config_loaded_docs = config_loader.load()
for doc in config_loaded_docs:
    doc.metadata.setdefault('source', str(Path(path)))
all_docs.extend(config_loaded_docs)

# 生成模块关系描述（可选）
module_desc = """
项目模块依赖关系：
1. api -> core
2. api/service -> core
3. core -> pojo
4. api/service 是控制器模块
5. core 是业务处理模块
6. pojo 是数据库实体类以及实体模型模块
"""
all_docs.append(Document(page_content=module_desc))

# 建议进行文档分块（Java代码建议配置）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    # 保持Java包声明完整
    separators=["\npackage ", "\nimport ", "\npublic class ", "\n\n"]
)
split_docs = text_splitter.split_documents(all_docs)


# 使用轻量级本地嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="GanymedeNil/text2vec-large-chinese",
    model_kwargs={'device': 'cpu'}  # 可改为'cuda'加速
)

# 创建带元数据的向量库
db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)