from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import ModelScopeEmbeddings
from concurrent.futures import ThreadPoolExecutor

# 配置各模块加载路径（Windows路径示例）
# 自动补全为绝对路径

base_dir = Path(r"D:\workspace\PuDongWJ\lj-xc-pudong-dsfsjycl\lj.xc.pudong.dsfsjycl")
module_paths = [
    str(base_dir / "api" / "service" / "src" / "main" / "java"),
    str(base_dir / "api" / "fast" / "src" / "main" / "java"),
    str(base_dir / "core" / "src" / "main" / "java"),
    str(base_dir / "fast" / "src" / "main" / "java"),
    str(base_dir / "pojo" / "src" / "main" / "java")
]

# 验证路径存在
valid_paths = [p for p in module_paths if os.path.exists(p)]

# 合并所有模块的文档
all_docs = []

def extract_java_metadata(file_path: str, content: str) -> dict:
    """从Java文件中提取元数据"""
    metadata = {
        'source': file_path,
        'module': Path(file_path).parent.name,
        'package': 'default'
    }
    
    # 提取包声明
    package_match = re.search(r'^package\s+([\w.]+);', content, re.M)
    if package_match:
        metadata['package'] = package_match.group(1)
    
    # 提取类/接口定义
    class_match = re.search(r'public\s+(class|interface)\s+(\w+)', content)
    if class_match:
        metadata['type'] = class_match.group(1)
        metadata['class'] = class_match.group(2)
    
    return metadata

def load_module(path):
    loader = DirectoryLoader(path, glob="**/*.java", loader_cls=TextLoader)
    return loader.load()

# for path in module_paths:
#     loader = DirectoryLoader(
#         path, 
#         glob="**/*.java",
#         # 排除测试代码
#         exclude=["**/test/**", "**/mock/**"],
#         loader_cls=lambda f: TextLoader(f, encoding='utf-8')
#     )
#     loaded_docs = loader.load()
#     for doc in loaded_docs:
#         doc.metadata.update(extract_java_metadata(doc.metadata['source'], doc.page_content))
#     all_docs.extend(loaded_docs)

with ThreadPoolExecutor(max_workers=4) as executor:
    all_docs = list(executor.map(load_module, valid_paths))

# 加载项目构建配置
pom_loader = DirectoryLoader(
    base_dir,
    glob="**/pom.xml",
    loader_cls=lambda f: TextLoader(f, encoding='utf-8')
)
pom_loaded_docs = pom_loader.load()
for doc in pom_loaded_docs:
    doc.metadata.update(extract_java_metadata(doc.metadata['source'], doc.page_content))
all_docs.extend(pom_loaded_docs) 

# 添加重要配置文件
resourcePath = str(base_dir / "api" / "service" / "src" / "main" / "resources")
config_loader = DirectoryLoader(
    resourcePath,
    glob="**/application*.yml",
    loader_cls=lambda f: TextLoader(f, encoding='utf-8')
)
config_loaded_docs = config_loader.load()
for doc in config_loaded_docs:
    doc.metadata.update(extract_java_metadata(doc.metadata['source'], doc.page_content))
all_docs.extend(config_loaded_docs)

print(all_docs[0].metadata)

# 生成模块关系描述（可选）
module_desc = """
项目模块依赖关系：
1. api -> core
2. api/service -> core
3. api/fast -> core
4. core -> pojo
5. api/service 是控制器模块
6. core 是业务处理模块
7. pojo 是数据库实体类以及实体模型模块
"""
all_docs.append(Document(page_content=module_desc))

# 建议进行文档分块（Java代码建议配置）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=[
        "\npublic class ",  # 类级分割
        "\nprotected class ",
        "\nprivate class ",
        "\npublic interface ",
        "\n}\n",  # 代码块结束
        "\n//",    # 注释分割
        "\n"
    ],
    keep_separator=True  # 保留分隔符
)
split_docs = text_splitter.split_documents(all_docs)


# 使用轻量级本地嵌入模型
embeddings = ModelScopeEmbeddings(
    model_id='codegeex2-6b'
)

# 创建带元数据的向量库
def safe_persist_vectorstore(docs, embeddings, path):
    """安全持久化向量库"""
    try:
        if os.path.exists(path):
            # 验证已有库的兼容性
            existing_db = Chroma(persist_directory=path, embedding_function=embeddings)
            if existing_db._collection.count() > 0:
                dim = len(embeddings.embed_query("test"))
                if existing_db._collection.metadata.get('dimension') != dim:
                    raise ValueError(f"维度不匹配: 已有{existing_db._collection.metadata['dimension']}D, 新模型{dim}D")
        
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=path
        )
        db._collection.metadata = {'dimension': len(embeddings.embed_query("test"))}
        db.persist()
        return db
    except Exception as e:
        print(f"持久化失败: {str(e)}")
        # 回退到临时内存库
        return Chroma.from_documents(docs, embeddings)
    
db = safe_persist_vectorstore(split_docs, embeddings, "./chroma_db")

# 测试嵌入维度一致性
assert len(db._collection.metadata['dimension']) == len(embeddings.embed_query("test"))

# 检查典型类是否完整存储
search_result = db.similarity_search("public class MainController", k=1)
assert "class MainController" in search_result[0].page_content