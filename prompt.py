from codeScanner import db
import requests
from typing import List, Dict

# 使用SonarQube结果改进提示词
def get_quality_hints(project_key: str, max_issues: int = 5) -> str:
    """
    从SonarQube获取当前项目质量问题
    
    :param project_key: SonarQube项目key
    :param max_issues: 最大返回问题数
    :return: 格式化质量提示
    """
    # SonarQube API配置
    sonar_url = "http://your-sonar-server/api/issues/search"
    params = {
        "componentKeys": project_key,
        "statuses": "OPEN",
        "ps": max_issues,
        "additionalFields": "rules"
    }
    
    try:
        response = requests.get(sonar_url, params=params, auth=("token", "your_sonar_token"))
        issues = response.json().get("issues", [])
        
        if not issues:
            return "✅ 当前SonarQube未检测到显著质量问题"
            
        hints = []
        for issue in issues:
            rule = issue["rule"].split(":")[-1]
            hints.append(
                f"⚠️ {rule}: {issue['message']}\n"
                f"   文件: {issue['component'].split(':')[1]}\n"
                f"   行: {issue['line']} (严重度: {issue['severity']})"
            )
        
        return "SonarQube最新质量问题报告：\n" + "\n".join(hints)
    
    except Exception as e:
        return f"❌ 无法获取SonarQube报告: {str(e)}"


def build_context(requirement: str, sonar_project_key: str = "your_project_key") -> str:
    """
    构建包含代码上下文和质量提示的完整提示词
    
    :param requirement: 用户需求描述
    :param sonar_project_key: SonarQube项目标识
    :return: 多维度上下文字符串
    """
    # 从向量库检索相关代码
    code_context = "\n".join(
        f"// 相关代码 {i+1}:\n{doc.page_content[:800]}" 
        for i, doc in enumerate(db.similarity_search(requirement, k=3))
    )
    
    # 获取质量提示
    quality_hints = get_quality_hints(sonar_project_key)
    
    # 组合完整上下文
    return f"""
    === 代码上下文 ===
    {code_context}
    
    === 质量改进建议 ===
    {quality_hints}
    
    === 需求实现要求 ===
    请基于以上参考代码，特别注意：
    1. 避免重复出现质量提示中的问题
    2. 保持与现有代码风格一致
    3. 优先使用项目中的工具类
    """