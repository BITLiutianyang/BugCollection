import requests
import time
from typing import Optional, List, Dict, Any

class GitHubAPI:
    def __init__(self, token: Optional[str] = None, base_url: str = "https://api.github.com"):
        """
        初始化 GitHub API 客户端
        
        :param token: GitHub 个人访问令牌 (推荐提供)
        :param base_url: GitHub API 基础 URL，默认为官方 API
        """
        self.base_url = base_url
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-API-Client"
        }
        
        if token:
            self.headers["Authorization"] = f"token {token}"
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """
        发送 GET 请求到 GitHub API
        
        :param endpoint: API 端点
        :param params: 请求参数
        :return: 请求响应对象
        """
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params)
        
        # 检查GitHub API 速率限制
        if response.status_code == 403:
            limit = int(response.headers.get('X-RateLimit-Remaining', 0))
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time())) - time.time()
            
            if limit <= 0:
                print(f"达到 API 速率限制，等待 {reset_time:.1f} 秒后重试")
                time.sleep(reset_time)
                response = self.session.get(url, params=params)
        
        response.raise_for_status()
        return response
    
    
    def _get2(self, url) -> requests.Response:
        """
        发送 GET 请求到 GitHub API
        """
        response = self.session.get(url)
        
        # 检查GitHub API 速率限制
        if response.status_code == 403:
            limit = int(response.headers.get('X-RateLimit-Remaining', 0))
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time())) - time.time()
            
            if limit <= 0:
                print(f"达到 API 速率限制，等待 {reset_time:.1f} 秒后重试")
                time.sleep(reset_time)
                response = self.session.get(url, params=params)
        
        response.raise_for_status()
        return response
    
    def _get_all_pages(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        获取所有分页数据
        
        :param endpoint: API 端点
        :param params: 请求参数
        :return: 合并后的所有数据
        """
        all_data = []
        current_page = 1
        
        while True:
            if params is None:
                current_params = {'page': current_page}
            else:
                current_params = params.copy()
                current_params['page'] = current_page
            
            current_params.setdefault('per_page', 100)  # 每页获取最大数量
            
            response = self._get(endpoint, current_params)
            data = response.json()
            
            if not data:
                break
            
            all_data.extend(data)
            current_page += 1
            
            # 如果没有下一页链接，则结束循环
            if 'Link' not in response.headers or 'rel="next"' not in response.headers['Link']:
                break
        
        return all_data
    
    def get_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        获取仓库基本信息
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :return: 仓库信息字典
        """
        return self._get(f"repos/{owner}/{repo}").json()
    
    def get_branches(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        获取仓库所有分支
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :return: 分支列表
        """
        return self._get_all_pages(f"repos/{owner}/{repo}/branches")
    
    def get_commits(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        获取仓库提交历史
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :param branch: 分支名称，默认为 main
        :param since: 起始日期 (ISO 8601 格式)
        :return: 提交历史列表
        """
        # params = {'sha': branch}
        # if since:
        #     params['since'] = since
        
        return self._get_all_pages(f"repos/{owner}/{repo}/commits")
    
    def get_commit(self, owner: str, repo: str, commit_sha: str) -> Dict[str, Any]:
        """
        获取特定提交详情
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :param commit_sha: 提交的 SHA 哈希值
        :return: 提交详情字典
        """
        return self._get(f"repos/{owner}/{repo}/commits/{commit_sha}").json()
    
    def get_file_content(self, owner: str, repo: str, file_path: str, ref: str = 'main') -> str:
        """
        获取文件内容
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :param file_path: 文件路径
        :param ref: 分支或提交哈希
        :return: 文件内容字符串
        """
        content = self._get(f"repos/{owner}/{repo}/contents/{file_path}", params={'ref': ref}).json()
        
        if 'content' in content:
            # GitHub 返回的 content 是 base64 编码的
            import base64
            return base64.b64decode(content['content']).decode('utf-8')
        else:
            raise Exception(f"无法获取文件内容: {content.get('message', '未知错误')}")
        
    def get_file_content_with_url(self, url) -> str:
        """
        获取文件内容
        """
        content = self._get2(url).json()
        
        if 'content' in content:
            # GitHub 返回的 content 是 base64 编码的
            import base64
            return base64.b64decode(content['content']).decode('utf-8')
        else:
            raise Exception(f"无法获取文件内容: {content.get('message', '未知错误')}")
    
    def get_directory_content(self, owner: str, repo: str, dir_path: str, ref: str = 'main') -> List[Dict[str, Any]]:
        """
        获取目录内容
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :param dir_path: 目录路径
        :param ref: 分支或提交哈希
        :return: 目录内容列表
        """
        return self._get(f"repos/{owner}/{repo}/contents/{dir_path}", params={'ref': ref}).json()
    
    def get_issues(self, owner: str, repo: str, state: str = 'all') -> List[Dict[str, Any]]:
        """
        获取仓库 issues
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :param state: 问题状态 (open, closed, all)，默认为 all
        :return: issues 列表
        """
        return self._get_all_pages(f"repos/{owner}/{repo}/issues", {'state': state})
    
    def get_pull_requests(self, owner: str, repo: str, state: str = 'all') -> List[Dict[str, Any]]:
        """
        获取仓库 pull requests
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :param state: PR 状态 (open, closed, merged, all)，默认为 all
        :return: PR 列表
        """
        return self._get_all_pages(f"repos/{owner}/{repo}/pulls", {'state': state})
    
    def get_contributors(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        获取仓库贡献者
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :return: 贡献者列表
        """
        return self._get_all_pages(f"repos/{owner}/{repo}/contributors")
    
    def search_repos(self, query: str, sort: str = 'best match', order: str = 'desc') -> List[Dict[str, Any]]:
        """
        搜索功能
        
        :param query: 搜索功能字符串
        :param sort: 排序字段 (stars, forks, help-wanted-issues, updated)
        :param order: 排序方向 (asc, desc)
        :return: 搜索结果列表
        """
        return self._get_all_pages(
            "search/repositories",
            {
                'q': query,
                'sort': sort,
                'order': order
            }
        )['items']
    
    def get_repo_events(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        获取仓库事件
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :return: 事件列表
        """
        return self._get_all_pages(f"repos/{owner}/{repo}/events")
    
    def get_repo_commits_diff(self, owner: str, repo: str, sha: str) -> Dict[str, Any]:
        """
        获取特定提交的 diff 信息
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :param sha: 提交的 SHA 哈希值
        :return: diff 信息
        """
        return self._get(f"repos/{owner}/{repo}/commits/{sha}").json()

    def get_repo_languages(self, owner: str, repo: str) -> Dict[str, int]:
        """
        获取仓库使用的编程语言统计
        
        :param owner: 仓库拥有者
        :param repo: 仓库名称
        :return: 语言统计字典
        """
        return self._get(f"repos/{owner}/{repo}/languages").json()#  
