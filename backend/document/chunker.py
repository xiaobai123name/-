"""
智能文档分块器
实现父子文档索引、滑动窗口、递归字符切分等策略
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .parser import ParsedDocument


@dataclass
class ChunkData:
    """切片数据结构"""
    id: str
    document_id: str
    content: str
    chunk_index: int
    chunk_type: str  # "parent" or "child"
    parent_chunk_id: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    section_title: Optional[str] = None
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartChunker:
    """智能文档分块器"""
    
    # 不同文档类型的默认分块参数
    DEFAULT_PARAMS = {
        "pdf": {
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
            "chunk_overlap": 50,
        },
        "docx": {
            "parent_chunk_size": 2500,
            "child_chunk_size": 600,
            "chunk_overlap": 60,
        },
        "md": {
            "parent_chunk_size": 1500,
            "child_chunk_size": 400,
            "chunk_overlap": 40,
        },
        "txt": {
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
            "chunk_overlap": 50,
        }
    }
    
    # 递归切分的分隔符优先级
    SEPARATORS = [
        "\n\n\n",  # 多个空行（章节分隔）
        "\n\n",    # 段落分隔
        "\n",      # 行分隔
        "。",      # 中文句号
        "！",      # 中文感叹号
        "？",      # 中文问号
        ".",       # 英文句号
        "!",       # 英文感叹号
        "?",       # 英文问号
        "；",      # 中文分号
        ";",       # 英文分号
        "，",      # 中文逗号
        ",",       # 英文逗号
        " ",       # 空格
        ""         # 字符级别
    ]
    
    def __init__(
        self,
        parent_chunk_size: Optional[int] = None,
        child_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        初始化分块器
        
        Args:
            parent_chunk_size: 父切片大小（字符数）
            child_chunk_size: 子切片大小（字符数）
            chunk_overlap: 切片重叠大小（字符数）
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _get_params(self, file_type: str) -> Dict[str, int]:
        """获取指定文件类型的分块参数"""
        defaults = self.DEFAULT_PARAMS.get(file_type, self.DEFAULT_PARAMS["pdf"])
        return {
            "parent_chunk_size": self.parent_chunk_size or defaults["parent_chunk_size"],
            "child_chunk_size": self.child_chunk_size or defaults["child_chunk_size"],
            "chunk_overlap": self.chunk_overlap or defaults["chunk_overlap"],
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的Token数量
        中文约1.5字符/token，英文约4字符/token
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
    
    def _recursive_split(
        self, 
        text: str, 
        max_size: int, 
        overlap: int,
        separators: Optional[List[str]] = None
    ) -> List[str]:
        """
        递归字符切分
        
        按分隔符优先级切分文本，保持语义完整性
        """
        if separators is None:
            separators = self.SEPARATORS.copy()
        
        if not text or len(text) <= max_size:
            return [text] if text.strip() else []
        
        # 尝试当前分隔符
        if not separators:
            # 没有更多分隔符，强制按字符切分
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + max_size, len(text))
                chunks.append(text[start:end])
                start = end - overlap if end < len(text) else end
            return chunks
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator and separator in text:
            # 使用当前分隔符切分
            parts = text.split(separator)
            chunks = []
            current_chunk = ""
            
            for part in parts:
                # 如果添加这部分会超过最大大小
                test_chunk = current_chunk + separator + part if current_chunk else part
                
                if len(test_chunk) <= max_size:
                    current_chunk = test_chunk
                else:
                    # 保存当前chunk
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # 如果单个部分超过最大大小，递归处理
                    if len(part) > max_size:
                        sub_chunks = self._recursive_split(part, max_size, overlap, remaining_separators)
                        chunks.extend(sub_chunks[:-1] if sub_chunks else [])
                        current_chunk = sub_chunks[-1] if sub_chunks else ""
                    else:
                        current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # 添加重叠
            if overlap > 0 and len(chunks) > 1:
                chunks = self._add_overlap(chunks, overlap)
            
            return chunks
        else:
            # 当前分隔符不存在，尝试下一个
            return self._recursive_split(text, max_size, overlap, remaining_separators)
    
    def _add_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """为切片添加重叠部分"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            # 从前一个chunk取overlap长度的后缀
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk
            overlapped_chunks.append(overlap_text + chunks[i])
        
        return overlapped_chunks
    
    def _extract_page_info(self, chunk_content: str, pages: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
        """从分页信息中提取切片对应的页码范围"""
        if not pages:
            return None, None
        
        start_page = None
        end_page = None
        
        for page in pages:
            page_content = page.get("content", "")
            page_num = page.get("page_number") or page.get("section_index", 0)
            
            # 检查切片内容是否在该页中
            if chunk_content[:100] in page_content or page_content[:100] in chunk_content:
                if start_page is None:
                    start_page = page_num
                end_page = page_num
        
        return start_page, end_page
    
    def chunk_document(
        self, 
        document: ParsedDocument, 
        document_id: str
    ) -> Tuple[List[ChunkData], List[ChunkData]]:
        """
        对文档进行父子双层切片
        
        Args:
            document: 解析后的文档
            document_id: 文档ID
            
        Returns:
            Tuple[List[ChunkData], List[ChunkData]]: (父切片列表, 子切片列表)
        """
        params = self._get_params(document.file_type)
        
        parent_chunks = []
        child_chunks = []
        
        # 第一步：创建父切片
        parent_texts = self._recursive_split(
            document.content,
            params["parent_chunk_size"],
            0  # 父切片不需要重叠
        )
        
        for i, parent_text in enumerate(parent_texts):
            if not parent_text.strip():
                continue
            
            parent_id = str(uuid.uuid4())
            start_page, end_page = self._extract_page_info(parent_text, document.pages)
            
            parent_chunk = ChunkData(
                id=parent_id,
                document_id=document_id,
                content=parent_text,
                chunk_index=i,
                chunk_type="parent",
                parent_chunk_id=None,
                start_page=start_page,
                end_page=end_page,
                section_title=self._extract_section_title(parent_text),
                token_count=self._estimate_tokens(parent_text),
                metadata={
                    "file_type": document.file_type,
                    "file_hash": document.file_hash
                }
            )
            parent_chunks.append(parent_chunk)
            
            # 第二步：将父切片切分为子切片
            child_texts = self._recursive_split(
                parent_text,
                params["child_chunk_size"],
                params["chunk_overlap"]
            )
            
            for j, child_text in enumerate(child_texts):
                if not child_text.strip():
                    continue
                
                child_chunk = ChunkData(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=child_text,
                    chunk_index=i * 1000 + j,  # 保持全局顺序
                    chunk_type="child",
                    parent_chunk_id=parent_id,
                    start_page=start_page,
                    end_page=end_page,
                    section_title=parent_chunk.section_title,
                    token_count=self._estimate_tokens(child_text),
                    metadata={
                        "file_type": document.file_type,
                        "parent_index": i,
                        "child_index": j
                    }
                )
                child_chunks.append(child_chunk)
        
        return parent_chunks, child_chunks
    
    def _extract_section_title(self, text: str) -> Optional[str]:
        """从文本中提取章节标题"""
        lines = text.strip().split('\n')
        if not lines:
            return None
        
        first_line = lines[0].strip()
        
        # 检查是否是Markdown标题
        if first_line.startswith('#'):
            return first_line.lstrip('#').strip()
        
        # 检查是否是短行（可能是标题）
        if len(first_line) < 50 and not first_line.endswith(('。', '.', '，', ',')):
            return first_line
        
        return None
    
    def chunk_to_dict(self, chunk: ChunkData) -> Dict[str, Any]:
        """将ChunkData转换为字典（用于数据库存储）"""
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "chunk_type": chunk.chunk_type,
            "parent_chunk_id": chunk.parent_chunk_id,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "section_title": chunk.section_title,
            "token_count": chunk.token_count,
            "extra_metadata": chunk.metadata
        }
