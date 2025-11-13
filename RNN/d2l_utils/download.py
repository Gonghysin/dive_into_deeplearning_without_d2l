"""
数据下载和管理模块
"""

import os
import hashlib
import requests
from pathlib import Path


# 数据集中心：存储数据集的 URL 和 SHA-1 哈希值
DATA_HUB = {}

# 数据集基础 URL
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 默认数据目录
DATA_DIR = Path.home() / '.d2l_data'


def download(name, cache_dir=None):
    """下载数据集
    
    参数:
        name: 数据集名称（在 DATA_HUB 中注册）
        cache_dir: 缓存目录（可选），默认为 ~/.d2l_data
    
    返回:
        str: 下载文件的路径
    
    示例:
        >>> DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt', 
        ...                            '090b5e7e70c295757f55df93cb0a180b9691891a')
        >>> file_path = download('time_machine')
    """
    # 检查数据集是否在 DATA_HUB 中
    if name not in DATA_HUB:
        raise ValueError(f"数据集 '{name}' 未在 DATA_HUB 中注册")
    
    url, sha1_hash = DATA_HUB[name]
    
    # 设置缓存目录
    if cache_dir is None:
        cache_dir = DATA_DIR
    else:
        cache_dir = Path(cache_dir)
    
    # 创建缓存目录
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取文件名
    fname = url.split('/')[-1]
    fpath = cache_dir / fname
    
    # 检查文件是否已存在且哈希值正确
    if fpath.exists():
        if sha1_hash and _check_sha1(fpath, sha1_hash):
            print(f"文件已存在: {fpath}")
            return str(fpath)
        else:
            print(f"文件存在但哈希值不匹配，重新下载...")
    
    # 下载文件
    print(f"正在从 {url} 下载...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 写入文件
        with open(fpath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # 验证哈希值
        if sha1_hash and not _check_sha1(fpath, sha1_hash):
            raise RuntimeError(f"下载文件的 SHA-1 哈希值不匹配")
        
        print(f"下载完成: {fpath}")
        return str(fpath)
    
    except Exception as e:
        # 如果下载失败，删除部分下载的文件
        if fpath.exists():
            fpath.unlink()
        raise RuntimeError(f"下载失败: {e}")


def _check_sha1(filename, sha1_hash):
    """检查文件的 SHA-1 哈希值
    
    参数:
        filename: 文件路径
        sha1_hash: 期望的 SHA-1 哈希值
    
    返回:
        bool: 哈希值是否匹配
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)  # 读取 1MB
            if not data:
                break
            sha1.update(data)
    
    return sha1.hexdigest() == sha1_hash


def register_dataset(name, url, sha1_hash=None):
    """注册数据集到 DATA_HUB
    
    参数:
        name: 数据集名称
        url: 数据集 URL
        sha1_hash: SHA-1 哈希值（可选）
    
    示例:
        >>> register_dataset('time_machine', 
        ...                  DATA_URL + 'timemachine.txt',
        ...                  '090b5e7e70c295757f55df93cb0a180b9691891a')
    """
    DATA_HUB[name] = (url, sha1_hash)

