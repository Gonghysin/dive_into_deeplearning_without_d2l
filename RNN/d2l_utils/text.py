"""
文本处理模块
"""
import random
import re
import torch
from .download import DATA_HUB, DATA_URL, download


# 注册时间机器数据集
DATA_HUB['time_machine'] = (
    DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中
    
    功能：
    1. 下载并读取《时间机器》文本
    2. 将每行文本转换为小写
    3. 移除所有非字母字符（保留空格）
    4. 去除首尾空格
    
    返回:
        list: 处理后的文本行列表
    
    示例:
        >>> lines = read_time_machine()
        >>> print(f'文本行数: {len(lines)}')
        >>> print(lines[0])
    """
    # 下载数据集
    file_path = download('time_machine')
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行：
    # 1. re.sub('[^A-Za-z]+', ' ', line): 将所有非字母字符替换为空格
    # 2. strip(): 去除首尾空格
    # 3. lower(): 转换为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元
    
    参数:
        lines: 文本行列表
        token: 词元类型，'word' 表示单词，'char' 表示字符
    
    返回:
        list: 词元列表的列表
    
    示例:
        >>> lines = ['hello world', 'machine learning']
        >>> tokenize(lines, 'word')
        [['hello', 'world'], ['machine', 'learning']]
        >>> tokenize(lines, 'char')
        [['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'], ...]
    """
    if token == 'word':
        # 按单词分割
        return [line.split() for line in lines]
    elif token == 'char':
        # 按字符分割
        return [list(line) for line in lines]
    else:
        raise ValueError(f"未知的词元类型: {token}。请使用 'word' 或 'char'")


def count_corpus(tokens):
    """统计词元的频率
    
    参数:
        tokens: 词元列表的列表，或单个词元列表
    
    返回:
        collections.Counter: 词元频率统计
    
    示例:
        >>> tokens = [['hello', 'world'], ['hello', 'python']]
        >>> counter = count_corpus(tokens)
        >>> print(counter.most_common(3))
        [('hello', 2), ('world', 1), ('python', 1)]
    """
    from collections import Counter
    
    # 如果 tokens 是二维列表，展平为一维
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    
    return Counter(tokens)


def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表
    
    这个函数将时光机器文本转换为字符级别的词元索引序列。
    
    参数:
        max_tokens: 返回的最大词元数量，-1 表示返回全部
    
    返回:
        corpus: 词元索引列表（一维列表）
        vocab: 词表对象
    
    说明:
        - 使用字符级别分词（'char'），每个字符都是一个词元
        - 将所有文本行展平成一个连续的索引序列
        - 适合用于字符级别的语言模型训练
    
    示例:
        >>> corpus, vocab = load_corpus_time_machine(max_tokens=10000)
        >>> print(f'语料库大小: {len(corpus)}')
        >>> print(f'词表大小: {len(vocab)}')
        >>> print(f'前10个索引: {corpus[:10]}')
    """
    # 读取时光机器数据集
    lines = read_time_machine()
    
    # 按字符分词（字符级别）
    tokens = tokenize(lines, 'char')
    
    # 构建词表
    vocab = Vocab(tokens)
    
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    # corpus 是一个一维的索引列表
    corpus = [vocab[token] for line in tokens for token in line]
    
    # 如果指定了最大词元数，截取前 max_tokens 个
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列
    
    这种方法随机采样子序列，打乱了序列之间的顺序关系。
    每个样本都是从原始序列上随机位置抽取的子序列。
    
    参数:
        corpus: 词元索引列表（一维）
        batch_size: 批量大小
        num_steps: 每个子序列的时间步数
    
    生成:
        X: 输入序列 (batch_size, num_steps)
        Y: 目标序列 (batch_size, num_steps)，比 X 向后偏移一个位置
    
    示例:
        >>> corpus = list(range(100))
        >>> for X, Y in seq_data_iter_random(corpus, 2, 5):
        ...     print(X.shape, Y.shape)
        ...     break
        torch.Size([2, 5]) torch.Size([2, 5])
    """
    # 从随机偏移量开始对序列进行分区，随机范围包括 num_steps-1
    # 这样可以增加随机性
    corpus = corpus[random.randint(0, num_steps - 1):]
    
    # 计算可以划分出多少个长度为 num_steps 的子序列
    num_subseqs = (len(corpus) - 1) // num_steps
    
    # 每个子序列的起始索引：0, num_steps, 2*num_steps, ...
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    
    # 打乱这些起始索引，实现随机采样
    random.shuffle(initial_indices)
    
    def data(pos):
        """返回从 pos 开始的长度为 num_steps 的序列"""
        return corpus[pos: pos + num_steps]
    
    # 计算可以生成多少个完整的批次
    num_batches = num_subseqs // batch_size
    
    # 生成每个批次
    for i in range(0, batch_size * num_batches, batch_size):
        # 获取当前批次的起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        # X: 输入序列，从位置 j 开始
        X = [data(j) for j in initial_indices_per_batch]
        # Y: 目标序列，从位置 j+1 开始（向后偏移1个位置）
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列
    
    这种方法保持了序列的顺序关系。
    将整个序列划分为 batch_size 个子序列，然后按时间步顺序迭代。
    
    参数:
        corpus: 词元索引列表（一维）
        batch_size: 批量大小
        num_steps: 每个子序列的时间步数
    
    生成:
        X: 输入序列 (batch_size, num_steps)
        Y: 目标序列 (batch_size, num_steps)，比 X 向后偏移一个位置
    
    示例:
        >>> corpus = list(range(100))
        >>> for X, Y in seq_data_iter_sequential(corpus, 2, 5):
        ...     print(X.shape, Y.shape)
        ...     break
        torch.Size([2, 5]) torch.Size([2, 5])
    
    说明:
        顺序分区的优点是能保持序列的连续性，适合需要长期依赖的任务。
        缺点是计算效率可能不如随机采样。
    """
    # 从随机偏移量开始，增加一些随机性
    offset = random.randint(0, num_steps)
    
    # 计算可用的词元数量（需要能被 batch_size 整除）
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    
    # 提取输入和目标序列
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    
    # 重塑为 (batch_size, -1) 的形状
    # 每一行是一个独立的子序列
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    
    # 计算可以生成多少个批次
    num_batches = Xs.shape[1] // num_steps
    
    # 按时间步顺序生成批次
    for i in range(0, num_steps * num_batches, num_steps):
        # 从每个子序列中提取长度为 num_steps 的片段
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
    
class SeqDataLoader:
    """序列数据加载器
    
    用于加载和迭代时间序列数据，支持随机采样和顺序分区两种方式。
    
    参数:
        batch_size: 批量大小
        num_steps: 每个序列的时间步数
        use_random_iter: 是否使用随机采样（True）还是顺序分区（False）
        max_tokens: 使用的最大词元数量
    
    属性:
        corpus: 词元索引列表
        vocab: 词表对象
        batch_size: 批量大小
        num_steps: 时间步数
    
    示例:
        >>> data_loader = SeqDataLoader(batch_size=32, num_steps=35, 
        ...                             use_random_iter=False, max_tokens=10000)
        >>> for X, Y in data_loader:
        ...     print(X.shape, Y.shape)
        ...     break
        torch.Size([32, 35]) torch.Size([32, 35])
    """
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        # 根据参数选择数据迭代函数
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        
        # 加载语料库和词表
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        
        # 保存参数
        self.batch_size, self.num_steps = batch_size, num_steps
        self.use_random_iter = use_random_iter
        self.max_tokens = max_tokens
    
    def __iter__(self):
        """返回数据迭代器"""
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    
    def __len__(self):
        """返回数据集大小（近似值）"""
        return len(self.corpus) // self.batch_size // self.num_steps

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """加载时光机器数据集并返回数据迭代器和词表
    
    参数:
        batch_size: 批量大小
        num_steps: 时间步数（序列长度）
        use_random_iter: 是否使用随机采样（默认 False，使用顺序分区）
        max_tokens: 使用的最大词元数量（默认 10000）
    
    返回:
        data_iter: 数据迭代器（SeqDataLoader 对象）
        vocab: 词表对象
    
    示例:
        >>> batch_size, num_steps = 32, 35
        >>> train_iter, vocab = load_data_time_machine(batch_size, num_steps)
        >>> for X, Y in train_iter:
        ...     print(X.shape, Y.shape)  # (32, 35), (32, 35)
        ...     break
    """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
    

class Vocab:
    """文本词表
    
    用于构建词汇表，将词元映射到索引，反之亦然。
    
    参数:
        tokens: 词元列表或词元列表的列表
        min_freq: 最小词频阈值，低于此频率的词元将被忽略
        reserved_tokens: 保留词元列表（如 '<pad>', '<bos>', '<eos>' 等）
    
    属性:
        idx_to_token: 索引到词元的映射列表
        token_to_idx: 词元到索引的映射字典
        token_freqs: 词元频率列表（按频率降序排列）
    
    示例:
        >>> tokens = [['hello', 'world'], ['hello', 'python']]
        >>> vocab = Vocab(tokens, min_freq=1, reserved_tokens=['<pad>'])
        >>> print(len(vocab))  # 词表大小
        >>> print(vocab['hello'])  # 获取 'hello' 的索引
        >>> print(vocab.to_tokens([0, 1, 2]))  # 将索引转换为词元
    """
    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """初始化词表
        
        参数:
            tokens: 词元列表或词元列表的列表
            min_freq: 最小词频，低于此频率的词元不会被加入词表
            reserved_tokens: 保留词元列表（如特殊标记）
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        # 统计词元频率
        counter = count_corpus(tokens)
        
        # 按出现频率排序（从高到低）
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        
        # 初始化词表：未知词元 '<unk>' 的索引为 0
        # 然后是保留词元（如 '<pad>', '<bos>', '<eos>' 等）
        self.idx_to_token = ['<unk>'] + reserved_tokens
        
        # 构建词元到索引的映射
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        
        # 将高频词元添加到词表中
        for token, freq in self._token_freqs:
            # 如果词频低于阈值，跳过（因为已排序，后面的都更低）
            if freq < min_freq:
                break
            # 如果词元还不在词表中，添加它
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        """返回词表大小（词汇量）"""
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        """根据词元获取索引
        
        参数:
            tokens: 单个词元或词元列表
        
        返回:
            int 或 list: 对应的索引或索引列表
            
        示例:
            >>> vocab = Vocab([['hello', 'world']])
            >>> vocab['hello']  # 返回单个索引
            2
            >>> vocab[['hello', 'world', 'unknown']]  # 返回索引列表
            [2, 3, 0]  # 'unknown' 不在词表中，返回 <unk> 的索引 0
        """
        if not isinstance(tokens, (list, tuple)):
            # 单个词元：返回其索引，如果不存在返回 <unk> 的索引
            return self.token_to_idx.get(tokens, self.unk)
        # 词元列表：递归调用，返回索引列表
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        """根据索引获取词元
        
        参数:
            indices: 单个索引或索引列表
        
        返回:
            str 或 list: 对应的词元或词元列表
            
        示例:
            >>> vocab = Vocab([['hello', 'world']])
            >>> vocab.to_tokens(2)  # 返回单个词元
            'hello'
            >>> vocab.to_tokens([0, 2, 3])  # 返回词元列表
            ['<unk>', 'hello', 'world']
        """
        if not isinstance(indices, (list, tuple)):
            # 单个索引：返回对应的词元
            return self.idx_to_token[indices]
        # 索引列表：返回词元列表
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        """未知词元 '<unk>' 的索引（始终为 0）"""
        return 0
    
    @property
    def token_freqs(self):
        """返回词元频率列表（按频率降序排列）
        
        返回:
            list: [(token, freq), ...] 列表
        """
        return self._token_freqs

