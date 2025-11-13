"""
文本预处理示例
演示如何读取和处理时间机器数据集，以及如何构建词表
"""

import collections
import re
import torch
from d2l_utils import read_time_machine, tokenize, count_corpus, Vocab, load_corpus_time_machine


if __name__ == '__main__':
    # ========== 第一部分：读取和分词 ==========
    print("=" * 50)
    print("第一部分：读取和分词")
    print("=" * 50)
    
    # 读取时间机器数据集
    lines = read_time_machine()
    print(f'文本行数: {len(lines)}')
    print(f'第一行: {lines[0]}')
    print(f'第二行: {lines[1]}')
    
    # 将文本分词（按单词）
    tokens = tokenize(lines, 'word')
    print(f'\n分词后的前ji行:')
    for i in range(10):
        print(f'  {tokens[i]}')
    
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])

    # 现在，我们可以将每一条文本行转换成一个数字索引列表。

    for i in [0,10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])
        print('-' * 50)

    corpus, vocab = load_corpus_time_machine()
    print(len(corpus), len(vocab))
