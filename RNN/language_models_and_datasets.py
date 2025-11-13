import random
import torch
from d2l_utils import read_time_machine, tokenize, Vocab, load_corpus_time_machine, plot, seq_data_iter_random, seq_data_iter_sequential



if __name__ == '__main__':

    #  自然语言统计
    tokens = tokenize(read_time_machine())
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus)
    print(vocab.token_freqs[:10])

    # 正如我们所看到的，最流行的词看起来很无聊， 这些词通常被称为停用词（stop words），因此可以被过滤掉
    freqs = [freq for token, freq in vocab.token_freqs]
    # plot(freqs, xlabel='token: x', ylabel='frequency: y', xscale='log', yscale='log')

    # 我们来看看二元语法的频率是否与一元语法的频率表现出相同的行为方式。
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    print('-'*60)
    print('二元语法')
    for item in bigram_vocab.token_freqs[:10]:
        print(item)

    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    print('-'*60)
    print('三元语法')
    for item in trigram_vocab.token_freqs[:10]:
        print(item)

    # 最后，我们直观地对比三种模型中的词元频率
    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    # plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x', ylabel='frequency: y', xscale='log', yscale='log', legend=['unigram', 'bigram', 'trigram'])

    # 读取长序列数据

    # 当序列变得太长而不能被模型一次性全部处理时， 我们可能希望拆分这样的序列方便模型读取。
    # 随机采样（random sampling）和 顺序分区（sequential partitioning）策略。

    # 随机采样
    print('-'*60)
    print('随机采样')
    my_seq = list(range(35))
    for X,Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

    # 顺序分区
    print('-'*60)
    print('顺序分区')
    for X,Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)