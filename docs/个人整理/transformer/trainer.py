import copy
import time

from model import *
import torch.nn as nn


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Helper: Construct a model from hyperparameters.
    从超参数到完整模型的函数

    src_vocab: int, encoder input vocab size
    tgt_vocab: int, decoder output vocab size
    n: int, number of encoder and decoder layer
    d_model: int, hidden layer output dimension
    d_ff: int, FFN layer inner output dimension
    h: int, number of head
    dropout: float, dropout rate
    """
    c = copy.deepcopy

    # 构建多头 attention 层
    attn = MultiHeadedAttention(h, d_model)
    # 构建 FFN 层
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 构建位置编码层
    position = PositionalEncoding(d_model, dropout)

    # encoder 层
    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    # decoder 层，这里 self_attn 和 src_attn 都是 multi head attention model
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    # source embedding 层，先构建 embedding，embedding输出是position的输入，再构建 position 层
    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    # target embedding 层，同上
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    # 最后的投影层，预测输出单词的层
    generator = Generator(d_model, tgt_vocab)
    # stack 起来每一层
    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    # 初始化所有参数的分布，很重要，会影响收敛的速度和梯度消失或者爆炸的问题
    # 这里拿到model里面所有需要被train的参数，并初始化维度大于1的参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def run_epoch(data_iter, model, loss_compute):
    """
    Standard Training and Logging Function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # 拿到model的output，及softmax之后的结果
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # 计算 loss
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # 每50个epoch的下一个epoch时输出一下当前的结果
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    # 计算平均loss对每个token词
    return total_loss / total_tokens
