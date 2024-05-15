import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import clones


class EncoderDecoder(nn.Module):
    """
    基础的Encoder-Decoder结构。
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # src embedding包含embedding + position embedding 的 sequential list
        self.src_embed = src_embed
        # 同上
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    定义生成器，由linear和softmax组成，最后的投影层
    Define standard linear + softmax generation step.

    d_model: input feature size, which is decoder model output usually 512
    vocab: output size, softmax dimension
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        # ex: [512, 10000] 输出预测线性层，输出维度及预测语言的词典长度，没跑softmax前
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 第一个词: [1, 512] * [512, 11] -> [1, 11] output出维度和tgt词的词典长度一样
        # 第二个词: [1, 512] * [512, 11] -> [1, 11] output出维度和tgt词的词典长度一样，区别在于这里提取的是
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """
    完整的Encoder包含N层，每一层都一样的输入和输出。
    编码器的每层encoder包含Self Attention 子层和 FFNN 子层，每个子层都使用了残差连接，和层标准化(layer-normalization)
    """

    def __init__(self, layer, N):
        super().__init__()
        # 生成N个一模一样的encode层，每层包含一个self-attention + (add * normalize) + FFNN + (add * normalize)
        self.layers = clones(layer, N)
        # 生成layer Normalization层，输入是注意力层的输出，一般是512维，最后一层encoder跑完后还有一层归一化
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        每一层的输入是x和mask
        """
        # x: [1, 10, 512] mask: [1, 1, 10]
        for layer in self.layers:
            x = layer(x, mask)
        # 最终输出和输入维度一样 ex:  x: [1, 10, 512]，最终 output 再 norm 一下
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Construct a layer norm module 层标准化

    features: input dimension, output dimension of multi-head attention, default 512
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        # [512,] 全都是1
        self.a_2 = nn.Parameter(torch.ones(features))
        # [512,] 全都是0
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 对每个单个样本的所有维度特征做归一化，及最后一个维度 -1
        # ex: x: [1, 10, 512] -> mean: [1, 10, 1]
        mean = x.mean(-1, keepdim=True)
        # 同上
        std = x.std(-1, keepdim=True)
        # 输出维度和输入一致，ex: [1, 10, 512]
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    我们称呼每个子层为：sublayer，每个子层的最终输出是 layerNorm(x + sublayer(x))。
    dropout被加在Sublayer上。为了便于进行残差连接，模型中的所有子层以及embedding层产生的输出的维度都为 d_model = 512。
    下面的SublayerConnection类用来处理单个Sublayer的输出，该输出将继续被输入下一个Sublayer

    size: int, input dimension usually 512
    dropout: float, dropout rate
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    每一层encoder都有两个子层。 第一层是一个multi-head self-attention层，第二层是一个简单的全连接前馈网络，
    对于这两层都需要使用SublayerConnection类进行处理。
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        # self multi head attention layer
        self.self_attn = self_attn
        # FFN layer
        self.feed_forward = feed_forward
        # self attention 和 FFN 之后都跟随着 add + normalize
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.
        """
        # 第一个sublayer代表 self-attention + add & normalize
        # 这里根据code来看应该是 embedding + self_attn(norm(embedding))
        # ex: x: [1, 10, 512]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个sublayer代表 FFN + add & normalize
        # 这里根据code来看应该是 embedding + attention + FFN(norm(embedding + attention))
        # 这里的输入 x 都代表的残差链接的原始input x，及多头attention后 + embedding 的结果
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    解码器也是由N = 6 个完全相同的decoder层组成。
    decode的时候是一个词一个词预测，所以 x等于一系列: [1, 1, 512] [1, 2, 512] ... [1, tgt_length, 512]
    """

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # decode: 第一个预测词 x: [1, 1, 10] memory: [1, 10, 512] src_mask: [1, 1, 10] tgt_mask: [1, 1, 1]
        # decode: 第二个预测词 x: [1, 2, 10] memory: [1, 10, 512] src_mask: [1, 1, 10] tgt_mask: [1, 2, 2]
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    单层decoder与单层encoder相比，decoder还有第三个子层，该层对encoder的输出执行attention：即encoder-decoder-attention层，
    q向量来自decoder上一层的输出，k和v向量是encoder最后层的输出向量。与encoder类似，我们在每个子层再采用残差连接，然后进行层标准化。
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        # self multi head attention output
        self.self_attn = self_attn
        # source attention use encoder output as key, value, same structure as self attention
        self.src_attn = src_attn
        # FFN 全连层，包含两个线性转换层，最后输入输出都是 512
        self.feed_forward = feed_forward
        # 每个 attention 和 FFN 之后都跟着一个残差链接 + 归一化，decoder总共有两个attention和一个FFN所以要三个connection层
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Follow Figure 1 (right) for connections.
        """
        # encoder层的输出作为key value 共同作用在 decoder层的
        m = memory
        # 第一个sublayer代表 self-attention + add & normalize，这里是tgt自己和自己的attention，用tgt_mask
        # 这里根据code来看应该是 embedding + self_attn(norm(embedding))
        # ex: 第一个词 x: [1, 1, 512]，第二个词 x: [1, 2, 512]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二个sublayer代表 encoder-decoder-attention + add & normalize，这里是tgt和src的attention，src_mask
        # 这里根据code来看应该是 embedding + tgt_attention + src_attn(norm(embedding + tgt_attention))
        # ex: 第一个词 x: [1, 1, 512] m: [1, 10, 512] src_mask: [1, 1, 10]
        # ex: 第二个词 x: [1, 2, 512] m: [1, 10, 512] src_mask: [1, 1, 10]
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 第三个sublayer代表 FFN + add & normalize
        # 这里根据code来看应该是 embedding + tgt_attn + src_attn + FFN(norm(embedding + tgt_attn + src_attn))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        Multi-head attention允许模型同时关注来自不同位置的不同表示子空间的信息，如果只有一个attention head，向量的表示能力会下降。

        h: int, number of head, default = 8
        d_model: int, output dimension also projection linear layer W_o output, default = 512
        """
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k which is d_model // 8
        self.d_k = d_model // h
        self.h = h
        # 因为我们设置d_v的维度等于d_k的维度，然后d_k的维度等于d_model // 8，
        # 则最终projection线性层等于 [h * d_v, d_model] -> [d_model, d_model]
        # 这里四层linear层分别是三层用于Q, k, v 的 W_q, W_k, W_v和最后一个W_0。每一个都是一样的维度
        # 因为我们assume embedding和每一层model的输出都是d_model维度
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        """
        Compute 'Scaled Dot Product Attention'
        Attention功能可以描述为将query和一组key-value映射到输出，其中query、key、value和输出都是向量。
        输出为value的加权和，其中每个value的权重通过query与相应key的计算得到。
        我们将particular attention称之为“缩放的点积Attention”(Scaled Dot-Product Attention")。
        其输入为query、key(维度是d_k)以及values(维度是d_v)。我们计算query和所有key的点积，然后对每个除以sqrt(d_k),
        最后用softmax函数获得value的权重。
        Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
        这里Q， K， V都已经是X * W 之后的结果，三个维度都是 (batch_size, 句子长度, hidden_dimension)
        这里说明为什么要进行scale缩放通过sqrt(d_k)：对于很大的d_k值, 点积大幅度增长，将softmax函数推向具有极小梯度的区域。
        为了抵消这种影响，我们将点积缩小sqrt(d_k)倍。
        """
        # 拿到输出维度及hidden dimension，ex: query: [1, 8, 10, 64]
        d_k = query.size(-1)
        # (Q * K^T) / sqrt(d_k) 这一步:
        # [batch, head, length, d_k] * [batch, head, d_k, length] --> [batch, head, length, length]
        # encode: [1, 8, 10, 64] * [1, 8, 64, 10] --> [1, 8, 10, 10]，decode tgt attn 和 encode 一样，只是 length 不一样
        # decode src attn: 第一个词 [1, 8, 1, 64] * [1, 8, 64, 10] --> [1, 8, 1, 10]，当前预测词对src每个词的attention score
        # decode src attn: 第二个词 [1, 8, 2, 64] * [1, 8, 64, 10] --> [1, 8, 2, 10]，当前预测词对src每个词的attention score
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 应用mask机制，屏蔽此单词关注到后面的单词，方法是scores的输出中对应mask矩阵中等于0的部分设置成-inf这个是一个极小的数字
            # encode: mask: [1, 1, 1, 10] 全都是 1，mask == 0 转化成 [1, 1, 1, 10] 全都是 false，
            # masked_fill把都是true的部分填入mask的value，也就是说mask原始tensor中为0的部分是我们想mask掉的部分
            # decode 的时候 tgt attention 会 apply mask，src attention 并不会apply，因为预测词可以看到所有src词
            # 当 decode 预测第二个词的时候，mask是 [1, 1, 2, 2] 的上半区为0，下半区为1的矩阵，此时表示第一个预测词只能看到自己不能看到
            # 第二个预测词，也就不能对第二个预测词产生 score，也就是score被一个极小值mask掉
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax转化 [batch, head, length, length] 成每个单词对其它单词的[0 - 1]区间的score
        # encode: [1, 8, 10, 10]，decode tgt attn 和 encode一样 ex: 第二个预测词 [1, 8, 2, 2] 只是长度不一样
        # decode src attention: 第一个词 [1, 8, 1, 10]， 第二个词 [1, 8, 2, 10]
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            # 如果有dropout，则apply
            p_attn = dropout(p_attn)
        # p_attn * v 这一步: [batch, head, length, length] * [batch, head, length, d_k] --> [batch, head, length, d_k]
        # encode: [1, 8, 10, 10] * [1, 8, 10, 64] --> [1, 8, 10, 64]
        # 最终得到每个词对其它每个词的value值的每一个维度上的加权和
        # decode tgt attention: 第二个词: [1, 8, 2, 2] * [1, 8, 2, 64] -> [1, 8, 2, 64]
        # decode src attention: [1, 8, 1, 10] * [1, 8, 10, 64] -> [1, 8, 1, 64]
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        query, key, value为没有线性转换之前的原始值，及embedding的输出结果或者上一层的output值
        三个维度都是一样 [batch, length, d_model]
        """
        # ex: query, key, value: [1, 10, 512]
        if mask is not None:
            # Same mask applied to all h heads.
            # encode: mask: [1, 1, 10] -> [1, 1, 1, 10] 转化维度和后续多头一致
            # decode: mask: [1, 1, 1] -> [1, 1, 1, 1] 第一个需要预测的词，[1, 1, 2, 2] 第二个需要预测的词
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # apply 线性转化后，每个Q，K，V都拆分成了 [batch, length, head, d_k] 转化成 [batch, head, length, d_k]
        # 这里转化的目的是方便后续多个头平行计算 attention score，并且后续可以直接多头concat起来
        # 这里 linear 是一个四个线性层的list，但是因为q, k, v只有三个，所以 zip 后只用到前三个线性层，最后一个线性层用于投影保持维度一致
        # 这里等价于 linear[0] * query，linear[1] * key，linear[2] * value 并统一转化维度
        # encode: query, key, value: [1, 10, 512] -> [1, 8, 10, 64]
        # decode tgt attn: query, key, value: [1, 1, 512] -> [1, 8, 1, 64] 第一个词, [1, 8, 2, 64] 第二个词
        # decode src attn: query, key, value: [1, 8, 1, 64], [1, 8, 10, 64], [1, 8, 10, 64] 第一个词
        # decode src attn: query, key, value: [1, 8, 2, 64], [1, 8, 10, 64], [1, 8, 10, 64] 第二个词
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # 计算每个头的attention结果 x -> [batch, head, length, d_k]   attn -> [batch, head, length, length]
        # encode: x: [1, 8, 10, 64]  attn: [1, 8, 10, 10]
        # decode tgt attn: 第一个词 x: [1, 8, 1, 64] attn: [1, 8, 1, 1], 第二个词 x: [1, 8, 2, 64] attn: [1, 8, 2, 2]
        # decode src attn: 第一个词 x: [1, 8, 1, 64] attn: [1, 8, 1, 10], 第二个词 x: [1, 8, 2, 64] attn: [1, 8, 2, 10]
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # 把attention之后的x转化成 [batch, length, head, d_k] 再把多头全部拼起来转化为 [batch, length, head * d_k -> d_model]
        # encode: x: [1, 8, 10, 64] -> [1, 10, 8, 64] -> [1, 10, 512]
        # decode src attention: 第一个词 x: [1, 8, 1, 64] -> [1, 1, 8, 64]] -> [1, 1, 512]
        # decode src attention: 第二个词 x: [1, 8, 2, 64] -> [1, 2, 8, 64]] -> [1, 2, 512]
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        # 最终再跑一个线性层 [batch, length, d_model] * [d_model, d_model] -> [batch, length, d_model]
        # encode: [1, 10, 512] * [512, 512] -> [1, 10, 512]
        # decode src attention: [1, 1, 512] * [512, 512] -> [1, 1, 512]
        # 保证每一层的encode输出都是一样的为d_model
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    两层线性变换，但它们在层与层之间使用不同的参数。输入和输出的维度都是 d_model = 512, 内层维度是 d_ff。
    （也就是第一层输入512维,输出2048维；第二层输入2048维，输出512维），并且在两个线性变换中间有一个ReLU激活函数。

    d_model: int, FFN output dimension also FFN input dimension, default = 512
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # [512, 2048]
        self.w_1 = nn.Linear(d_model, d_ff)
        # [2048, 512]
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # encode: relu([1, 10, 512] * [512, 2048]) * [2048, 512] -> [1, 10, 512]
        # decode src attention: 第一个词 relu([1, 1, 512] * [512, 2048]) * [2048, 512] -> [1, 1, 512]
        # decode src attention: 第二个词 relu([1, 2, 512] * [512, 2048]) * [2048, 512] -> [1, 2, 512]
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """
    输入嵌入层，同时是encoder的输入
    并且在embedding层中最终的输出，我们还需要将这些权重乘以 sqrt(d_model)

    d_model: int, output of embedding also as input of encoder
    vocab: int, 整个训练数据集词典的大小
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        # ex: [10000, 512]
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # ex: lut: [10000, 512] x: [1, 10] -> [1, 10, 512] 执行简单的每个词的 lookup
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    位置编码embedding，用来表示每个词的相对位置在一个句子里面。
    其中pos是位置，i是维度。也就是说，位置编码的每个维度对应于一个正弦或者余弦曲线。及每个词在同一个维度上是一个正弦或者余弦曲线。
    用三角函数版本，是因为它可能允许模型外推到，比训练时遇到的序列更长的序列。及predict阶段可以位置编码同一个维度下更长的句子。
    这里position encoding后的结果并不需要被train，是一开始初始化后固定死了。

    d_model: int, pe的输出维度，和embedding维度一样，所以可以直接叠加之后
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # pe -> [5000, 512] 一个batch下的长度和embedding维度
        pe = torch.zeros(max_len, d_model)
        # 构建位置的index，先生成一个 0 - 4999 的tensor，再unsqueeze变成 [5000, 1]的维度
        position = torch.arange(0, max_len).unsqueeze(1)
        # 先计算分母部分
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # [0, 2, 4, 6, 8, ...] 为 sin 函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # [1, 3, 5, 7, 9, ...] 为 cos 函数
        pe[:, 1::2] = torch.cos(position * div_term)
        # 也就是说偶数 embedding 维度位置为 sin，奇数 embedding 维度位置为 cos
        # 添加batch维度 [5000, 512] -> [1, 5000, 512]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 这里 x 是 embedding的output，输入进来直接进行叠加，position并不需要被 train，requires_grad设置为false
        # 这里我们需要提取出和input进来的embedding的长度一样的部分进行直接叠加，我们提前设置pe最长长度为5000
        # encode: ex: x: [1, 10, 512], pe: [1, 5000, 512] -> [1, 10, 512] x.size(1) 提取单个样本句子长度，之后再叠加
        # decode: ex: x: [1, 1, 512] 第一个词，[1, 2, 512] 第二个词
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
