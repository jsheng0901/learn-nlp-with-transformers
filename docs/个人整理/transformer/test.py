from trainer import make_model
import torch
from utils import subsequent_mask


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    # [1, 10] 表示一个样本，每个位置的单词的index，及一个长度为10个词的句子，转化每个词为对应词典里面的index后的tensor
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # [1, 1, 10] 全是1
    src_mask = torch.ones(1, 1, 10)
    # 先跑model里面的encode部分，最终输出 memory: [1, 10, 512] 维度没变对比embedding后
    memory = test_model.encode(src, src_mask)
    # [1, 1] 一个0，第一个需要预测的词用index为0的词开始预测，一般0index在词典里面是一个special token代表开始翻译。
    ys = torch.zeros(1, 1).type_as(src)
    # predict 输出 一个词接着一个词
    for i in range(9):
        # out 第一个词: [1, 1, 512]，第二个词: [1, 2, 512]
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # prob 第一个词: [1, 11]，第二个词: [1, 11]
        # 注意这里第二个词的out是[1, 2, 512]也就是两个词都有512维度的输出，我们需要的是第二个词的预测，
        # 所以out[:, -1]提取出的是[1, 512]，这里拿到的是第二个词的512维度的输出，再去跑generator算出对应最大的概率词在整个预测词典中
        prob = test_model.generator(out[:, -1])
        # 第一个输出是prob的最大值本身，next_word是prob中最大值的index
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # concat SPE, 第一个, 第二个词 ... 预测出来的词，ys: [0, 10, 4, ...]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    # 这里输出是一个开始翻译的SPE token + 9个词对应在预测词典的index。每一次epoch 都要跑一次encode，然后跑对应预测词数量的decode。
    # ex: 一次的预测输出 [ 0,  9,  1, 10, 10, 10, 10, 10, 10, 10]
    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


if __name__ == "__main__":
    run_tests()
