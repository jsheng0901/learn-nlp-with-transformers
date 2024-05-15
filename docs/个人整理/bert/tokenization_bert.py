import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    # 读取已经设置好的词典的地址文件
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    # ex: tokens = ['青\n', '面\n', '風\n', '龸\n', 'ﬁ\n', 'ﬂ\n', '！\n', '，\n', '－\n', '．\n', '／\n',  'were\n']
    # bert base uncased 有 30522 个词，每个词是是一个tokens里面的元素
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        # ex: [('[PAD]', 0), ('[unused0]', 1), ....] 建立词和index的关系
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 清除基本的之前clean里面留下的正常的空字符，并返回一个list
    text = text.strip()
    if not text:
        return []
    # ex: text --> 'i like natural language progressing!'
    # ex: tokens --> ['i', 'like', 'natural', 'language', 'progressing!']
    tokens = text.split()
    return tokens


class BertTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 构建词到index的有序字典
        self.vocab = load_vocab(vocab_file)
        # 构建反向的index到词的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            # 初始化基础的tokenizer
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 初始化wordpiece的tokenizer
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        # 先做基本的tokenizer
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 这里是对basic tokenizer后的每个词再进行 wordpiece tokenizer，ex: 'i', 'like', ...
                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 如果此token不是不能split的，再跑 wordpiece tokenizer，把token词拆成sub_word，分别放进split_tokens
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 最终拿到的是经过两次 tokenize 之后的 split list
        # ex: text --> 'I like natural language progressing!'
        # ex: split_tokens --> ['i', 'like', 'natural', 'language', 'progressing', '!']
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        # 如果只有一个句子输入
        # ex: token_ids_1 --> ['i', 'like', 'natural', 'language', 'progressing', '!']
        # ex: output [101, 1045, 2066, 3019, 2653, 27673, 999, 102], 101 102 分别是 cls 和 sep
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果只有一个句子输入，则全部都是0
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，前半句是0后半句是1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


class BasicTokenizer(object):

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.
        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        BasicTokenizer负责处理的第一步——按标点、空格等分割句子，并处理是否统一小写，以及清理非法字符。
        - 对于中文字符，通过预处理（加空格）来按字分割；
        - 同时可以通过never_split指定对某些词不进行分割；
        - 这一步是可选的（默认执行）。
        """
        # union() returns a new set by concatenating the two sets.
        # ex: {'[CLS]', '[MASK]', '[PAD]', '[SEP]', '[UNK]'} 合并两个 never split 的 list
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 基本地清理text，包括移除空白字符和不合理的字符
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # ex: ['i', 'like', 'natural', 'language', 'progressing!']
        orig_tokens = whitespace_tokenize(text)
        # ex: ['i', 'like', 'natural', 'language', 'progressing', '!']
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))
        # 重新join起来成text后再去跑一次空白字符移除，这里标点符号和字母直接会加上空白，所以不用担心会把标点符号重新和字母合并
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # ex: ['i', 'like', 'natural', 'language', 'progressing', '!'] 输出还是token后的list
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 去除重音符合
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 对标点符号进行拆分，这里如果把标点符合加入 never_split，则此标点符号不会和字符拆分开
        if never_split is not None and text in never_split:
            return [text]
        # ex: 'progressing!' -> ['p', 'r', 'o', 'g', 'r', 'e', 's', 's', 'i', 'n', 'g', '!'] 对每个字母进行拆分
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # check 是否是标点符号
            if _is_punctuation(char):
                # 如果是标点符号，标点符号自己成为一个list
                # ex： [['p', 'r', 'o', 'g', 'r', 'e', 's', 's', 'i', 'n', 'g'], ['!']]
                output.append([char])
                # 标记之后的字母从新开始一个list
                start_new_word = True
            else:
                # 如果需要从新构建一个词，此时先加入一个空的list，之后把所有的字母加入这个list，
                # 只有第一个字母或者遇到标点符号之后，才会从新开一个list，也就是拆分标点符号在一个text里面每个拆分开的单独组成一个list
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        # 从新把list里面每个字母组合成text
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        # 清理不合理的字符和移除空白在text里面
        # ex: text --> 'i like natural language progressing!'
        output = []
        for char in text:
            cp = ord(char)
            # 查看是否是人为定义的空白字符，认为定义的空字符或者不合理的字符，直接跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 查看是否是直接的空白字符，ex: " ", "\t", "\n"，如果是则加入output
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # ex: output: ['i', ' ', 'l', 'i', 'k', 'e', ' ', 'n', 'a', 't', 'u', 'r', 'a', 'l', ' ',
        # 'l', 'a', 'n', 'g', 'u', 'a', 'g', 'e', ' ', 'p', 'r', 'o', 'g', 'r', 'e', 's', 's', 'i', 'n', 'g', '!']
        # 每个字母都拆分开
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        WordPieceTokenizer在词的基础上，进一步将词分解为子词（sub-word）。
        - sub-word 介于 char 和 word 之间，既在一定程度保留了词的含义，又能够照顾到英文中单复数、时态导致的词表爆炸
          和未登录词的 OOV（Out-Of-Vocabulary）问题，将词根与时态词缀等分割出来，从而减小词表，也降低了训练难度；
        - 例如，tokenizer 这个词就可以拆解为“token”和“##izer”两部分，注意后面一个词的“##”表示接在前一个词后面。
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            # ex: token --> 'like'
            # ex: chars --> ['l', 'i', 'k', 'e']
            chars = list(token)
            # 判断这个token也就这个词是不是超过长度，如果是则设置为 UNK
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            # loop 整个 chars
            while start < len(chars):
                # 初始化end指针为最右边
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    # 当start指针大于0的时候说明前面的部分已经被拆分开，此时用 '##'表示当前词是前面一个词的后缀部分
                    if start > 0:
                        substr = "##" + substr
                    # 如果这个 sub_string 在词典里面，则说明我们可以找到分词
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    # 如果找不到分词，end指针向左走，直到找到一个分词在词典里面
                    end -= 1
                # 如果loop完整个chard都找不到这个词，说明是一个UNK词，bad flag设置为 True
                if cur_substr is None:
                    is_bad = True
                    break
                # 更新start指针等于end指针，前面的部分已经加入cur_substr
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
