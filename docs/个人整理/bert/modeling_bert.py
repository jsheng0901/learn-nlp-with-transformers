from transformers.models.bert.modeling_bert import *


class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    1. word_embeddings，上文中 sub-word 对应的嵌入。
    2. token_type_embeddings，用于表示当前词所在的句子，辅助区别句子与 padding、句子对间的差异。
    3. position_embeddings，句子中每个词的位置嵌入，用于区别词的顺序。和 transformer 论文中的设计不同，这一块是训练出来的，
       而不是通过 sin cos wave 函数计算得到的固定嵌入。一般认为这种实现不利于拓展性（难以直接迁移到更长的句子中）。
    三个 embedding 不带权重相加，并通过一层 LayerNorm+dropout 后输出，其大小为(batch_size, sequence_length, hidden_size)。
    """

    def __init__(self, config):
        super().__init__()
        # 构建词嵌入 -> [词典长度，embedding维度] -> [30522, 768]，对于padding的位置，输入index是0
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 构建位置嵌入 -> [最长的句子长度，embedding维度] -> [512, 768]，这里是可训练的参数
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 构建词类型的嵌入 -> [词的类型长度，embedding维度] -> [2, 768]
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # 构建LN层 -> [768, ]，对hidden_size做归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 构建dropout，默认参数为0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # input_shape -> [1, 8]
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # 提取对应的position id编码，self.position_ids -> [1, 512]，position_ids -> [1, 8]
        # ex: [0, 1, 2, 3, 4, 5, 6, 7]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        # 第一个embedding，词嵌入的结果
        if inputs_embeds is None:
            # 转化 input id 从[1, 8] -> [1, 8, 30522] 每个词变成对应词典长度的 one-hot encoding
            # 然后 [1, 8, 30522] * [30522, 768] -> [1, 8, 768] 得到每个词对应的初始化的embedding
            inputs_embeds = self.word_embeddings(input_ids)
        # 第二个embedding，类别嵌入的结果
        # 转化 token_type_ids 从[1, 8] -> [1, 8, 2] 每个词变成对应类别长度的 one-hot encoding，这里就两类，属于第一个或者第二个句子
        # 然后 [1, 8, 2] * [2, 768] -> [1, 8, 768] 得到每个词对应的初始化的 token type embedding 也就是类别 embedding
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # 词嵌入和类别嵌入叠加 -> [1, 8, 768]
        embeddings = inputs_embeds + token_type_embeddings
        # 第三个embedding，位置嵌入的结果，注意这里只有absolute类型的位置嵌入才会直接叠加，不然embedding只包含词嵌入和类别嵌入
        if self.position_embedding_type == "absolute":
            # 转化 token_type_ids 从[1, 8] -> [1, 8, 512] 每个词变成对应位置长度的 one-hot encoding，这里位置最长512
            # 然后 [1, 8, 512] * [512, 768] -> [1, 8, 768] 得到每个词对应的初始化的 position embedding 也就是位置 embedding
            position_embeddings = self.position_embeddings(position_ids)
            # 继续和词嵌入类别嵌入的结果叠加 -> [1, 8, 768]
            embeddings += position_embeddings
        # 三种embedding不带权重的叠加之后
        # 进行 LN + dropout 层后直接输出进入encoder
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # 输出为 [batch_size, sequence_length, hidden_size] -> [1, 8, 768]
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 默认值：12个头
        self.num_attention_heads = config.num_attention_heads
        # 默认值：64维度每个头
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 默认值：768维度，所有头拼接起来后
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # query 线性层参数矩阵 -> [768, 768]，对接embedding的output，所以第一个维度768是对应的embedding的output维度
        # 注意这里的 hidden_size 和 all_head_size 在一开始是一样的。至于为什么要看起来多此一举地设置这一个变量，
        # 显然是因为上面的剪枝函数，剪掉几个 attention head 以后 all_head_size 自然就小了，也就不等于hidden_size了，所以需要分开参数
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # key 线性层参数矩阵 -> [768, 768]
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # value 线性层参数矩阵 -> [768, 768]
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # 默认值：0.1
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        # 标记是否是decoder
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 转化成多头机制的shape [1, 8, 768] -> [1, 8, 12, 64]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 转化input到这个size
        x = x.view(new_x_shape)
        # 从新对size进行排序 [1, 8, 12, 64] -> [1, 12, 8, 64]，代表一个mini batch，有12个头，每个头对应句子里面8个词，每个词64维度
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # embedding output hidden_states -> [1, 8, 768] * [768, 768] -> [1, 8, 768]
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        # 这里如果是cross-attention的话，对应的应该是decoder的结构，需要用到之前encoder的output key value
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # key * hidden_states -> [1, 8, 768] * [768, 768] -> [1, 8, 768] -> 转化多头 -> [1, 12, 8, 64]
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # 同上 value 和 key 一模一样输出 [1, 12, 8, 64]
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        # [1, 8, 768] -> 转化多头 -> [1, 12, 8, 64]
        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # query: [1, 12, 8, 64] * key: [1, 12, 8, 64] -> [1, 12, 64, 8]  -> [1, 12, 8, 8] 转化成12个头的 attention score
        # [batch_size, num_attention_heads, sequence_length, sequence_length] 符合多个头单独计算获得 attention score 的形状。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对于不同的position embedding 类型有三种不同的操作
        # absolute：默认值，这部分就不用处理；
        # relative_key：对 key_layer 作处理，将其与这里的positional_embedding和 key 矩阵相乘作为 key 相关的位置编码；
        # relative_key_query：对 key 和 value 都进行相乘以作为位置编码。
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
            # 关于einsum，参考链接 https://rockt.github.io/2018/04/30/einsum
            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        # 这里进行缩放 [1, 12, 8, 8] -> scale / sqrt(64)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # 在这里进行attention mask的应用，这里的attention_mask在BertModel的 get_extended_attention_mask 已经提前计算好了。
            # function里面将原本为 1 的部分变为 0，而原本为 0 的部分（即 padding）变为一个较大的负数
            # 得到的attention mask 里面 0代表我们希望attend到的位置， -10000代表我们不希望attend的位置，也就是mask。
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # 这里进行对attention score 的 softmax -> [1, 12, 8, 8]
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # 这里 attention 用了dropout，不知不知道是不是担心矩阵太稠密，这里也提到很不寻常，但是原始 Transformer 论文就是这么做的
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        # head_mask 就是之前提到的对多头计算的 mask，如果不设置默认是全 1，在这里就不会起作用
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # context_layer 即 attention 矩阵与 value 矩阵的乘积，
        # 原始的大小为：(batch_size, num_attention_heads, sequence_length, attention_head_size)
        # [1, 12, 8, 8] * [1, 12, 8, 64] -> [1, 12, 8, 64]
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer 进行转置和 view 操作以后，形状就恢复了(batch_size, sequence_length, hidden_size)
        # [1, 12, 8, 64] -> [1, 8, 12, 64]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 最后把所有head拼接起来 [1, 8, 12, 64] -> [1, 8] + [768] -> [1, 8, 768]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # 把新的shape应用在context_layer上面 -> [1, 8, 768]
        context_layer = context_layer.view(new_context_layer_shape)
        # 最终输出，如果需要attention output，则输出为 ([1, 8, 768], [1, 12, 8, 8])
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性输出层，接在self-attention之后 -> [768, 768]
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 构建LN层 -> [768, ]，对hidden_size维度做归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 构建dropout，默认参数为0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # hidden_states为self-attention的输出，[1, 8, 768] * [768, 768] -> [1, 8, 768]
        # input_tensor为self-attention的输入，也就是没有跑attention的时候的结果，对于第一层及embedding的output
        hidden_states = self.dense(hidden_states)
        # 先做dropout层
        hidden_states = self.dropout(hidden_states)
        # 残差连接后再做 LN(attention output + attention input)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 输出没变化还是 [1, 8, 768]
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # self 就是多头注意力的实现，而 output 实现 attention 后的全连接 + dropout + residual + LayerNorm 一系列操作。
        # 构建 self-attention层
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        # 构建 attention之后的线性输出层
        self.output = BertSelfOutput(config)
        # 这里有个减枝的集合来存储剪完之后的attention heads
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 先跑self-attention层，如果没有attention score output，输出为 [1, 8, 768] 为每个词对其他词的加权注意力在每个维度上
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 再跑output层 -> 对于第一层：LN(dropout(Linear(self-attention)) + embedding) -> [1, 8, 768]
        attention_output = self.output(self_outputs[0], hidden_states)
        # 最终输出为 attention_output + 如果有attention score 原始值
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 这里的全连接做了一个扩展，以 bert-base 为例，扩展维度为 3072，是原始维度 768 的 4 倍
        # 线性层 -> [768, 768 * 4 -> 3072]
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 这里的激活函数默认实现为 gelu，这也是整个BertLayer层里面唯一一个激活函数，softmax只算非线性转换。
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性转化层 [1, 8, 768] * [768, 3072] - > [1, 8, 3072]
        hidden_states = self.dense(hidden_states)
        # apply gelu activation function
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 这里和attention里面的self-output几乎一模一样，只是输入维度不一样，一个是self-attention的输出768，这一个是线性转换的输出3072
        # 这里接在线性转换之后，所以线性层为 -> [768 * 4 -> 3072, 768]
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 构建LN层 -> [768, ]，对hidden_size做归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 构建dropout，默认参数为0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 再做一次线性转换回来 [1, 8, 3072] * [3072, 768] -> [1, 8, 768]
        hidden_states = self.dense(hidden_states)
        # 先做dropout
        hidden_states = self.dropout(hidden_states)
        # 残差连接后再做 LN(attention output after linear layer + attention output)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 同样的维度对比输入hidden_states，还是[1, 8, 768]
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 每一个BertLayer，包含一个BertAttention，里面包括 self-attention + self-output
        self.attention = BertAttention(config)
        # 是否是decoder层
        self.is_decoder = config.is_decoder
        # 是否要加cross-attention，这个主要用于decode和encode之间的attention，transformer用
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        # 做完了attention后还有一个全连接+激活的操作
        self.intermediate = BertIntermediate(config)
        # 做完线性转化后，在这里又是一个全连接 + dropout + LayerNorm，还有一个残差连接 residual connect
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # attention输出：[1, 8, 768]
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # apply chunk 在 [1, 8, 768]上面，让线性层变成多个独立的小线性层跑，这样可以save memory
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        # 最终输出维度一样还是 [1, 8, 768]
        return outputs

    def feed_forward_chunk(self, attention_output):
        # attention的output做了一个线性转换 -> [1, 8, 3072]
        intermediate_output = self.intermediate(attention_output)
        # 输入最终的output层为，线性转换之后的attention和线性转化前的原始attention -> [1, 8, 768]
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 默认参数是12层，每一层是一个BertLayer，前一个的输出是下一个的输入
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            # 如果要记录每一层最终输出值，在这进行append起来
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            # 每一层的output [1, 8, 768] 是下一层的 input hidden_states
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果要记录每一层的attention score 原始输出值，在这进行append起来
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 最后一层的output需要跳出loop后额外append一次
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 如果default状态下设置输出类型，只输出最后一层的 hidden_states -> [1, 8, 768] 结果
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # pooler 层为取出了句子的第一个token，即[CLS]对应的向量，然后过一个全连接层和一个激活函数后输出
        # 线性层 -> [768, 768]，用于提取完CLS的output之后
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数默认 tanh
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # 提取第一个token对应的最后一层的output -> [1, 768]
        first_token_tensor = hidden_states[:, 0]
        # 线性转换层 [1, 768] * [768, 768] -> [1, 768]
        pooled_output = self.dense(first_token_tensor)
        # 这里有一个tanh的激活层
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性输出层 -> [768, 768]
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 构建LN层 -> [768, ]，对hidden_size维度做归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states ex: [1, 8, 768] Bert encode 后每个单词的output
        # 线性转换层 [1, 8, 768] * [768, 768] -> [1, 8, 768]
        hidden_states = self.dense(hidden_states)
        # 这里有一个Gelu的激活层
        hidden_states = self.transform_act_fn(hidden_states)
        # 这里还有个LN层，在激活函数之后，保证数据的分布不会分散开
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # transform层，用来完成一次线性变换 + 激活函数 + LN
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # 线性输出层，用于预测每个词是什么类别 [768, 30522]
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 重新初始化了一个全 0 向量作为预测权重的 bias
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        # 这里需要后续连起来bias因为需要保证bias的size和之前resize后一致
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # [1, 8, 768] -> [1, 8, 768]
        hidden_states = self.transform(hidden_states)
        # [1, 8, 768] * [768, 30522] -> [1, 8, 30522]
        hidden_states = self.decoder(hidden_states)

        # [batch_size, seq_length, vocab_size] -> [1, 8, 768]，即预测每个句子每个词是什么类别的概率值（注意这里没有做 softmax）
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 代表LM的部分
        self.predictions = BertLMPredictionHead(config)
        # 代表NSP的线性层，这里没有用 BertOnlyNSPHead
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # [1, 8, 768] -> [1, 8, 30522] 每个词的每个类别的预测概率值
        prediction_scores = self.predictions(sequence_output)
        # [1, 768] * [768, 2] -> [1, 2] NSP是二分类的问题，输出是两个类别的概率值预测
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762).

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        # BertConfig object，包含所有参数，比如：
        # "hidden_dropout_prob": 0.1，
        # "hidden_size": 768，
        # "initializer_range": 0.02，
        # "intermediate_size": 3072，
        # "layer_norm_eps": 1e-12，
        # "max_position_embeddings": 512
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # 默认参数下，加入最后的 pooling 层
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        input_ids：经过 tokenizer 分词后的 sub-word 对应的下标列表；
        attention_mask：在 self-attention 过程中，这一块 mask 用于标记 sub-word 所处句子和 padding 的区别，将 padding 部分填充为0；
        token_type_ids：标记 sub-word 当前所处句子（第一句/第二句/ padding）；
        position_ids：标记当前词所在句子的位置下标；
        head_mask：用于将某些层的某些注意力计算无效化；
        inputs_embeds：如果提供了，那就不需要input_ids，跨过 embedding lookup 过程直接作为 Embedding 进入 Encoder 计算；
        encoder_hidden_states：这一部分在 BertModel 配置为 decoder 时起作用，将执行 cross-attention 而不是 self-attention；
        encoder_attention_mask：同上，在 cross-attention 中用于标记 encoder 端输入的 padding；
        past_key_values：这个参数貌似是把预先计算好的 K-V 乘积传入，以降低 cross-attention 的开销（因为原本这部分是重复计算）；
        use_cache：将保存上一个参数并传回，加速 decoding；
        output_attentions：是否返回中间每层的 attention 输出；
        output_hidden_states：是否返回中间每层的输出；
        return_dict：是否按键值对的形式（ModelOutput 类，也可以当作 tuple 用）返回输出，默认为真

        Summary:
        在 HuggingFace 实现的 Bert 模型中，使用了多种节约memory的技术：
        gradient checkpoint，不保留前向传播节点，只在用时计算；apply_chunking_to_forward，按多个小批量和低维度计算 FFN
        BertModel 包含复杂的封装和较多的组件。以 bert-base 为例，主要组件如下：
        总计Dropout出现了1+(1+1+1)x12=37次；
        总计LayerNorm出现了1+(1+1)x12=25次； BertModel 有极大的参数量。以 bert-base 为例，其参数量为 109M。
        """
        # 例子 ex:
        # 'input_ids': tensor([[  101,  1045,  2066,  3019,  2653, 27673,   999,   102]])
        # 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]])
        # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 对于padding的位置，需要提供mask做attention的时候，因为没有意义对于其它词的attention
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        # batch_size -> 1， seq_length -> 8
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 在这里把 attention_mask从原始的[1, 0]变为[0, -1e4]的取值。
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 输出为 [batch_size, sequence_length, hidden_size] -> [1, 8, 768]
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 最后一层的output -> [1, 8, 768]
        sequence_output = encoder_outputs[0]
        # 最后跑一个pool layer 提取出我们想要的第一个token pool 后的结果 -> [1, 768]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        # 输出包含最后一层的所有词的sequence_output -> [1, 8, 768]，pooled_output ->[1, 768]
        # 或者可选的所有层的output + 所有层的attention score
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


# Below are all bert application class

# Bert pre-training model, contains MLM and NSP
class BertForPreTraining(BertPreTrainedModel):
    """
    Bert的训练的两种方式：
    Masked Language Model（MLM）：
        在句子中随机用[MASK]替换一部分单词，然后将句子传入 BERT 中编码每一个单词的信息，最终用[MASK]的编码信息预测该位置的正确单词，
        这一任务旨在训练模型根据上下文理解单词的意思；
    Next Sentence Prediction（NSP）：
        将句子对 A 和 B 输入 BERT，使用[CLS]的编码信息进行预测 B 是否 A 的下一句，这一任务旨在训练模型理解预测句子间的关系。
    """
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        # Bert model 的 init，初始化 model
        # 这里设置的是默认add_pooling_layer=True，即会提取[CLS]对应的输出用于 NSP 任务
        self.bert = BertModel(config)
        # Bert model 包含 MLM 和 NSP 两个任务
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            next_sentence_label: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        # labels：形状为[batch_size, seq_length] ，代表 MLM 任务的标签，注意这里对于原本未被遮盖的词设置为 -100，被遮盖词才会有它们对应的 id，和任务设置是反过来的。
        # 例如，原始句子是I want to [MASK] an apple，这里我把单词eat给遮住了输入模型，对应的label设置为[-100, -100, -100, 【eat对应的id】, -100, -100]；
        # 为什么要设置为 -100 而不是其他数？因为torch.nn.CrossEntropyLoss默认的ignore_index=-100，也就是说对于标签为 100 的类别输入不会计算 loss。
        # next_sentence_label：这一个输入很简单，就是 0 和 1 的二分类标签。
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # input_ids ex: tensor([[  101,  1045,  2066,  3019,  2653, 27673,   999,   102]])
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # sequence_output: [1, 8, 768] 每个单词的最终encode的output
        # pooled_output: [1, 768] 第一个单词 [CLS] 的output
        sequence_output, pooled_output = outputs[:2]
        # prediction_scores: [1, 8, 30522]
        # seq_relationship_score: [1, 2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            # prediction_scores [1, 8, 30522] -> [8, 30522]，每个词的预测概率值，这里只对label里面不是-100的进行loss计算
            # labels -> [8] 被mask的词是正确的id，没有被mask的是-100，相当于只有id的那个词会和预测概率的30522个结果进行计算CELoss
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            # 两个loss直接相加，权重一模一样
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Bert NSP only model
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        # 线性层 [768, 2]
        self.cls = BertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        # [1, 768] * [768, 2] -> [1, 2] 对于两个class的概率预测值，没有sigmoid和softmax的应用
        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Bert model for sequence classification task
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Bert model 标准输出，有pooling的
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # dropout 层，比例来自于自定义或者同hidden层一样
        self.dropout = nn.Dropout(classifier_dropout)
        # 线性层 [768, num_labels]
        # 如果初始化的num_labels=1，那么就默认为回归任务，使用 MSELoss，否则认为是分类任务。这里没有sigmoid
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [CLS] token 的输出 [1, 768]
        pooled_output = outputs[1]
        # dropout 层
        pooled_output = self.dropout(pooled_output)
        # 线性层 [1, 768] * [768, num_labels] -> [1, num_labels] 没有sigmoid和softmax
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # regression 是 MSELoss
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            #  classification 是 CELoss
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Bert for token classification task
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # 序列标注任务的输入为单个句子文本，输出为每个 token 对应的类别标签。 由于需要用到每个token对应的输出而不只是某几个，
        # 所以这里的BertModel不用加入 pooling 层
        # Bert model 标准输出，没有pooling的输出，因为最终的output是对每个单词的类别判断而不是对[CLS]的判断
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # dropout 层，比例来自于自定义或者同hidden层一样
        self.dropout = nn.Dropout(classifier_dropout)
        # 线性层 [1024, num_labels] 这里没有sigmoid，基本上任务都是多分类问题，一般num_labels > 1
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 每一个 token 的输出 [1, 30, 1024]
        sequence_output = outputs[0]
        # dropout 层
        sequence_output = self.dropout(sequence_output)
        # 线性层 [1, 30, 1024] * [1024, 9] -> [1, 30, 9] 每个单词对应的每个label的概率值，没有sigmoid
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 分类任务的标准CELoss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Bert for question answering  task
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        # 问答任务的输入为问题 +（对于 BERT 只能是一个）回答组成的句子对，输出为起始位置和结束位置用于标出回答中的具体文本。
        # 这里需要两个输出，即对起始位置的预测和对结束位置的预测，两个输出的长度都和句子长度一样，从其中挑出最大的预测值对应的下标作为预测的位置。
        # 对超出句子长度的非法 label，会将其压缩（torch.clamp_）到合理范围。
        super().__init__(config)
        self.num_labels = config.num_labels
        # Bert model 标准输出，没有pooling的输出，因为最终的output是对每个单词的是否是其实或者终止位置的判断，相当于每个位置是二分类
        self.bert = BertModel(config, add_pooling_layer=False)
        # 线性层 [1024, num_labels] 这里没有sigmoid，基本上label都是2，因为是判断是否是起始或者终止点
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 每一个 token 的输出 [1, 111, 1024]
        sequence_output = outputs[0]
        # 线性层 [1, 111, 1024] * [1024, 2] -> [1, 111, 2] 每个单词对应的每个label的概率值，没有sigmoid
        logits = self.qa_outputs(sequence_output)
        # [1, 111, 2] -> [1, 111, 1] + [1, 111, 1] 对最后一个维度进行2分为1 split，label 1 是start， label 2 是 end
        start_logits, end_logits = logits.split(1, dim=-1)
        # [1, 111, 1] -> [1, 111] 消除最后一个维度，因为是1 -> [batch_size, sequence_length]
        start_logits = start_logits.squeeze(-1).contiguous()
        # 同上 start
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
