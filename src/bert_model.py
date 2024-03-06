import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn, ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import TruncatedNormal, initializer, Normal
# from .model_utils.config import PretrainedConfig
from .utils import  finfo
#make_causal_mask,
# from ...activations import ACT2FN
from typing import Dict

activation_map = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(False),
    'gelu_approximate': nn.GELU(),
    'swish':nn.SiLU()
}



class PretrainedConfig:
    """
    Abstract class for Pretrained models config.
    """
    is_composition = False
    # Add for handle attribute_map
    attribute_map: Dict[str, str] = {}

    def __init__(self, **kwargs):
        self.ms_dtype = kwargs.pop("ms_dtype", None)
        if 'torch_dtype' in kwargs:
            self.ms_dtype = kwargs.pop("torch_dtype", None)
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)

        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", True
        )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.cross_attention_hidden_size = kwargs.pop("cross_attention_hidden_size", None)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

       # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.output_scores = kwargs.pop("output_scores", False)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.label2id is not None and not isinstance(self.label2id, dict):
            raise ValueError("Argument label2id should be a dictionary.")
        if self.id2label is not None:
            if not isinstance(self.id2label, dict):
                raise ValueError("Argument id2label should be a dictionary.")
            num_labels = kwargs.pop("num_labels", None)

            if num_labels is not None and len(self.id2label) != num_labels:
                logger.warning(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{self.id2label}. The number of labels wil be overwritten to {self.num_labels}."
                )
            self.id2label = {int(key): value for key, value in self.id2label.items()}
            # Keys are always strings in JSON so convert ids to int here.
        else:
            self.num_labels = kwargs.pop("num_labels", 2)

        # if self.ms_dtype is not None and isinstance(self.ms_dtype, str):
        #     if is_mindspore_available():
        #         import mindspore

        #         self.ms_dtype = getattr(mindspore, self.ms_dtype)

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)

        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type} "
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err


class BertConfig(PretrainedConfig):
    """
    Configuration for BERT-base
    """

    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_relative_positions=False,
        # pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_relative_positions = use_relative_positions
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class BertEmbeddings(nn.Cell):
    """
    Embeddings for BERT, include word, position and token_type
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Cell):
    """
    Self attention layer for BERT.
    """

    def __init__(self, config, causal, init_cache=False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Dense(
            config.hidden_size,
            self.all_head_size,
        )
        self.key = nn.Dense(
            config.hidden_size,
            self.all_head_size,
        )
        self.value = nn.Dense(
            config.hidden_size,
            self.all_head_size,
        )

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(-1)

        self.causal = causal
        self.init_cache = init_cache

        # self.causal_mask = make_causal_mask(
        #     ops.ones((1, config.max_position_embeddings), dtype=mstype.bool_),
        #     dtype=mstype.bool_,
        # )

        if not init_cache:
            self.cache_key = None
            self.cache_value = None
            self.cache_index = None
        else:
            self.cache_key = Parameter(
                initializer(
                    "zeros",
                    (
                        config.max_length,
                        config.max_batch_size,
                        config.num_attention_heads,
                        config.attention_head_size,
                    ),
                )
            )
            self.cache_value = Parameter(
                initializer(
                    "zeros",
                    (
                        config.max_length,
                        config.max_batch_size,
                        config.num_attention_heads,
                        config.attention_head_size,
                    ),
                )
            )
            self.cache_index = Parameter(Tensor(0, mstype.int32))

    def _concatenate_to_cache(self, key, value, query, attention_mask):
        if self.init_cache:
            batch_size = query.shape[0]
            num_updated_cache_vectors = query.shape[1]
            max_length = self.cache_key.shape[0]
            indices = ops.arange(
                self.cache_index, self.cache_index + num_updated_cache_vectors
            )
            key = ops.scatter_update(self.cache_key, indices, key.swapaxes(0, 1))
            value = ops.scatter_update(self.cache_value, indices, value.swapaxes(0, 1))

            self.cache_index += num_updated_cache_vectors

            pad_mask = ops.broadcast_to(
                ops.arange(max_length) < self.cache_index,
                (batch_size, 1, num_updated_cache_vectors, max_length),
            )
            attention_mask = ops.logical_and(attention_mask, pad_mask)

        return key, value, attention_mask

    def transpose_for_scores(self, input_x):
        r"""
        transpose for scores
        """
        new_x_shape = input_x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        input_x = input_x.view(*new_x_shape)
        return input_x.transpose(0, 2, 1, 3)

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_states = self.transpose_for_scores(mixed_query_layer)
        key_states = self.transpose_for_scores(mixed_key_layer)
        value_states = self.transpose_for_scores(mixed_value_layer)        

        if self.causal:
            # query_length, key_length = query_states.shape[1], key_states.shape[1]
            # if self.has_variable("cache", "cached_key"):
            #     mask_shift = self.variables["cache"]["cache_index"]
            #     max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            #     causal_mask = ops.slice(
            #         self.causal_mask,
            #         (0, 0, mask_shift, 0),
            #         (1, 1, query_length, max_decoder_length),
            #     )
            # else:
            #     causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            # causal_mask = ops.broadcast_to(
            #     causal_mask, (batch_size,) + causal_mask.shape[1:]
            # )
            causal_mask = None
        else:
            causal_mask = None

        if attention_mask is not None and self.causal:
            attention_mask = ops.broadcast_to(
                attention_mask.expand_dims(-2).expand_dims(-3), causal_mask.shape
            )
            attention_mask = ops.logical_and(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:            
            attention_mask = attention_mask.expand_dims(-2).expand_dims(-3)                 

        if self.causal and self.init_cache:
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            attention_bias = ops.select(
                attention_mask > 0,
                ops.zeros_like(attention_mask).astype(hidden_states.dtype),
                (ops.ones_like(attention_mask) * finfo(hidden_states.dtype, "min")).astype(
                    hidden_states.dtype
                ),
            )
        else:
            attention_bias = None

        # Take the dot product between "query" snd "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_states, key_states.swapaxes(-1, -2))
        attention_scores = attention_scores / ops.sqrt(
            Tensor(self.attention_head_size, mstype.float32)
        )
        # Apply the attention mask is (precommputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_bias

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = ops.matmul(attention_probs, value_states)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class BertSelfOutput(nn.Cell):
    r"""
    Bert Self Output
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Cell):
    r"""
    Bert Attention
    """

    def __init__(self, config, causal, init_cache=False):
        super().__init__()
        self.self = BertSelfAttention(config, causal, init_cache)
        self.output = BertSelfOutput(config)

    def construct(self, hidden_states, attention_mask=None, head_mask=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class  BertIntermediate(nn.Cell):
    r"""
    Bert Intermediate
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = activation_map.get(config.hidden_act, nn.GELU(False))

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Cell):
    r"""
    Bert Output
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.intermediate_size,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Cell):
    r"""
    Bert Layer
    """

    def __init__(self, config, init_cache=False):
        super().__init__()
        self.attention = BertAttention(config, causal=config.is_decoder, init_cache=init_cache)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        if config.add_cross_attention:
            self.crossattention = BertAttention(config, causal=False, init_cache=init_cache)
        
    def construct(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states = None,
                encoder_attention_mask = None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
            )
            attention_output = cross_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Cell):
    r"""
    Bert Encoder
    """

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.CellList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def _set_recompute(self):
        for layer in self.layer:
            layer.recompute()

    def construct(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states = None,
                encoder_attention_mask = None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask
                )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions += (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs += (all_hidden_states,)
        if self.output_attentions:
            outputs += (all_attentions,)
        return outputs

class BertPooler(nn.Cell):
    r"""
    Bert Pooler
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
        )
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding.
        # to the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Cell):
    r"""
    Bert Prediction Head Transform
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
        )
        self.transform_act_fn = activation_map.get(config.hidden_act, nn.GELU(False))
        self.LayerNorm = nn.LayerNorm(
            (config.hidden_size,), epsilon=config.layer_norm_eps
        )

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states





class BertPreTrainingHeads(nn.Cell):
    r"""
    Bert PreTraining Heads
    """

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output, masked_lm_positions):
        prediction_scores = self.predictions(sequence_output, masked_lm_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel():
    """BertPretrainedModel"""
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_recompute = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(
                initializer(
                    Normal(self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.has_bias:
                cell.bias.set_data(
                    initializer("zeros", cell.bias.shape, cell.bias.dtype)
                )
        elif isinstance(cell, nn.Embedding):
            weight = initializer(
                Normal(self.config.initializer_range),
                cell.weight.shape,
                cell.weight.dtype,
            )
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer("ones", cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))


class BertModel(nn.Cell):
    r"""
    Bert Model
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.num_hidden_layers = config.num_hidden_layers

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def construct(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states = None,
        encoder_attention_mask = None
    ):
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)
        if position_ids is None:       
            position_ids = ops.broadcast_to(ops.arange(ops.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = (
                    head_mask.expand_dims(0)
                    .expand_dims(0)
                    .expand_dims(-1)
                    .expand_dims(-1)
                )
                head_mask = ops.broadcast_to(
                    head_mask, (self.num_hidden_layers, -1, -1, -1, -1)
                )
            elif head_mask.ndim == 2:
                head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]
        # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


# class BertForPretraining(nn.Cell):
#     r"""
#     Bert For Pretraining
#     """

#     def __init__(self, config):
#         super().__init__()
#         self.bert = BertModel(config)
#         self.cls = BertPreTrainingHeads(config)
#         self.vocab_size = config.vocab_size

#         self.cls.predictions.decoder.weight = (
#             self.bert.embeddings.word_embeddings.embedding_table
#         )
    

#     def construct(
#         self,
#         input_ids,
#         attention_mask=None,
#         token_type_ids=None,
#         masked_lm_positions=None,
#         position_ids=None,
#         head_mask=None,       
#     ):
#         print("=====in the construct of the BertPretraining the shape of the masked_lm_position:", masked_lm_positions.shape)
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#         )
#         # ic(outputs) # [shape(batch_size, 128, 256), shape(batch_size, 256)]

#         sequence_output, pooled_output = outputs[:2]
#         prediction_scores, seq_relationship_score = self.cls(
#             sequence_output, pooled_output, masked_lm_positions
#         )

#         outputs = (
#             prediction_scores,
#             seq_relationship_score,
#         ) + outputs[2:]
#         # ic(outputs) # [shape(batch_size, 128, 256), shape(batch_size, 256)]

#         return outputs


class BertLMPredictionHead(nn.Cell):
    r"""
    Bert LM Prediction Head
    """

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(
            config.hidden_size,
            config.vocab_size,
            has_bias=False,
        )

        self.bias = Parameter(initializer("zeros", config.vocab_size), "bias")

    def construct(self, hidden_states, masked_lm_positions):
        batch_size, seq_len, hidden_size = hidden_states.shape
        if masked_lm_positions is not None:
            flat_offsets = ops.arange(batch_size) * seq_len
            flat_position = (masked_lm_positions + flat_offsets.reshape(-1, 1)).reshape(
                -1
            )
            flat_sequence_tensor = hidden_states.reshape(-1, hidden_size)
            hidden_states = ops.gather(flat_sequence_tensor, flat_position, 0)
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertForSequenceClassification(BertPreTrainedModel):
    """Bert Model for classification tasks"""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.classifier = nn.Dense(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(p=classifier_dropout)

    def construct(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = (logits,) + outputs[2:]

        return output


__all__ = [
    "BertEmbeddings",
    "BertAttention",
    "BertEncoder",
    "BertIntermediate",
    "BertLayer",
    "BertModel",
    "BertLMPredictionHead",
    "BertForSequenceClassification",
]

