import numpy as np

import mindspore as ms
from mindspore import amp, context, ops
from mindspore.common.initializer import initializer, TruncatedNormal
from mindspore import nn, Parameter, Tensor
from mindspore import ops as P
from mindspore.ops import functional as F
from mindspore.common.api import jit
from mindspore.ops import composite as C
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


from .bert_model import BertModel, BertPreTrainingHeads

import time

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class BertForPretraining(nn.Cell):
    r"""
    Bert For Pretraining
    """

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.vocab_size = config.vocab_size

        self.cls.predictions.decoder.weight = (
            self.bert.embeddings.word_embeddings.embedding_table
        )
    

    def construct(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        masked_lm_positions=None,
        position_ids=None,
        head_mask=None,       
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        # ic(outputs) # [shape(batch_size, 128, 256), shape(batch_size, 256)]

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output, masked_lm_positions
        )
        outputs = (
            prediction_scores,
            seq_relationship_score,
        ) + outputs[2:]
        # ic(outputs) # [shape(batch_size, 128, 256), shape(batch_size, 256)]

        return outputs


class BertPretrainingLoss(nn.Cell):
    """
    Provide bert pre-training loss.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        Tensor, total loss.
    """

    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.log_softmax=nn.LogSoftmax(axis=-1)

    def construct(self, prediction_scores, seq_relationship_score, masked_lm_ids,
                  masked_lm_weights, next_sentence_labels):
        """Defines the computation performed."""
        prediction_scores = self.log_softmax(prediction_scores)
        one_hot_labels = self.onehot(masked_lm_ids.view(-1), self.vocab_size, self.on_value, self.off_value)      
        per_example_loss = self.neg(self.reduce_sum(prediction_scores * one_hot_labels, -1))
        numerator = self.reduce_sum(masked_lm_weights.view(-1) * per_example_loss, ())
        denominator = self.reduce_sum(masked_lm_weights.view(-1), ()) + self.cast(F.tuple_to_array((1e-5,)), ms.float32)
        masked_lm_loss = numerator / denominator
        # next_sentence_loss
        next_sentence_loss = F.cross_entropy(seq_relationship_score, next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss
    

class BertNetworkWithLoss(nn.Cell):

    def __init__(self, config):
        super().__init__()
        self.bert = BertForPretraining(config)
        self.loss = BertPretrainingLoss(config)
    
    def construct(self, 
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights):     
        bert_output = self.bert(input_ids, input_mask, token_type_id, masked_lm_positions)
        prediction_scores, seq_relationship_score = bert_output[0], bert_output[1]
       
        total_loss = self.loss(prediction_scores, seq_relationship_score,
                               masked_lm_ids, masked_lm_weights, next_sentence_labels)
        
        return total_loss 
    

class BertTrainOneStepCell(nn.TrainOneStepCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        enable_clip_grad (boolean): If True, clip gradients in BertTrainOneStepCell. Default: True.
    """

    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True):
        super(BertTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()
        self.enable_clip_grad = enable_clip_grad
        self.enable_tuple_broaden = True

    def set_sens(self, value):
        self.sens = value

    @jit
    def clip_grads(self, grads):
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        return grads

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights):
        """Defines the computation performed."""
        weights = self.weights

        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           ms.float32))
        if self.enable_clip_grad:
            grads = self.clip_grads(grads)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class BertTrainOneStepWithLossScaleCellForAdam(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Different from BertTrainOneStepWithLossScaleCell, the optimizer takes the overflow
    condition as input.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertTrainOneStepWithLossScaleCellForAdam, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ms.float32))
        self.enable_tuple_broaden = True

    @jit
    def clip_grads(self, grads):
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        return grads

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           ms.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.clip_grads(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if self.loss_scaling_manager is not None:
            overflow = self.loss_scaling_manager(scaling_sens, cond)
        self.optimizer(grads, overflow)
        return loss, cond, scaling_sens.value()


class BertTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ms.float32))
        self.enable_tuple_broaden = True

    @jit
    def clip_grads(self, grads):
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        return grads

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           ms.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        degree_sens = self.cast(scaling_sens * self.degree, ms.float32)
        grads = self.hyper_map(F.partial(grad_scale, degree_sens), grads)
        grads = self.clip_grads(grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return loss, cond, scaling_sens.value()
    

class BertPretrainEval(nn.Cell):
    '''
    Evaluate MaskedLM prediction scores
    '''
    def __init__(self, config, network=None):
        super().__init__(auto_prefix=False)
        if network is None:
            self.network = BertForPretraining(config)
        else:
            self.network = network
        self.argmax = P.Argmax(axis=-1, output_type=ms.int32)
        self.equal = P.Equal()
        self.sum = P.ReduceSum()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.allreduce = P.AllReduce()
        self.reduce_flag = False
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reduce_flag = True
        self.log_softmax = nn.LogSoftmax(axis=-1)

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights):
        """Calculate prediction scores"""
        bs, _ = self.shape(input_ids)
        mlm, _ = self.network(input_ids, input_mask, token_type_id, masked_lm_positions)       
        mlm = self.log_softmax(mlm)       
        index = self.argmax(mlm)
        index = self.reshape(index, (bs, -1))
        eval_acc = self.equal(index, masked_lm_ids)
        eval_acc = self.cast(eval_acc, ms.float32)
        real_acc = eval_acc * masked_lm_weights
        acc = self.sum(real_acc)
        total = self.sum(masked_lm_weights)

        if self.reduce_flag:
            acc = self.allreduce(acc)
            total = self.allreduce(total)

        return acc, total

cast = P.Cast()

add_grads = C.MultitypeFuncGraph("add_grads")

@add_grads.register("Tensor", "Tensor")
def _add_grads(accu_grad, grad):
    return accu_grad + cast(grad, ms.float32)

update_accu_grads = C.MultitypeFuncGraph("update_accu_grads")

@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, cast(grad, ms.float32)))

accumulate_accu_grads = C.MultitypeFuncGraph("accumulate_accu_grads")

@accumulate_accu_grads.register("Tensor", "Tensor")
def _accumulate_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign_add(accu_grad, cast(grad, ms.float32)))


zeroslike = P.ZerosLike()
reset_accu_grads = C.MultitypeFuncGraph("reset_accu_grads")


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, zeroslike(accu_grad)))


class BertTrainAccumulationAllReducePostWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    To mimic higher batch size, gradients are accumulated N times before weight update.

    For distribution mode, allreduce will only be implemented in the weight updated step,
    i.e. the sub-step after gradients accumulated N times.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        accumulation_steps (int): Number of accumulation steps before gradient update. The global batch size =
                                batch_size * accumulation_steps. Default: 1.
    """

    def __init__(self, network, optimizer, scale_update_cell=None, accumulation_steps=1, enable_global_norm=False):
        super(BertTrainAccumulationAllReducePostWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.enable_global_norm = enable_global_norm
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))
        self.local_step = Parameter(initializer(0, [1], ms.int32))
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], ms.int32))
        self.accu_loss = Parameter(initializer(0, [1], ms.float32))

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.overflow_reducer = F.identity
        if self.is_distributed:
            self.overflow_reducer = P.AllReduce()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, ms.float32)
        self.less_equal = P.LessEqual()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ms.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        # update accumulation parameters
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)
        self.local_step = self.select(is_accu_step, self.local_step + self.one, self.one)
        self.accu_loss = self.select(is_accu_step, self.accu_loss + loss, loss)
        mean_loss = self.accu_loss / self.local_step
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)

        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           ms.float32))

        accu_succ = self.hyper_map(accumulate_accu_grads, self.accu_grads, grads)
        mean_loss = F.depend(mean_loss, accu_succ)

        overflow = ops.logical_not(amp.all_finite(grads))
        accu_overflow = self.select(overflow, self.one, self.zero)
        self.accu_overflow = self.select(is_accu_step, accu_overflow, self.zero)

        if not is_accu_step:
            # apply grad reducer on grads
            grads = self.grad_reducer(self.accu_grads)
            scaling = scaling_sens * self.degree * self.accumulation_steps
            grads = self.hyper_map(F.partial(grad_scale, scaling), grads)
            if self.enable_global_norm:
                grads = C.clip_by_global_norm(grads, 1.0, None)
            else:
                grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
            accu_overflow = F.depend(accu_overflow, grads)
            accu_overflow = self.overflow_reducer(accu_overflow)
            overflow = self.less_equal(self.base, accu_overflow)
            accu_succ = self.hyper_map(reset_accu_grads, self.accu_grads)
            overflow = F.depend(overflow, accu_succ)
            overflow = self.reshape(overflow, (()))
            if sens is None:
                overflow = self.loss_scaling_manager(self.loss_scale, overflow)
            if not overflow:
                self.optimizer(grads)

        return mean_loss, overflow, scaling_sens.value()


class BertTrainAccumulationAllReduceEachWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    To mimic higher batch size, gradients are accumulated N times before weight update.

    For distribution mode, allreduce will be implemented after each sub-step and the trailing time
    will be overided by backend optimization pass.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        accumulation_steps (int): Number of accumulation steps before gradient update. The global batch size =
                                  batch_size * accumulation_steps. Default: 1.
    """
    def __init__(self, network, optimizer, scale_update_cell=None, accumulation_steps=1, enable_global_norm=False):
        super(BertTrainAccumulationAllReduceEachWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.enable_global_norm = enable_global_norm
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))
        self.local_step = Parameter(initializer(0, [1], ms.int32))
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], ms.int32))
        self.accu_loss = Parameter(initializer(0, [1], ms.float32))

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.overflow_reducer = F.identity
        if self.is_distributed:
            self.overflow_reducer = P.AllReduce()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, ms.float32)
        self.less_equal = P.LessEqual()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ms.float32))

    @C.add_flags(has_effect=True)
    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        # update accumulation parameters
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)
        self.local_step = self.select(is_accu_step, self.local_step + self.one, self.one)
        self.accu_loss = self.select(is_accu_step, self.accu_loss + loss, loss)
        mean_loss = self.accu_loss / self.local_step
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)

        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           ms.float32))


        accu_grads = self.hyper_map(add_grads, self.accu_grads, grads)
        scaling = scaling_sens * self.degree * self.accumulation_steps
        grads = self.hyper_map(F.partial(grad_scale, scaling), accu_grads)
        grads = self.grad_reducer(grads)

        overflow = ops.logical_not(amp.all_finite(grads))
        if self.reducer_flag:
            overflow = self.allreduce(overflow.to(ms.float32)) >= self.base
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)
        self.accu_overflow = self.select(is_accu_step, accu_overflow, self.zero)
        overflow = self.reshape(overflow, (()))

        if is_accu_step:
            succ = False
            accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, accu_grads)
            succ = F.depend(succ, accu_succ)
        else:
            if sens is None:
                overflow = self.loss_scaling_manager(self.loss_scale, overflow)
            if overflow:
                succ = False
            else:
                if self.enable_global_norm:
                    grads = C.clip_by_global_norm(grads, 1.0, None)
                else:
                    grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

                succ = self.optimizer(grads)

            accu_succ = self.hyper_map(reset_accu_grads, self.accu_grads)
            succ = F.depend(succ, accu_succ)

        ret = (mean_loss, overflow, scaling_sens.value())
        return F.depend(ret, succ)
