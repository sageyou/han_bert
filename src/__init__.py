# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bert Init."""
from .bert_for_pre_training import BertNetworkWithLoss, BertForPretraining, \
    BertPretrainingLoss, BertTrainOneStepCell, BertTrainOneStepWithLossScaleCell, \
    BertTrainAccumulationAllReduceEachWithLossScaleCell, \
    BertTrainAccumulationAllReducePostWithLossScaleCell, \
    BertTrainOneStepWithLossScaleCellForAdam, BertPretrainEval

from .bert_model import BertAttention, BertConfig,  BertModel, \
    BertOutput, BertSelfAttention

from .adam import AdamWeightDecayForBert, AdamWeightDecayOp
__all__ = [
    "BertNetworkWithLoss", "BertForPretraining", 
    "BertPretrainingLoss", "BertTrainOneStepCell", 
    "BertTrainOneStepWithLossScaleCell", 
    "BertTrainAccumulationAllReduceEachWithLossScaleCell",
    "BertTrainAccumulationAllReducePostWithLossScaleCell",
    "BertTrainOneStepWithLossScaleCellForAdam", "BertPretrainEval",
    "BertAttention", "BertConfig",  "BertModel", "BertOutput", "BertSelfAttention",
    "AdamWeightDecayForBert", "AdamWeightDecayOp"
]
