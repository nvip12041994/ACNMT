# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from re import L

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

# ================================================================
import os
import torch
import numpy as np
import torch.nn.functional as F
import copy
from torch.autograd import Variable
from fairseq.sequence_generator import SequenceGenerator
from fairseq import scoring
import time

@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("ac_cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary
        self.vocab_size = len(task.target_dictionary)
        #self.scorer = scoring.build_scorer("bleu", self.tgt_dict)
        #self.entropy_coeff = 1
        #self.gamma = 0.99

    
    def forward(self, model, sample, user_parameter=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        
        bsz, src_len = sample['net_input']['src_tokens'].size()[:2]
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        if user_parameter is not None and user_parameter["valid_bleu"] and user_parameter["dis_score"] and not user_parameter["do_valid"]:
            avg_dis_score = np.mean(np.array(user_parameter["dis_score"]))
            avg_valid_bleu = np.mean(np.array(user_parameter["valid_bleu"]))/100
            reward = 0.3*avg_dis_score + 0.7*avg_valid_bleu
            loss = loss*(1-reward)
        # if user_parameter is not None:
        #     lprobs, target = self.compute_lprob(model, net_output, sample)
        #     probs = torch.exp(lprobs) * (0.3*user_parameter["valid_discs"] + 0.7*user_parameter["valid_bleu"])
        #     lprobs = torch.log(probs)
        #     lprobs = lprobs.view(-1, lprobs.size(-1))
        #     loss = F.nll_loss(
        #         lprobs,
        #         target,
        #         ignore_index=self.padding_idx,
        #         reduction="sum" if reduce else "none",
        #         #reduction="none",
        #     )
        # else:
        #     loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(
                0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    def compute_lprob(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)        
        target = model.get_targets(sample, net_output).view(-1)
        return lprobs, target

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none", #mean
            # reduction="none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(
                    meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
