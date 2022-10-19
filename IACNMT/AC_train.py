#!/usr/bin/env python3 -u

"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import math
import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from modify_trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from fairseq.sequence_generator import SequenceGenerator

import random
import torch
from torch.autograd import Variable
from fairseq import search

from discriminator_lightconv import Discriminator_lightconv
import sacrebleu



import time
import subprocess
from discriminator_bert_rank import DiscriminativeNMTReranker
from discriminative_reranking_criterion import KLDivergenceRerankingCriterion
from fairseq.dataclass import ChoiceEnum
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.data import (
    ConcatDataset,
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    indexed_dataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset,
    TokenBlockDataset,
)

ACTIVATION_FN_CHOICES = ChoiceEnum(utils.get_available_activation_fns())
JOINT_CLASSIFICATION_CHOICES = ChoiceEnum(["none", "sent"])
SENTENCE_REP_CHOICES = ChoiceEnum(["head", "meanpool", "maxpool"])
_EPSILON = torch.finfo(torch.float32).eps
TARGET_DIST_NORM_CHOICES = ChoiceEnum(["none", "minmax"])

def get_gpu_memory_map():   
    result = subprocess.check_output(
        [
            '/bin/nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    
    return float(result)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    #logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    # print("================================================================")
    # print("Setup task, e.g., translation, language modeling, etc.")
    # print(cfg.task)
    # print("================================================================")
    task = tasks.setup_task(cfg.task)
    
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    # logger.info(model)
    # logger.info("task: {}".format(task.__class__.__name__))
    # logger.info("model: {}".format(model.__class__.__name__))
    # logger.info("criterion: {}".format(criterion.__class__.__name__))
    # logger.info(
    #     "num. model params: {:,} (num. trained: {:,})".format(
    #         sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
    #         sum(getattr(p, "_orig_size", p).numel() for p in model.parameters() if p.requires_grad),
    #     )
    # )
    print("Generator model loaded successfully!")
    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per GPU = {} and batch size per GPU = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    

    class DiscriminativeNMTRerankerConfig():
        def __init__(self, pretrained_model = "", sentence_rep = "head", dropout =0.1,
                    attention_dropout = 0.0, activation_dropout =0.0, classifier_dropout =0.0, 
                    embed_dim = 768, ffn_embed_dim = 2048, encoder_layers = 12, attention_heads = 8, 
                    encoder_normalize_before = False, apply_bert_init = False, freeze_embeddings = False,
                    activation_fn = "relu", n_trans_layers_to_freeze = 0,
                    joint_classification = "none", num_joint_layers = 1, 
                    joint_normalize_before = False):
            self.pretrained_model        = pretrained_model
            self.sentence_rep            = SENTENCE_REP_CHOICES(sentence_rep)
            self.dropout                 = dropout
            self.attention_dropout       = attention_dropout
            self.activation_dropout      = activation_dropout
            self.classifier_dropout      = classifier_dropout
            self.embed_dim               = embed_dim
            self.ffn_embed_dim           = ffn_embed_dim
            self.encoder_layers          = encoder_layers
            self.attention_heads         = attention_heads
            self.encoder_normalize_before= encoder_normalize_before
            self.apply_bert_init         = apply_bert_init
            self.activation_fn           = ACTIVATION_FN_CHOICES(activation_fn)
            self.freeze_embeddings       = freeze_embeddings
            self.n_trans_layers_to_freeze= n_trans_layers_to_freeze
            self.joint_classification    = JOINT_CLASSIFICATION_CHOICES(joint_classification), 
            self.num_joint_layers        = num_joint_layers
            self.joint_normalize_before  = joint_normalize_before

    class KLDivergenceRerankingCriterionConfig():
        def __init__ (self, target_dist_norm = 'minmax', temperature = 1.0, forward_batch_size = 10, mt_beam = 5):
            
            self.target_dist_norm   = TARGET_DIST_NORM_CHOICES
            self.temperature        = temperature
            self.forward_batch_size = forward_batch_size
            self.mt_beam            = mt_beam

    config_criterion  = KLDivergenceRerankingCriterionConfig()
    config_model = DiscriminativeNMTRerankerConfig()
    discriminator  = DiscriminativeNMTReranker(task = task, args = config_model)
    d_criterion      = KLDivergenceRerankingCriterion(cfg = config_criterion)

    #----------------------------------------------------------------
    use_cuda = (torch.cuda.device_count() >= 1)
    d_optimizer = eval("torch.optim." + 'SGD')(filter(lambda x: x.requires_grad,
                                                                discriminator.parameters()),
                                                        0.0001,
                                                        momentum=0.9,
                                                        nesterov=True)
    if use_cuda:
        discriminator.cuda()
        d_criterion.cuda()
    else:
        discriminator.cpu()
    print("Discriminator loaded successfully!")
    discriminator_path = "{}/discriminator_{}.pt".format(trainer.cfg.checkpoint.save_dir,epoch_itr.epoch)
    
    
    print("Policy gradient criterion loaded successfully!")
    if os.path.isfile(discriminator_path):
        print("Load from discriminator_{}.pt".format(epoch_itr.epoch))
        checkpoint = torch.load(discriminator_path)
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        d_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
        print ("Discriminator load sate successfully")
    else:
        print ("No State of Discriminator loaded")

    # Initialize generator
    # translator = SequenceGenerator(
    #     [model],
    #     task.tgt_dict,
    #     search_strategy = search.Sampling(tgt_dict = task.tgt_dict,sampling_topk=-1, sampling_topp=0.95),
        
    #     beam_size=1,
    #     max_len_a=1.2,
    #     max_len_b=10,
    # )
    
    translator = SequenceGenerator(
        [model],
        task.tgt_dict,
        search_strategy = search.Sampling(tgt_dict = task.tgt_dict,sampling_topk=-1, sampling_topp=0.95),
        
        beam_size=5,
        match_source_len=True,
    )

    if use_cuda:
        translator.cuda()
    print("SequenceGenerator loaded successfully!")
    #----------------------------------------------------------------

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()    
    print("Start training")     
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr, 
                                          discriminator, translator, d_criterion,d_optimizer,
                                          model)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
        
        
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))
    
    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False

#-------------------------------------------------------------
from fairseq import scoring
import torch.nn.functional as F
def tensor_padding_to_fixed_length(input_tensor,max_len,pad):
    output_tensor = input_tensor.cpu()
    p1d = (0,max_len - input_tensor.shape[0])
    output_tensor = F.pad(input_tensor,p1d,"constant",pad)
    return output_tensor.cuda()


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}

def FindMaxLength(lst):
    maxList = max(lst, key = lambda i: len(i))
    maxLength = len(maxList)
    return maxLength

def get_bleu(in_sent, target_sent):
    bleu = sacrebleu.corpus_bleu([in_sent], [[target_sent]])
    out = " ".join(
        map(str, [bleu.score, bleu.sys_len, bleu.ref_len] + bleu.counts + bleu.totals)
    )
    return out


def get_ter(in_sent, target_sent):
    ter = sacrebleu.corpus_ter([in_sent], [[target_sent]])
    out = " ".join(map(str, [ter.score, ter.num_edits, ter.ref_length]))
    return out

def get_token_translate_from_sample(network,user_parameter,sample,scorer,src_dict,tgt_dict):
    network.eval()        
    
    translator = user_parameter["translator"]    
    target_tokens  = sample['target']
    src_tokens = sample['net_input']['src_tokens']
    segment_label = user_parameter["segment_label"]
    bos = user_parameter["bos"]
        
    with torch.no_grad():
        hypos = translator.generate([network],sample = sample)

    # tmp = []
    bleus = []
    input_src = []
    input_tgt = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        #print("==================")
        has_target = sample["target"] is not None

        # Remove padding
        if "src_tokens" in sample["net_input"]:
            src_token = utils.strip_pad(
                sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
            )
        else:
            src_token = None

        target_token = None
        
        if has_target:
            target_token = (
                utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
            )
        
        src_str = src_dict.string(src_token, None)
        target_str = tgt_dict.string(
                    target_token,
                    None,
                    escape_unk=True,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(translator),
                )
        # Process top predictions
        
        for j, hypo in enumerate(hypos[i][: 5]): # nbest = 5            
            hypo_token, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo["tokens"].int().cpu(),
                src_str=src_str,
                alignment=None,
                align_dict=None,
                tgt_dict=tgt_dict,
                remove_bpe=None,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(translator),
            )
            input_src.append(torch.cat((torch.tensor([bos]).to(src_token.device),
                                        src_token[:-1], hypo["tokens"][:-1], 
                                        torch.tensor([segment_label]).to(src_token.device),
                                        src_token[:-1], target_token.to(src_token.device)),0))
            #input_tgt.append(torch.cat((src_token, target_token.to(src_token.device)),0))
            #scorer.add(target_token, hypo_token)
            #bleu_score = scorer.score()
            bleu_score = get_bleu(hypo_str,target_str)
            bleus.append(bleu_score)
    
    # max_len = FindMaxLength(tmp)
    # hypo_tokens_out = torch.empty(size=(len(tmp),max_len), dtype=torch.int64,device = 'cuda')
    # for i in range(len(tmp)):
    #     hypo_tokens_out[i]= tensor_padding_to_fixed_length(tmp[i],max_len,tgt_dict.pad())

    torch.cuda.empty_cache()
    return input_src,bleus

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

import numpy as np

def make_sample(input_src,bleus,user_parameter):
    label = torch.from_numpy(np.array([np.fromstring(i, dtype=float, sep=' ') for i in bleus]))
    src_lengths = torch.tensor([torch.numel(i) for i in input_src])
    
    tensor_input = collate_tokens(input_src,1) 
    dataset = {
            "id": torch.tensor([i for i in range(0,len(input_src))]),
            "net_input": {
                "src_tokens": tensor_input,
                "src_lengths": src_lengths,
            },
            "nsentences": len(input_src),
            "ntokens": NumelDataset(tensor_input, reduce=True),
            "target": label.to(tensor_input.device),
        }
    return dataset
#-------------------------------------------------------------
@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, 
    discriminator, translator, d_criterion, d_optimizer, model
) -> Tuple[List[Optional[float]], bool]:    
    """Train the model for one epoch and return validation losses."""
    
    max_len_src = epoch_itr.dataset.src.sizes.max()
    max_len_target = epoch_itr.dataset.tgt.sizes.max()
    max_len_hypo = math.ceil(max_len_src*translator.max_len_a) + translator.max_len_b
    # state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
    dis_score = []
    valid_bleu = []
    do_valid = False
    user_parameter = {
        "max_len_src": max_len_src,
        "max_len_target": max_len_target,
        "max_len_hypo": max_len_hypo,
        "discriminator": discriminator, 
        "translator": translator,
        "d_criterion": d_criterion,
        "d_optimizer": d_optimizer,
        "tokenizer": trainer.task.tokenizer,
        "segment_label": task.source_dictionary.eos(),
        "bos": task.source_dictionary.bos(),
        "dis_score" : dis_score,
        "valid_bleu" : valid_bleu,
        "do_valid" : do_valid
        # "state_list": state_list,
        # "action_list": action_list,
        # "reward_list": reward_list,
        # "next_state_list": next_state_list,
        # "done_list": done_list,
    }
    
    scorer = scoring.build_scorer("bleu", task.target_dictionary)
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    
    def train_discriminator(user_parameter,sample):
        model = user_parameter["discriminator"]
        criterion = user_parameter["d_criterion"]
        optimizer = user_parameter["d_optimizer"]
        model.train()
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        # if ignore_grad:
        #     loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            #optimizer.backward(loss)
            loss.backward()
        return loss, sample_size, logging_output
    
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)
    #print(discriminator)
    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    #valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)
    #print(valid_losses)
    for i, samples in enumerate(progress):     
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):

            # enter action into the env
            user_parameter["do_valid"] = False
            # part I: train the generator
            log_output = trainer.train_step(samples, user_parameter)
            # part II: train the discriminator
            if user_parameter is not None:
                for i, sample in enumerate(samples):
                    if "target" in sample:
                        sample["target"] =  sample["target"].to(device)
                        sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].to(device)
                    else:
                        sample = sample.to(device)
                    
                    # select an action from the agent's policy
                    input_src, bleus = get_token_translate_from_sample(model,user_parameter,
                                                    sample, scorer,task.source_dictionary,task.target_dictionary)

                    dis_sample = make_sample(input_src=input_src, bleus=bleus, user_parameter=user_parameter)
                    dis_loss, dis_sample_size, dis_logging_output = train_discriminator(user_parameter, dis_sample)
                    del dis_loss

            #print("After batch {0} GPU memory used {1:.3f}".format(i,get_gpu_memory_map()))
            
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
            #if num_updates % 10 == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)
                print("discriminator loss = {:.2f}".format(dis_logging_output["loss"]))
                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch, user_parameter
        )
        
        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    user_parameter,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation
    #test
    #do_validate = True
    # Validate
    valid_losses = [None]
    if do_validate:
        user_parameter["do_valid"] = True
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets, user_parameter)

    should_stop |= should_stop_early(cfg, valid_losses[0])
    
    # Save checkpoint
    #do_save = True
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )
        if user_parameter is not None:
            discriminator_path = "{}/discriminator_{}.pt".format(trainer.cfg.checkpoint.save_dir,epoch_itr.epoch)
            torch.save({
                'model_state_dict' : user_parameter["discriminator"].state_dict(),
                'optimizer_state_dict': user_parameter["d_optimizer"].state_dict(),
            }, discriminator_path)
            

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
    user_parameter,
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample, user_parameter)

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    generator_option = ['data-bin/PhoMT_iwlst32k.tokenized.vi-en',
                        '--arch', 'transformer',
                        '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)',
                        '--reset-optimizer',
                        '--lr', '0.0005', '--clip-norm', '0.0',
                        #'--dropout', '0.3',
                        #'--label-smoothing', '0.1',
                        #'--seed', '2048',
                        #'--max-tokens', '200',
                        '--batch-size', '32', #16
                        #'--max-epoch', '33',
                        '--lr-scheduler', 'inverse_sqrt',
                        '--weight-decay', '0.0',
                        '--user-dir', './user_dir',   
                        '--criterion', 'ac_cross_entropy',
                        '--task','RL_Translation',
                        '--max-update', '800000', '--warmup-updates', '4000', '--warmup-init-lr' ,'1e-07',
                        '--no-progress-bar',
                        '--bpe','subword_nmt',
                        '--eval-bleu',
                        '--eval-bleu-args', '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}',
                        '--eval-bleu-remove-bpe',
                        '--eval-bleu-detok', 'moses',
                        '--best-checkpoint-metric', 'bleu',
                        '--maximize-best-checkpoint-metric',
                        '--restore-file', 'checkpoints/transformer_ac/checkpoint_best.pt',
                        '--update-freq', '1',
                        '--save-dir', 'checkpoints/transformer_ac']
    
    args = options.parse_args_and_arch(parser, input_args= generator_option, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
