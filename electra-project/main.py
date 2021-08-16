import os
import glob
import json
import argparse
import logging
from attrdict import AttrDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastprogress.fastprogress import master_bar, progress_bar
from transformers import ElectraModel, ElectraTokenizer, AdamW

from processor.dataloader import ner_processors as processors
from src import (
    CONFIG_CLASS,
    TOKENIZER_CLASSES,
    MODEL_FOR_TOKEN_CLASSIFICATION,
)

model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
logger = logging.getLogger(__name__)


class ElectraForNER(nn.Module):
    def __init__(self, config):
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(in_features=config.hidn_dim, out_features=config.label_num)

        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(self, input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               labels=None):

        outputs = self.electra(input_ids)
        outputs = self.dropout(outputs)
        logits = self.linear(outputs)

        if attention_mask is None:
            loss = self.loss_fct(np.argmax(logits), labels)
        else:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fct(active_logits, active_labels)

        return logits, loss


'''
1) add layer for finetuning 
2) data processing 
3) finetuning 실제 코드 2를 거쳐서 1을 시행할 수 있도록 하는 코드 작성 
'''


def main_finetune(parser):
    # argparse configs/config.json

    # data file load and data processing (dataloader 등)

    # model 불러오기

    # 학습
    '''

    for e in epoch:
        for i, batch in enumerate(dataloader): >> 데이터를 먼저 만들고 정해야됨
            # train

            # 특정 step에서 valid(evaluate)
             성능이 좋은 top-k개의 모델 저장

    '''

    with open(os.path("../config.json")) as f :
        args = AttrDict(json.load(f))

    logger.info("Training/Evaluation prarams {}".format(args))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    processor = processors[args.task](args)
    labels = processor.get_labels()
    config = CONFIG_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        num_labels=tasks_num_labels[args.task],
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(
        args.model_name_or_path,
        config=config
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Load dataset
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None

    if dev_dataset == None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use testset

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in
            sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--config_file", type=str, required=True)
    parser = parser.parse_args()

    main_finetune(parser)
