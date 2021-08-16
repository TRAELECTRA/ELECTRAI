import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import ElectraModel, ElectraTokenizer, ElectraConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from fastprogress.fastprogress import master_bar, progress_bar

from torch.utils.data import DataLoader

import argparse
import json
import os
import glob
import logging

model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")



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


from processor.dataloader import NaverNERProcessor, ner_convert_examples_to_features, ner_load_and_cache_examples

def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, ops=args.adam_epsilon)


    if os.path.isfile(os.model_path, "optimizer.pt") and \
            os.path.isfile(os.path.join(args.model_path, "scheduler.pt")):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_path, "scheduler.pt")))

    logger.info("********** Running training **********")
    logger.info(f"Num Epochs = {args.n_epoch}")

    tr_loss = 0.0

    model.zero_grad()

    mb = master_bar(range(int(args.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    args.gradient_accumulation_steps >= len(train_dataloader) == (step + 1)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step is not None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }
    preds = np.argmax(preds, axis=2)

    labels = processors[args.task](args).get_labels()

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    pad_token_label_id = CrossEntropyLoss().ignore_index

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    result = compute_metrics(args.task, out_label_list, preds_list)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))
        logger.info("\n" + show_ner_report(out_label_list, preds_list))  # Show report for each tag result
        f_w.write("\n" + show_ner_report(out_label_list, preds_list))

    return results



def main_finetune():
    # argparse configs/config.json
    args = None
    # data file load and data processing (dataloader 등)

    processor = NaverNERProcessor(args)
    labels = processor.get_labels()

    config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # 학습
    '''

    for e in epoch:
        for i, batch in enumerate(dataloader): >> 데이터를 먼저 만들고 정해야됨
            # train

            # 특정 step에서 valid(evaluate)
             성능이 좋은 top-k개의 모델 저장

    '''


if __name__ == '__main__':
    main_finetune()
