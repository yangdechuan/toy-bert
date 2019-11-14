import argparse
import random
import logging

from tqdm import tqdm, trange
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import AdamW, WarmupLinearSchedule

from utils import convert_examples_to_features, Sst2Processor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
fh = logging.FileHandler("log.txt", mode="a")
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
sh.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(sh)
logger.addHandler(fh)

parser = argparse.ArgumentParser()
parser.add_argument("--bert-model", default="bert-base-uncased", type=str)
parser.add_argument("--max-seq-length", default=128, type=int)
parser.add_argument("--train-batch-size", default=32, type=int)
parser.add_argument("--eval-batch-size", default=32, type=int)
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
parser.add_argument("--warmup-steps", default=0, type=int)
parser.add_argument("--logging-steps", default=100, type=int)
parser.add_argument("--weight-decay", default=0, type=float)
parser.add_argument("--learning-rate", default=5e-5, type=float)
parser.add_argument("--max-grad-norm", default=1.0, type=float)
parser.add_argument("--seed", default=233, type=int)
parser.add_argument("--use-tensorboard", action="store_true")
args = parser.parse_args()


def load_examples(tokenizer, mode="train"):
    sst2 = Sst2Processor()
    # examples: list of InputExample objects
    if mode == "train":
        examples = sst2.get_train_examples()
    elif mode == "dev":
        examples = sst2.get_dev_examples()
    else:
        examples = None
    # features: list of InputFeatures
    features = convert_examples_to_features(examples, tokenizer, sst2.get_labels(), max_length=args.max_seq_length)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.int64)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.int64)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.int64)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.int64)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def train(train_dataset, model, device, evaluate_during_training=False, eval_dataset=None):
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_sampler)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    logger.info("***** Running training *****")
    logger.info("example number: {}".format(len(train_dataset)))
    logger.info("batch size: {}".format(args.train_batch_size))
    logger.info("epoch size: {}".format(args.epochs))
    logger.info("gradient accumulation step number: {}".format(args.gradient_accumulation_steps))
    logger.info("total step number: {}".format(t_total))
    logger.info("warmup step number: {}".format(args.warmup_steps))
    global_step = 0
    tr_loss, logging_loss = 0, 0
    for epoch in range(1, args.epochs + 1):
        logger.info("====Epoch {}".format(epoch))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            if global_step % args.logging_steps == 0:
                if args.use_tensorboard:
                    if hasattr(model, "module"):
                        tb_writer.add_histogram("classifier.weight", model.module.classifier.weight, global_step)
                        tb_writer.add_histogram("classifier.bias", model.module.classifier.bias, global_step)
                    else:
                        tb_writer.add_histogram("classifier.weight", model.classifier.weight, global_step)
                        tb_writer.add_histogram("classifier.bias", model.classifier.bias, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss
                if evaluate_during_training:
                    result = evaluate(eval_dataset, model, device)
                    logger.info("eval accuracy: {}, eval loss: {}".format(result["acc"], result["loss"]))
                    for k, v in result.items():
                        tb_writer.add_scalar("eval_{}".format(k), v, global_step)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
    logger.info("***** Finish Training! *****")


def evaluate(eval_dataset, model, device):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, sampler=eval_sampler)
    eval_loss = 0
    nb_eval_steps = 0
    preds = None
    labels = None
    model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[0:2]

        if torch.cuda.device_count() > 1:
            tmp_eval_loss = tmp_eval_loss.mean().item()
        eval_loss += tmp_eval_loss
        nb_eval_steps += 1
        if preds is None:
            preds = logits.cpu().numpy()
            labels = inputs["labels"].cpu().numpy()
        else:
            preds = np.append(preds, logits.cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    eval_loss = eval_loss / nb_eval_steps
    acc = metrics.accuracy_score(labels, preds)
    result = {
        "acc": acc,
        "loss": eval_loss,
    }
    return result


def main():
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = BertForSequenceClassification.from_pretrained(args.bert_model)
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    train_dataset = load_examples(tokenizer, mode="train")
    eval_dataset = load_examples(tokenizer, mode="dev")

    train(train_dataset, model, device,
          evaluate_during_training=True, eval_dataset=eval_dataset)
    # evaluate(eval_dataset, model, device)


if __name__ == "__main__":
    main()
