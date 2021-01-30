#!/usr/bin/env python3
import argparse
import gc
import json
import numpy as np
import os
import pickle
import random
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

import transformer_constant
import transformer_data_utils
from transformer_data_utils import to_torch
from models import TransformerModel


"""
Args
"""

parser = argparse.ArgumentParser()
parser.add_argument("-model_id", help="Identifier for model")
parser.add_argument('-device', type=int, default=0, help='CUDA device')
parser.add_argument("-n_gpu", help="Number of GPUs.", type=int, default=1)
parser.add_argument("-mode",
                    help="Whether to train or test",
                    default="train",
                    choices=["train", "test"])
parser.add_argument("-local_rank",
                    type=int,
                    default=-1,
                    help="For distributed training: local_rank")

# Data
parser.add_argument(
  "-train_data",
  help="Train data",
  default="train/wiki_et_zeroshot_60k_ex_random/train_*.json")
parser.add_argument(
  "-dev_data",
  help="Dev data",
  default="validation/dev_wiki_et_zeroshot_60k_ex_random_999.json")
parser.add_argument("-eval_data", help="Test data", default="")
parser.add_argument("-goal",
                    help="category vocab size.",
                    default="60k",
                    choices=["60k", "ufet"])
parser.add_argument("-seed", help="Pytorch random Seed", default=113)
parser.add_argument("-context_window_size",
                    help="Left and right context size.",
                    default=100)

# learning
parser.add_argument("-num_epoch",
                    help="The number of epoch",
                    default=5000,
                    type=int)
parser.add_argument("-per_gpu_train_batch_size",
                    help="The batch size per GPU",
                    default=8,
                    type=int)
parser.add_argument("-per_gpu_eval_batch_size",
                    help="The batch size per GPU",
                    default=8,
                    type=int)
parser.add_argument("-learning_rate_enc",
                    help="BERT: start learning rate",
                    default=2e-5,
                    type=float)
parser.add_argument("-learning_rate_cls",
                    help="BERT: start learning rate",
                    default=1e-3,
                    type=float)
parser.add_argument("-adam_epsilon_enc",
                    default=1e-8,
                    type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("-adam_epsilon_cls",
                    default=1e-8,
                    type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("-hidden_dropout_prob",
                    help="Dropout rate",
                    default=.1,
                    type=float)
parser.add_argument("-warmup_steps",
                    default=0,
                    type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument(
  "-gradient_accumulation_steps",
  type=int,
  default=1,
  help="Number of updates steps to accumulate before performing a backward/update pass.",
)

# Model
parser.add_argument(
  "-model_type",
  default="bert-base-uncased",
  choices=[
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-large-uncased-whole-word-masking",
    "roberta-base",
    "roberta-large"
  ]
)
parser.add_argument("-threshold",
                    help="threshold",
                    default=0.5,
                    type=float)
parser.add_argument("-avg_pooling",
                    help="Averaging all hidden states instead of using [CLS].",
                    action='store_true')

# Save / log related
parser.add_argument("-save_period",
                    help="How often to save",
                    default=1000,
                    type=int)
parser.add_argument("-eval_period",
                    help="How often to run dev",
                    default=500,
                    type=int)
parser.add_argument("-log_period",
                    help="How often to save",
                    default=1000,
                    type=int)
parser.add_argument("-eval_after",
                    help="How often to run dev",
                    default=10,
                    type=int)
parser.add_argument("-load", help="Load existing model.", action='store_true')
parser.add_argument("-reload_model_name", help="")
parser.add_argument("-reload_model_name_desc", help="")


"""
Utils
"""

SIGMOID = nn.Sigmoid()


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def get_data_gen(dataname, mode, args, tokenizer):
  data_path = os.path.join(transformer_constant.FILE_ROOT, dataname)
  dataset = transformer_data_utils.DatasetLoader(data_path, args, tokenizer)
  if mode == 'train':
    data_gen = dataset.get_batch(args.train_batch_size,
                                 args.max_position_embeddings,
                                 args.num_epoch,
                                 eval_data=False)
  else:  # test mode
    data_gen = dataset.get_batch(args.eval_batch_size,
                                 args.max_position_embeddings,
                                 1,
                                 eval_data=True)
  return data_gen


def get_all_datasets(args, tokenizer):
  train_gen_list = []
  if args.mode in ['train']:
    train_gen_list.append(
      get_data_gen(args.train_data, 'train', args, tokenizer))
  return train_gen_list


def get_datasets(data_lists, args, tokenizer):
  data_gen_list = []
  for dataname, mode in data_lists:
    data_gen_list.append(get_data_gen(dataname, mode, args, tokenizer))
  return data_gen_list


def evaluate_data(batch_num, dev_fname, model, args, device):
  model.eval()
  dev_gen = get_data_gen(dev_fname, 'test', args, model.transformer_tokenizer)
  gold_pred = []
  eval_loss = 0.
  total_ex_count = 0
  for batch in tqdm(dev_gen):
    total_ex_count += len(batch['targets'])
    inputs, targets = to_torch(batch, device)
    loss, output_logits = model(inputs, targets)
    output_index = get_output_index(output_logits, threshold=args.threshold)
    gold_pred += get_gold_pred_str(output_index,
                                   batch['targets'].data.cpu().clone(),
                                   args.goal)
    eval_loss += loss.clone().item()
  eval_str = get_eval_string(gold_pred)
  _, _, _, _, _, macro_f1 = macro(gold_pred)
  eval_loss_str = 'Eval loss: {0:.7f} at step {1:d}'.format(eval_loss,
                                                            batch_num)
  print('==>  EVAL: seen ' + repr(total_ex_count) + ' examples.')
  print(eval_loss_str)
  print(gold_pred[:3])
  print('==> ' + eval_str)
  model.train()
  dev_gen = None
  return eval_loss, macro_f1


def f1(p, r):
  if r == 0.:
    return 0.
  return 2 * p * r / float(p + r)


def macro(true_and_prediction):
  num_examples = len(true_and_prediction)
  p = 0.
  r = 0.
  pred_example_count = 0.
  pred_label_count = 0.
  gold_label_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
      pred_label_count += len(predicted_labels)
      per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
      p += per_p
    if len(true_labels):
      gold_label_count += 1
      per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
      r += per_r
  if pred_example_count > 0:
    precision = p / pred_example_count
  if gold_label_count > 0:
    recall = r / gold_label_count
  avg_elem_per_pred = pred_label_count / pred_example_count
  return num_examples, pred_example_count, avg_elem_per_pred, precision, \
         recall, f1(precision, recall)


def micro(true_and_prediction):
  num_examples = len(true_and_prediction)
  num_predicted_labels = 0.
  num_true_labels = 0.
  num_correct_labels = 0.
  pred_example_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
    num_predicted_labels += len(predicted_labels)
    num_true_labels += len(true_labels)
    num_correct_labels += len(
      set(predicted_labels).intersection(set(true_labels)))
  if pred_example_count == 0:
    return num_examples, 0, 0, 0, 0, 0
  precision = num_correct_labels / num_predicted_labels
  recall = num_correct_labels / num_true_labels
  avg_elem_per_pred = num_predicted_labels / pred_example_count
  return num_examples, pred_example_count, avg_elem_per_pred, precision, \
         recall, f1(precision, recall)


def load_model(
  reload_model_name, save_dir, model_id, model, optimizer_enc=None,
  optimizer_cls=None, scheduler_enc=None, scheduler_cls=None
):
  if reload_model_name:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, reload_model_name)
  else:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, model_id)
  checkpoint = torch.load(model_file_name)
  model.load_state_dict(checkpoint['state_dict'])
  if optimizer_enc and optimizer_cls:  # Continue training
  #if optimizer_enc and optimizer_cls and scheduler_enc and scheduler_cls:  # Continue training
    optimizer_enc.load_state_dict(checkpoint['optimizer_enc'])
    optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
    # scheduler_enc.load_state_dict(checkpoint['scheduler_enc'])
    # scheduler_cls.load_state_dict(checkpoint['scheduler_cls'])
  else:  # Test
    total_params = 0
    # Log params
    for k in checkpoint['state_dict']:
      elem = checkpoint['state_dict'][k]
      param_s = 1
      for size_dim in elem.size():
        param_s = size_dim * param_s
      #print(k, elem.size())
      total_params += param_s
    param_str = ('Number of total parameters..{0:d}'.format(total_params))
    print(param_str)
  print('Loading model from ... {0:s}'.format(model_file_name))


def get_output_index(outputs, threshold=0.5):
  """
  Given outputs from the decoder, generate prediction index.
  """
  pred_idx = []
  outputs = SIGMOID(outputs).data.cpu().clone()
  for single_dist in outputs:
    single_dist = single_dist.numpy()
    arg_max_ind = np.argmax(single_dist)
    pred_id = [arg_max_ind]
    pred_id.extend(
      [i for i in range(len(single_dist))
       if single_dist[i] > threshold and i != arg_max_ind])
    pred_idx.append(pred_id)
  return pred_idx


def get_gold_pred_str(pred_idx, gold, goal):
  """
  Given predicted ids and gold ids, generate a list of (gold, pred) pairs of length batch_size.
  """
  if goal == '60k':
    id2word_dict = transformer_constant.ID2ANS_DICT_60K
  elif goal == 'ufet':
    id2word_dict = transformer_constant.ID2ANS_DICT_UFET
  else:
    print('ERROR: Invalid input...' + goal)
    raise
  gold_strs = []
  for gold_i in gold:
    gold_strs.append(
      [id2word_dict[i] for i in range(len(gold_i)) if gold_i[i] == 1])
  pred_strs = []
  for pred_idx1 in pred_idx:
    pred_strs.append([(id2word_dict[ind]) for ind in pred_idx1])
  else:
    return list(zip(gold_strs, pred_strs))


def get_eval_string(true_prediction):
  """
  Given a list of (gold, prediction)s, generate output string.
  """
  count, pred_count, avg_pred_count, p, r, f1 = micro(true_prediction)
  _, _, _, ma_p, ma_r, ma_f1 = macro(true_prediction)
  output_str = "Eval: {0} {1} {2:.3f} P:{3:.3f} R:{4:.3f} F1:{5:.3f} Ma_P:{6:.3f} Ma_R:{7:.3f} Ma_F1:{8:.3f}".format(
    count, pred_count, avg_pred_count, p, r, f1, ma_p, ma_r, ma_f1)
  accuracy = sum([set(y) == set(yp) for y, yp in true_prediction]) * 1.0 / len(true_prediction)
  output_str += '\t Dev accuracy: {0:.1f}%'.format(accuracy * 100)
  return output_str


"""
Training 
"""

def _train(args, model, device):
  """Training function. """
  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  args.eval_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  print('==> Loading data generator... ')
  train_gen_list = get_all_datasets(args, model.transformer_tokenizer)
  print('done. {} data gen(s)'.format(len(train_gen_list)))
  print('Model Type: {}'.format(args.model_type))
  total_loss = 0.
  batch_num = 0
  best_macro_f1 = 0.
  start_time = time.time()
  init_time = time.time()
  #t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
  print('Total {} named params.'.format(
    len([n for n, p in model.named_parameters()])))
  no_decay = ["bias", "LayerNorm.weight"]
  classifier_param_name = ["classifier.linear.weight"]
  encoder_parameters = [
    {
      "params": [p for n, p in model.named_parameters()
                 if not any(nd in n for nd in no_decay)
                 and n not in classifier_param_name],
      "weight_decay": 0.0  #args.weight_decay,
    },
    {
      "params": [p for n, p in model.named_parameters()
                 if any(nd in n for nd in no_decay)
                 and n not in classifier_param_name],
      "weight_decay": 0.0
    },
  ]
  classifier_parameters = [
    {
      "params": [p for n, p in model.named_parameters()
                 if n in classifier_param_name],
      "weight_decay": 0.0
    },
  ]
  print(
    'Encoder {}, Classifier {}'.format(
      sum([len(p['params']) for p in encoder_parameters]),
      sum([len(p['params']) for p in classifier_parameters])
    )
  )
  # Use different optimizers for encoder params and classifier params.
  optimizer_enc = AdamW(
    encoder_parameters, lr=args.learning_rate_enc, eps=args.adam_epsilon_enc)
  optimizer_cls = AdamW(
    classifier_parameters, lr=args.learning_rate_cls, eps=args.adam_epsilon_cls)
  # scheduler_enc = get_linear_schedule_with_warmup(
  #   optimizer_enc, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
  # )
  # scheduler_cls = get_linear_schedule_with_warmup(
  #   optimizer_enc, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
  # )
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
  if args.load:
    load_model(args.reload_model_name,
               transformer_constant.EXP_ROOT,
               args.model_id,
               model,
               optimizer_enc,
               optimizer_cls)
  optimizer_enc.zero_grad()
  optimizer_cls.zero_grad()
  set_seed(args)
  while True:
    batch_num += 1  # single batch composed of all train signal passed by.
    for data_gen in train_gen_list:
      try:
        batch = next(data_gen)
        inputs, targets = to_torch(batch, device)
      except StopIteration:
        print('Done!')
        torch.save(
          {
            'state_dict': model.state_dict(),
            'optimizer_cls': optimizer_cls.state_dict(),
            'optimizer_enc': optimizer_enc.state_dict(),
            # 'scheduler_cls': scheduler_cls.state_dict(),
            # 'scheduler_enc': scheduler_enc.state_dict(),
            'args': args
          },
          '{0:s}/{1:s}.pt'.format(transformer_constant.EXP_ROOT, args.model_id)
        )
        return
      model.train()
      if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
          batch["token_type_ids"] if args.model_type
                                     in ["bert", "xlnet", "albert"] else None
        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
      loss, output_logits = model(inputs, targets)
      inputs, targets = None, None
      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
      loss.backward()
      total_loss += loss.item()
      if batch_num % args.gradient_accumulation_steps == 0:
        optimizer_enc.step()
        optimizer_cls.step()
        # scheduler_enc.step()
        # scheduler_cls.step()
        optimizer_enc.zero_grad()
        optimizer_cls.zero_grad()

        if batch_num % args.log_period == 0 and batch_num > 0:
          gc.collect()
          cur_loss = float(1.0 * loss.clone().item())
          elapsed = time.time() - start_time
          train_loss_str = ('|loss {0:3f} | at {1:d}step | @ {2:.2f} ms/batch'.format(cur_loss, batch_num,
                                                                                      elapsed * 1000 / args.log_period))
          start_time = time.time()
          print(train_loss_str)

        if batch_num % args.eval_period == 0 and batch_num > 0:
          output_index = get_output_index(output_logits, threshold=args.threshold)
          gold_pred_train = get_gold_pred_str(
            output_index,
            batch['targets'].data.cpu().clone(),
            args.goal)
          print(gold_pred_train[:10])
          accuracy = sum([set(y) == set(yp) for y, yp in gold_pred_train]) * \
                     1.0 / len(gold_pred_train)
          train_acc_str = '==> Train accuracy: {0:.1f}%'.format(accuracy * 100)
          print(train_acc_str)

    if batch_num % args.eval_period == 0 and batch_num > args.eval_after:
      # Evaluate Loss on the Turk Dev dataset.
      print('---- eval at step {0:d} ---'.format(batch_num))
      _, macro_f1 = evaluate_data(batch_num, args.dev_data, model, args, device)
      if best_macro_f1 < macro_f1:
        best_macro_f1 = macro_f1
        save_fname = '{0:s}/{1:s}_best.pt'.format(
          transformer_constant.EXP_ROOT, args.model_id)
        torch.save(
          {
            'state_dict': model.state_dict(),
            'optimizer_cls': optimizer_cls.state_dict(),
            'optimizer_enc': optimizer_enc.state_dict(),
            # 'scheduler_cls': scheduler_cls.state_dict(),
            # 'scheduler_enc': scheduler_enc.state_dict(),
            'args': args
          },
          save_fname
        )
        print(
          'Total {0:.2f} minutes have passed, saving at {1:s} '.format(
            (time.time() - init_time) / 60, save_fname))

    if batch_num % args.save_period == 0 and batch_num > 30000:
      save_fname = '{0:s}/{1:s}_{2:d}.pt'.format(
        transformer_constant.EXP_ROOT, args.model_id, batch_num)
      torch.save(
        {
          'state_dict': model.state_dict(),
          'optimizer_cls': optimizer_cls.state_dict(),
          'optimizer_enc': optimizer_enc.state_dict(),
          # 'scheduler_cls': scheduler_cls.state_dict(),
          # 'scheduler_enc': scheduler_enc.state_dict(),
          'args': args
        },
        save_fname
      )
      print(
        'Total {0:.2f} minutes have passed, saving at {1:s} '.format(
          (time.time() - init_time) / 60, save_fname))


"""
Test
"""

def _test(args, model, device):
  assert args.load
  test_fname = args.eval_data
  args.eval_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  data_gens = get_datasets([(test_fname, 'test')],
                           args,
                           model.transformer_tokenizer)
  model.eval()
  load_model(args.reload_model_name,
             transformer_constant.EXP_ROOT,
             args.model_id,
             model)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
    print("==> use", torch.cuda.device_count(), "GPUs.")
  for name, dataset in [(test_fname, data_gens[0])]:
    print('Processing... ' + name)
    total_gold_pred = []
    total_annot_ids = []
    total_probs = []
    total_ys = []
    for batch_num, batch in enumerate(dataset):
      if batch_num % 1 == 0:
        print(batch_num)
      if not isinstance(batch, dict):
        print('==> batch: ', batch)
      inputs, targets = to_torch(batch, device)
      annot_ids = batch.pop('ex_ids')
      if args.args.n_gpu > 1:
        output_logits = model(inputs, targets)
      else:
        _, output_logits = model(inputs)
      output_index = get_output_index(output_logits, threshold=args.threshold)
      output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
      y = inputs['targets'].data.cpu().clone().numpy()
      gold_pred = get_gold_pred_str(output_index, y, args.goal)
      total_probs.extend(output_prob)
      total_ys.extend(y)
      total_gold_pred.extend(gold_pred)
      total_annot_ids.extend(annot_ids)
    pickle.dump({'gold_id_array': total_ys, 'pred_dist': total_probs},
                open(transformer_constant.FILE_ROOT + '/outputs/{0:s}.pkl'.format(args.model_id), "wb"))
    print(len(total_annot_ids), len(total_gold_pred))
    with open(transformer_constant.FILE_ROOT + '/outputs/{0:s}.json'.format(args.model_id), 'w') as f_out:
      output_dict = {}
      counter = 0
      for a_id, (gold, pred) in zip(total_annot_ids, total_gold_pred):
        output_dict[a_id] = {"gold": gold, "pred": pred}
        counter += 1
      json.dump(output_dict, f_out)
    #eval_str = get_eval_string(total_gold_pred)
    eval_str = 'none'
    print(eval_str)


def main():
  args = parser.parse_args()
  # Lower text for BERT uncased models
  args.do_lower = True if 'uncased' in args.model_type else False
  # Setup CUDA, GPU & distributed training
  assert torch.cuda.is_available()
  if args.local_rank == -1:
    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device
  set_seed(args)
  # Load pretrained model and tokenizer
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
  model = TransformerModel(args, transformer_constant.ANSWER_NUM_DICT[args.goal])
  if args.local_rank == 0:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
  model.to(args.device)
  #print(model)
  args.max_position_embeddings = model.transformer_config.max_position_embeddings
  print('-' * 80)
  for k, v in vars(args).items():
    print(k, ':', v)
  print('-' * 80)
  if args.mode == 'train':
    print('==> mode: train')
    _train(args, model, device)
  elif args.mode == 'test':
    print('==> mode: test')
    _test(args, model, device)
  else:
    raise ValueError("invalid value for 'mode': {}".format(args.mode))


if __name__ == '__main__':
  main()
