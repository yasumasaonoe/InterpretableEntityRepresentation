import argparse
import torch
import torch.nn as nn

from transformers import AutoConfig
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer


TRANSFORMER_MODELS = {
  'bert-base-uncased': (BertModel, BertTokenizer),
  'bert-large-uncased': (BertModel, BertTokenizer),
  'bert-large-uncased-whole-word-masking': (BertModel, BertTokenizer),
  'roberta-base': (RobertaModel, RobertaTokenizer),
  'roberta-large': (RobertaModel, RobertaTokenizer)
}


class SimpleDecoder(nn.Module):
  def __init__(self, output_dim, answer_num):
    super(SimpleDecoder, self).__init__()
    self.answer_num = answer_num
    self.linear = nn.Linear(output_dim, answer_num, bias=False)

  def forward(self, inputs, output_type=None):
    output_embed = self.linear(inputs)
    return output_embed


class ModelBase(nn.Module):
  """Base model class."""
  def __init__(self):
    super(ModelBase, self).__init__()
    self.loss_func = nn.BCEWithLogitsLoss()
    self.sigmoid_fn = nn.Sigmoid()

  def define_loss(self, logits, targets):
    loss = self.loss_func(logits, targets)
    return loss

  def forward(self, feed_dict):
    raise NotImplementedError


class TransformerModel(ModelBase):
  """Transformer entity typing model."""
  def __init__(self, args, answer_num):
    super(TransformerModel, self).__init__()
    print('Initializing <{}> model...'.format(args.model_type))

    _model_class, _tokenizer_class = TRANSFORMER_MODELS[args.model_type]
    self.transformer_tokenizer = _tokenizer_class.from_pretrained(args.model_type)
    self.transformer_config = AutoConfig.from_pretrained(args.model_type)
    # Load pretrained MLM (e.g., BERT, RoBERTa).
    self.encoder = _model_class.from_pretrained(args.model_type)
    # Linear layer (i.e., type embeddings).
    self.classifier = SimpleDecoder(self.transformer_config.hidden_size, answer_num)
    self.dropout = nn.Dropout(args.hidden_dropout_prob)
    self.avg_pooling = args.avg_pooling

  def forward(self, inputs, targets=None):
    # Get output from Transformer, shape = (batch size, max length, dim).
    outputs = self.encoder(
      inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      token_type_ids=inputs["token_type_ids"]
    )
    # Get mention & context representation, shape = (batch size, dim) after
    # this step.
    if self.avg_pooling:  # Averaging all hidden states
      outputs = (outputs[0] * inputs["attention_mask"].unsqueeze(-1)).sum(1)\
                / inputs["attention_mask"].sum(1).unsqueeze(-1)
    else:  # Use [CLS]
      outputs = outputs[0][:, 0, :]
    # Elementwise dropout.
    outputs = self.dropout(outputs)
    # Get logits for each type, (batch size, number of types)
    logits = self.classifier(outputs)

    if targets is not None:
      # Training mode. Compute BCE loss.
      loss = self.define_loss(logits, targets)
    else:
      # Eval mode.
      loss = None
    return loss, logits
