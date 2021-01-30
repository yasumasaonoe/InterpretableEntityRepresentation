def load_vocab_dict(vocab_file_name, vocab_max_size=None, start_vocab_count=None, common_vocab_file_name=None):
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    if vocab_max_size:
      text = text[:vocab_max_size]
    if common_vocab_file_name:
        print('==> adding common training set types')
        print('==> before:', len(text))
        with open(common_vocab_file_name, 'r') as fc:
            common = [x.strip() for x in fc.readlines()]
        print('==> common:', len(common))
        text = list(set(text + common))
        print('==> after:', len(text))
    if start_vocab_count:
      file_content = dict(zip(text, range(0 + start_vocab_count, len(text) + start_vocab_count)))
    else:
      file_content = dict(zip(text, range(0, len(text))))
  return file_content


CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = '[PAD]'

# Check these paths!
BASE_PATH = './'
FILE_ROOT = './data'
EXP_ROOT = './model'

ANSWER_NUM_DICT = {'60k': 60000, 'ufet': 10331}

#CATEGORY_VOCAB
ANS2ID_DICT_60K = load_vocab_dict(BASE_PATH + "/ontology/wiki_types_full.txt",
                                  vocab_max_size=60000)
ANS2ID_DICT_UFET = load_vocab_dict(BASE_PATH + "/ontology/ufet_types.txt")


id2ans_60k = {v: k for k, v in ANS2ID_DICT_60K.items()}
id2ans_ufet = {v: k for k, v in ANS2ID_DICT_UFET.items()}

ID2ANS_DICT_60K = id2ans_60k
ID2ANS_DICT_UFET = id2ans_ufet
