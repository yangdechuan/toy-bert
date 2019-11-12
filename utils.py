import csv


class InputExample(object):
    def __init__(self, text_a, text_b=None, label=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class Sst2Processor(object):
    def __init__(self):
        self.train_file = "data/SST-2/train.tsv"
        self.dev_file = "data/SST-2/dev.tsv"

    def get_train_examples(self):
        examples = []
        for line in self._read_tsv(self.train_file):
            examples.append(InputExample(text_a=line[0], text_b=None, label=line[1]))
        return examples

    def get_dev_examples(self):
        examples = []
        for line in self._read_tsv(self.dev_file):
            examples.append(InputExample(text_a=line[0], text_b=None, label=line[1]))
        return examples

    def get_labels(self):
        return ["0", "1"]

    @classmethod
    def _read_tsv(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as fr:
            reader = csv.reader(fr, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
        return lines[1:]


class ColaProcessor(object):
    def __init__(self):
        self.train_file = "data/CoLA/train.tsv"
        self.dev_file = "data/CoLA/dev.tsv"

    def get_train_examples(self):
        examples = []
        for line in self._read_tsv(self.train_file):
            examples.append(InputExample(text_a=line[3], text_b=None, label=line[1]))
        return examples

    def get_dev_examples(self):
        examples = []
        for line in self._read_tsv(self.dev_file):
            examples.append(InputExample(text_a=line[3], text_b=None, label=line[1]))
        return examples

    def get_labels(self):
        return ["0", "1"]

    @classmethod
    def _read_tsv(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as fr:
            reader = csv.reader(fr, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
        return lines


def convert_examples_to_features(examples, tokenizer, label_list,
                                 max_length=512):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example.text_a,
                                       example.text_b,
                                       add_special_tokens=True,
                                       max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        token_type_ids = token_type_ids + [0] * padding_length
        label = label_map[example.label]

        features.append(InputFeature(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     label=label))
    return features
