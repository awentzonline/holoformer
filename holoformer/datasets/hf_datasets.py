import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


class HfDatasetDataModule(pl.LightningDataModule):
    def __init__(self, dataset='wikitext/wikitext-2-raw-v1',
                 tokenizer_name='bert-base-uncased', batch_size=10,
                 max_seq_len=256, num_workers=0, **kwargs):
        super().__init__()
        self.dataset_name, self.dataset_config = dataset.split('/')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def setup(self, stage):
        self.dataset = datasets.load_dataset(self.dataset_name, self.dataset_config)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].filter(
                lambda x: len(x['text']) > 80
            )
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )
            self.dataset[split].set_format(type="torch", columns=['input_ids'])

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'], batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset['validation'], batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def convert_to_features(self, example_batch, indices=None):
        texts_or_text_pairs = example_batch['text']
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length,
            pad_to_max_length=True, truncation=True,
        )
        return features
