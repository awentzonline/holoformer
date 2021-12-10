import datetime
import os

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from holoformer.models import hrr
from .app import App


class DocSearchApp(App):
    keys = (
        0,  # token
        1,  # doc
    )

    def run(self):
        self.make_docs()
        try:
            self.repl_loop()
        except KeyboardInterrupt:
            pass

    def repl_loop(self):
        while True:
            print('*' * 20)
            terms = input('Search:')
            if terms == 'rendertb':
                print('rendering to tensorboard...')
                self.render_tensorboard_embedding()
                continue
            embedded = self.embed_text(terms)
            embedded_doc = embedded.sum(0)
            y = hrr.unbind(self.corpus, embedded_doc)
            y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
            cosine = torch.matmul(self.doc_ids, y.unsqueeze(1)).squeeze(-1)
            best_matches = torch.topk(cosine, 5)[1]
            print(best_matches)
            for i in best_matches:
                best_doc = self.raw_docs[i]
                best_doc = best_doc[:100]
                print(best_doc, '\n')

    def embed_text(self, text):
        tokens = self.tokenizer.encode(text)
        embedded = self.vocab[tokens]
        return embedded

    def render_tensorboard_embedding(self):
        run_id = str(datetime.datetime.now())
        sw = SummaryWriter(log_dir=os.path.join(self.args.logdir, run_id))
        labels = [d[:50].replace('\n', ' ') for d in self.raw_docs]
        print('Labels', len(labels), 'Docs', self.docs.shape)
        sw.add_embedding(self.docs, metadata=labels)

    def make_docs(self):
        df = pd.read_pickle(self.args.corpus)
        raw_docs = df[self.args.text_col].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer)

        num_docs = len(raw_docs)
        max_doc_len = max(map(len, raw_docs))
        vocab_size = len(self.tokenizer)

        self.key_ids = hrr.init((len(self.keys), self.args.dims))
        #self.position_ids = hrr.init((max_doc_len, self.args.dims))
        self.vocab = hrr.init((vocab_size, self.args.dims))
        #self.vocab = hrr.bind(self.vocab, self.key_ids[0])
        self.doc_ids = hrr.init((num_docs, self.args.dims))

        self.docs = []
        self.raw_docs = []
        for raw_doc in raw_docs:
            self.raw_docs.append(raw_doc)
            try:
                tokens = self.tokenizer.encode(raw_doc)
            except:
                print(t)
                raise
            tokens_embedded = self.vocab[tokens]
            doc = tokens_embedded.sum(0)
            self.docs.append(doc)
        print('Num docs:', len(self.docs))
        self.docs = torch.stack(self.docs)
        self.corpus = hrr.bind(self.docs, self.doc_ids).sum(0)
        print('Corpus shape', self.corpus.shape)

    @classmethod
    def add_args_to_parser(self, p):
        p.add_argument('corpus', help='Name of corpus dataframe')
        p.add_argument('--text_col', default='text')
        p.add_argument('--tokenizer', default='bert-base-uncased')
        p.add_argument('--dims', default=100, type=int)
        p.add_argument('--logdir', default='logs')


if __name__ == '__main__':
    DocSearchApp.run_from_cli()
