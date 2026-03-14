from collections import Counter

PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN = '<pad>', '<unk>', '<sos>', '<eos>'
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


def tokenize(text: str) -> list:
    return text.lower().strip().split()


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.idx2word = {i: tok for tok, i in self.word2idx.items()}

    def build(self, sentences: list):
        counter = Counter()
        for s in sentences:
            counter.update(tokenize(s))
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
        print(f'vocab size: {len(self.word2idx):,}')

    def encode(self, sentence: str) -> list:
        return [self.word2idx.get(w, UNK_IDX) for w in tokenize(sentence)]

    def decode(self, indices, skip_special=True) -> str:
        skip = set(SPECIAL_TOKENS) if skip_special else set()
        words = []
        for i in indices:
            i = i.item() if hasattr(i, 'item') else i
            w = self.idx2word.get(i, UNK_TOKEN)
            if w == EOS_TOKEN:
                break
            if w not in skip:
                words.append(w)
        return ' '.join(words)

    def __len__(self):
        return len(self.word2idx)
