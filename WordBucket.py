from collections import defaultdict
from typing import *


class WordBucket:
    def __init__(self, name):
        self.words_freq: DefaultDict[str, int]
        self.words_freq = defaultdict(int)
        self.name = name


    def add_word(self, word: str):
        self.words_freq[word] += 1

    def transfer_common(self, target: 'WordBucket'):
        common_words = self.word_set() & target.word_set()
        map(lambda w: target.add_count(w, self.pop_word(w)), common_words)

    def add_count(self, word: str, delta: int):
        self.words_freq[word] += delta

    def pop_word(self, word: str) -> int:
        return self.words_freq.pop(word)

    def word_set(self) -> Set[str]:
        return set(self.words_freq)

    def count_unique_word(self) -> int:
        return len(self.words_freq)