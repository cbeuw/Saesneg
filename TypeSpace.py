from collections import defaultdict
from operator import itemgetter
from typing import *


class TypeSpace:
    def __init__(self, name):
        self.words_freq: DefaultDict[str, int] = defaultdict(int)
        self.name: str = name
        self.encodings: DefaultDict[str, bytes]

    def add_word(self, word: str):
        self.words_freq[word] += 1

    def transfer_common(self, target: 'TypeSpace') -> Set[str]:
        common_words = self.word_set() & target.word_set()
        for word in common_words:
            target.add_count(word, self.pop_word(word))
        return common_words

    def add_count(self, word: str, delta: int):
        self.words_freq[word] += delta

    def pop_word(self, word: str) -> int:
        return self.words_freq.pop(word)

    def word_set(self) -> Set[str]:
        return set(self.words_freq)

    def count_unique_word(self) -> int:
        return len(self.words_freq)

    def total_count(self) -> int:
        return sum(self.words_freq.values())

    def weight_order(self) -> List[Tuple[str, float]]:  # descending
        total: int = self.total_count()
        weights: List[Tuple[str, float]] = [(word, count / total) for word, count in self.words_freq.items()]
        return sorted(weights, key=itemgetter(1), reverse=True)
