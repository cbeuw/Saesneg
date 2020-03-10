import math
from typing import *

import TypeSpace


class Codifier:
    def __init__(self, bucket: TypeSpace):
        self.weights_stack: List[Tuple[str, float]] = bucket.weight_order()
        self.exponent = math.floor(math.log2(bucket.count_unique_word()))  # num of bits
        self.encoding_size = 1 << self.exponent
        # probability of uniform distribution in range of 0 to encoding_size
        self.unit_probability = 1 / self.encoding_size

        bytes_size = math.ceil(self.exponent / 8)
        self.codes: List[CodeBucket] = [CodeBucket(value.to_bytes(bytes_size, 'big')) for value in
                                        range(0, self.encoding_size)]
        self.codify()

    def codify(self):
        cur_code_value: int = 0
        for cur_code_value in range(0, self.encoding_size):
            if len(self.weights_stack) == 0:
                break
            code_total_weight: float = 0
            # we want unit_probability * 1/size
            while code_total_weight < self.unit_probability and len(self.weights_stack) > 0:
                word, weight = self.weights_stack.pop()
                self.codes[cur_code_value].add_word(word)
                code_total_weight += weight
        if cur_code_value == self.encoding_size:
            return

        def squash(i: int):
            shift: str = self.codes[i].pop_word()
            i += 1
            while i < self.encoding_size and len(self.codes[i]) > 0:
                to_next = self.codes[i].pop_word()
                self.codes[i].add_word(shift)
                shift = to_next
                i += 1
            self.codes[i].add_word(shift)

        for i in range(cur_code_value, self.encoding_size):
            squash_start: int = 0
            while squash_start < self.encoding_size - 1 and len(self.codes[squash_start]) == len(
                    self.codes[squash_start + 1]):
                squash_start += 1
            squash(squash_start)


class CodeBucket:
    __slots__ = ["value", "words_stack"]

    def __init__(self, value: bytes):
        self.value = value
        self.words_stack: List[str] = []

    # must be added increasingly
    def add_word(self, word: str):
        self.words_stack.append(word)

    def pop_word(self) -> str:
        # last in first out because the word last in will have the highest weight. moving it to the next bucket,
        # which is presumably smaller, will result in the least error
        return self.words_stack.pop()

    def __len__(self) -> int:
        return len(self.words_stack)
