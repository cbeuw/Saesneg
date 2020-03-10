import multiprocessing
import random
import re
from operator import itemgetter
from typing import *

from nltk import pos_tag
from nltk import word_tokenize, data
from nltk.tokenize.util import align_tokens

from TypeSpace import TypeSpace

POS = str
Span = Tuple[int, int]

apostrophised = {
    "wouldn't": "!MD",
    "shouldn't": "!MD",
    "couldn't": "!MD",
    "oughtn't": "!MD",
    "mustn't": "!MD",
    "needn't": "!MD",
    "won't": "!MD",
    "shan't": "!MD",
    "can't": "!MD",

    "would've": "COND",
    "should've": "COND",
    "could've": "COND",
    "ought've": "COND",

    "isn't": "!COPZ",
    "wasn't": "!COPZ",
}

literalised_pos = {"PRP", "PRP$", "DT"}


class TaggedToken:
    __slots__ = ['word', 'tag', 'capitalised', 'trailing_space']

    def __init__(self, word: str, tag: POS, spaced: bool):
        self.word: str = word if tag == 'NNP' or tag == 'NNPS' else word.lower()
        self.tag: POS = tag
        self.capitalised: bool = word[0].isupper()
        self.trailing_space = spaced

    def reassign_tag(self, new_tag: POS):
        self.tag = new_tag
        if not (new_tag == 'NNP' or new_tag == 'NNPS'):
            self.word = self.word.lower()


class Corpifier:
    pos_spaces: Dict[POS, TypeSpace] = {}
    literal_space = TypeSpace('LITERAL')

    tagged_sents: List[List[TaggedToken]]

    # literal_set: Set[str] = set()

    @staticmethod
    def get_usable_tokens(sent: str) -> List[TaggedToken]:
        # TODO: complexity is bad!
        # TODO: stricter linting rules?
        tokens = word_tokenize(sent)

        quote_pattern = r"``|''|\""
        quotes = [m.group() for m in re.finditer(quote_pattern, sent)]
        restored_tokens = [
            quotes.pop(0) if re.match(quote_pattern, tok) else tok
            for tok in tokens
        ]

        token_spans = align_tokens(restored_tokens, sent)
        tagged: List[str, POS] = pos_tag(tokens)

        regulars: Dict[Span, POS] = {}
        for i, span in enumerate(token_spans):
            regulars[span] = tagged[i][1] if tagged[i][1] not in literalised_pos else 'LITERAL'

        word_spans = Corpifier.span_word(sent)

        irregulars = {}  # irregular span to index
        for i, s in enumerate(token_spans):
            if s not in word_spans:
                irregulars[s] = i

        def merge_irregular_spans(spans: Dict[Span, int]) -> Set[Span]:
            spans_t = spans.items()
            shift: List[Tuple[Span, List[int]]] = []  # span of words and
            for s, i in spans_t:
                if len(shift) == 0:
                    shift.append((s, [i]))
                    continue

                if s[0] == shift[len(shift) - 1][0][1]:
                    prev = shift.pop(len(shift) - 1)
                    prev_span, ind = prev[0], prev[1]
                    new_span = (prev_span[0], s[1])
                    ind.append(i)
                    shift.append((new_span, ind))
                else:
                    shift.append((s, [i]))

            ret: Set[Span] = set()
            for span, _ in shift:
                ret.add(span)
            return ret

        merged_irregulars = merge_irregular_spans(irregulars)

        ret: List[TaggedToken] = []
        for i, w_span in enumerate(word_spans):
            tag: POS
            word: str = sent[w_span[0]:w_span[1]]
            spaced = not (i < len(word_spans) - 1 and w_span[1] == word_spans[i + 1][0])
            if w_span in merged_irregulars:
                if word.lower() in apostrophised:
                    tag = apostrophised[word.lower()]
                else:
                    tag = 'LITERAL'
            elif w_span in regulars:
                tag = regulars[w_span]
            else:
                tag = 'LITERAL'
                # print("unexpected literal: " + word + " from " + sent)
            ret.append(TaggedToken(word, tag, spaced))
        return ret

    def populate_spaces(self, text):
        sent_detector = data.load('tokenizers/punkt/english.pickle')
        sents = sent_detector.tokenize(text.strip())
        pool = multiprocessing.Pool()

        # sents = pool.map(self.clean_sent, sents)
        self.tagged_sents: List[List[TaggedToken]] = pool.map(Corpifier.get_usable_tokens, sents)
        # tagged_sents = list(map(Corpifier.get_usable_tokens, sents))

        for tagged_tokens in self.tagged_sents:
            for token in tagged_tokens:
                if token.tag == 'LITERAL':
                    self.literal_space.add_word(token.word)
                    continue

                if token.tag not in self.pos_spaces:
                    bkt = TypeSpace(token.tag)
                    bkt.add_word(token.word)
                    self.pos_spaces[token.tag] = bkt
                else:
                    self.pos_spaces[token.tag].add_word(token.word)

    # this makes sure that each word has a unique tag, so that a
    # word appearing in one type space doesn't appear in another one
    def singularise_words(self):
        spaces_ordered = list(self.pos_spaces.values())
        spaces_ordered.sort(key=lambda b: b.count_unique_word(), reverse=True)

        retagged_words: Dict[str, POS] = {}
        for b in spaces_ordered:
            # if a word appears as both a tagged word and a literal, make it literal
            retagged = b.transfer_common(self.literal_space)
            for word in retagged:
                retagged_words[word] = 'LITERAL'

        for i, bkt in enumerate(spaces_ordered):
            # if a word appears in two type spaces, move it from the larger type space into the smaller one
            for j in range(len(spaces_ordered) - 1, i):
                retagged = bkt.transfer_common(spaces_ordered[j])
                for word in retagged:
                    retagged_words[word] = spaces_ordered[j].name

        # if a type space has 0 word in it, delete it. if a type space has only 1 word in it, mark that one word
        # as literal and delete it
        for bkt in spaces_ordered:
            if bkt.count_unique_word() == 0:
                self.pos_spaces.pop(bkt.name)
            elif bkt.count_unique_word() == 1:
                lone_word = bkt.word_set().pop()
                self.literal_space.add_count(lone_word, bkt.words_freq[lone_word])
                retagged_words[lone_word] = 'LITERAL'
                self.pos_spaces.pop(bkt.name)

        for sent in self.tagged_sents:
            for tok in sent:
                if tok.word in retagged_words:
                    tok.reassign_tag(retagged_words[tok.word])

    def corpify(self, text: str):
        self.populate_spaces(text)
        self.singularise_words()
        for i in range(0, 100):
            print(self.make_rand_nonsense())

    def make_rand_nonsense(self) -> str:
        tagged_sent: List[TaggedToken] = random.choice(self.tagged_sents)
        ret: str = ""
        for tok in tagged_sent:
            if tok.tag == 'LITERAL':
                ret += tok.word if not tok.capitalised else tok.word.capitalize()
            else:
                word: str = random.choice(tuple(self.pos_spaces[tok.tag].word_set()))
                ret += word if not tok.capitalised else word.capitalize()
            if tok.trailing_space:
                ret += " "
        return ret

    @staticmethod
    def span_word(sent: str) -> List[Span]:
        # All punctuations are in individual groups. Ellipses are grouped together
        # Apostrophes at the beginning and end of sentences are in individual groups
        # Apostrophes followed by an s are grouped with the word ending in s
        # Alphanumeric characters, underscore and apostrophes are words grouped together
        return [m.span() for m in re.finditer(r"\.{3}|[?,!:'();.\"/]|[\w—-]+s'|[\w—-]+'[\w—-]+|[\w—-]+", sent)]

    # count unique words of each tag except for LITERAL
    @staticmethod
    def count_usable_tags(tagged_sents: List[List[TaggedToken]]) -> List[Tuple[POS, int]]:
        usable_tags: Dict[POS, Set[str]]
        usable_tags = {}
        for sent in tagged_sents:
            for t in sent:
                if t.tag == 'LITERAL':
                    continue
                if t.tag not in usable_tags:
                    usable_tags[t.tag] = set()
                else:
                    usable_tags[t.tag].add(t.word)
        tags_count = [(item[0], len(item[1])) for item in usable_tags.items()]
        return sorted(tags_count, key=itemgetter(1), reverse=True)


if __name__ == '__main__':
    text = open('1984', encoding='utf-8').read()
    text += open('The_Bell_Jar', encoding='utf-8').read()
    corp = Corpifier()
    corp.corpify(text)
    tags_count = [(item[0], item[1].count_unique_word()) for item in corp.pos_spaces.items()]
    print(sorted(tags_count, key=itemgetter(1), reverse=True))
    print(('LITERAL', corp.literal_space.count_unique_word()))
    # print(corp.literal_space.words_freq.keys())
    # Space = Codifier(corp.pos_spaces['.'])
    # for c in Space.codes:
    # print(bin(int.from_bytes(c.value, byteorder='big')) + str(c.words_stack))
