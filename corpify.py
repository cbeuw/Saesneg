import multiprocessing
import re
from operator import itemgetter, getitem
from typing import *

from nltk import word_tokenize, data
from nltk import pos_tag

from nltk.tokenize.util import align_tokens

from WordBucket import WordBucket

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




class TaggedToken:
    __slots__ = ['word', 'tag', 'span', 'capitalised']

    def __init__(self, word: str, tag: POS, span: Span):
        self.word = word if tag == 'NNP' or tag == 'NNPS' else word.lower()
        self.tag = tag
        self.span = span
        self.capitalised = word[0].isupper()


class Corpifier:
    buckets: Dict[POS, WordBucket]
    buckets = {}
    literal_bucket = WordBucket('LITERAL')
    #literal_set: Set[str] = set()

    @staticmethod
    def get_usable_tokens(sent: str) -> List[TaggedToken]:
        # TODO: complexity is bad!
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
            regulars[span] = tagged[i][1]

        word_spans = Corpifier.span_word(sent)

        irregulars = {}  # irregular span to index
        for i, s in enumerate(token_spans):
            if s not in word_spans:
                irregulars[s] = i

        def word_of(span: Span) -> str:
            return sent[span[0]:span[1]]

        '''
        def merge_nw_span(spans: Dict[Span, int], tags: List[Tuple[str, POS]]) -> List[Tuple[str, List[POS]]]:
            spans_t = spans.items()
            shift: List[Tuple[Span, List[int]]] = []
            for s, i in spans_t:
                if len(shift) == 0:
                    shift.append((s, [i]))
                    continue

                if s[0] == shift[len(shift)-1][0][1]:
                    prev = shift.pop(len(shift)-1)
                    prev_span, ind = prev[0], prev[1]
                    new_span = (prev_span[0], s[1])
                    ind.append(i)
                    shift.append((new_span, ind))
                else:
                    shift.append((s, [i]))

            ret = []
            for _, inds in shift:
                corres_tags: List[Tuple[str, POS]] = list(map(lambda index: getitem(tags, index), inds))
                merged_word: str = ""
                poses: List[POS] = []
                for word, pos in corres_tags:
                    merged_word += word
                    poses.append(pos)
                ret.append((merged_word, poses))
            return ret
        '''


        def merge_irregular_spans(spans: Dict[Span, int]) -> Set[Span]:
            spans_t = spans.items()
            shift: List[Tuple[Span, List[int]]] = []    # span of words and
            for s, i in spans_t:
                if len(shift) == 0:
                    shift.append((s, [i]))
                    continue

                if s[0] == shift[len(shift)-1][0][1]:
                    prev = shift.pop(len(shift)-1)
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
        for w_span in word_spans:
            tag: POS
            word: str = word_of(w_span)
            if w_span in merged_irregulars:
                if word.lower() in apostrophised:
                    tag = apostrophised[word.lower()]
                else:
                    tag = 'LITERAL'
            elif w_span in regulars:
                tag = regulars[w_span]
            else:
                tag = 'LITERAL'
                #print("unexpected literal: " + word + " from " + sent)
            ret.append(TaggedToken(word, tag, w_span))
        return ret



        '''
        for word, tags in literals:
            if word not in apostrophised:
                self.literal_dict[word] = tags

        for i, t in enumerate(tagged):
            span = token_spans[i]
            # don't change the pronoun as that would change the grammatical person
            if span in irregulars or t[1] == "PRP" or t[1] == "PRP$":
                if t[1] in apostrophised:
                    new_tag = apostrophised[t[1]]
                else:
                    new_tag = "LITERAL"
            else:
                new_tag = t[1]
            tagged_token = TaggedToken(t[0], new_tag, span)
            tagged[i] = tagged_token

        return tagged
        '''


    def populate_buckets(self, text):
        sent_detector = data.load('tokenizers/punkt/english.pickle')
        sents = sent_detector.tokenize(text.strip())
        pool = multiprocessing.Pool()

        #tagged_sents: List[List[TaggedToken]]
        tagged_sents = pool.map(Corpifier.get_usable_tokens, sents)
        tagged_sents = list(map(Corpifier.get_usable_tokens, sents))

        for tagged_tokens in tagged_sents:
            for token in tagged_tokens:
                if token.tag == 'LITERAL':
                    self.literal_bucket.add_word(token.word)
                    continue

                if token.tag not in self.buckets:
                    bkt = WordBucket(token.tag)
                    bkt.add_word(token.word)
                    self.buckets[token.tag] = bkt
                else:
                    self.buckets[token.tag].add_word(token.word)

    # this makes sure that each word has a unique tag, so that a
    # word appearing in one bucket doesn't appear in another one
    def singularise_words(self):
        buckets_ordered = list(self.buckets.values())
        buckets_ordered.sort(key=lambda b: b.count_unique_word(), reverse=True)
        for b in buckets_ordered:
            b.transfer_common(self.literal_bucket)

        for i, bkt in enumerate(buckets_ordered):
            for j in range(len(buckets_ordered)-1, i):
                bkt.transfer_common(buckets_ordered[j])

    @staticmethod
    def span_word(sent: str) -> List[Span]:
        # No --
        return [m.span() for m in re.finditer(r"([\w'—-])+|(\.{3})|[?,!:();.\"/]", sent)]

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
    corp.populate_buckets(text)
    corp.singularise_words()
    tags_count = [(item[0], item[1].count_unique_word()) for item in corp.buckets.items()]
    print(sorted(tags_count, key=itemgetter(1), reverse=True))
    #print(*list(corp.literal_set), sep='\n')

    '''
    sent_detector = data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(text.strip())

    pool = multiprocessing.Pool()

    tagged_sents = pool.map(Corpifier.get_usable_tokens, sents)

    print(Corpifier.count_usable_tags(tagged_sents))
    '''
