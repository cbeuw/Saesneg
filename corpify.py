from nltk import data
from nltk import word_tokenize
import nltk
from operator import itemgetter

if __name__ == '__main__':
    text = open('1984', encoding='utf-8').read()
    sent_detector = data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(text.strip())
    tags_count = {}
    for s in sents:
        tags = nltk.pos_tag(word_tokenize(s))
        for t in tags:
            if t[1] not in tags_count:
                tags_count[t[1]] = {t[0]}
            else:
                tags_count[t[1]].add(t[0])

    for k, v in tags_count.items():
        tags_count[k] = len(v)
    print(sorted(tags_count.items(), key=itemgetter(1), reverse=True))




