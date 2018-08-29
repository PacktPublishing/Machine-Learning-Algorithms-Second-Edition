from __future__ import print_function

from nltk import word_tokenize, pos_tag, ne_chunk, tree2conlltags


if __name__ == '__main__':
    sentence_1 = 'My friend John lives in Paris'

    # Perform a POS tagging
    tokens_1 = word_tokenize(sentence_1)
    tags_1 = pos_tag(tokens_1)

    print(sentence_1)
    print(tags_1)

    # Peform a POS and NER tagging
    sentence_2 = 'Search a hotel in Cambridge near the MIT'

    tokens_2 = word_tokenize(sentence_2)
    tags_2 = pos_tag(tokens_2)

    print('\n')
    print(sentence_2)
    print(tree2conlltags(ne_chunk(tags_2)))

