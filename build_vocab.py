from collections import Counter
from pyvi import ViTokenizer
import string, pickle

from preprocessing_data import load_data
from config import Config

def vocabulary(json, threshold=10):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    max_len = 0
    for i in range(len(json)):
        caption = json[i]['caption']
        print(caption)
        max_len = max(max_len, len(caption.split()))

        tokens = ViTokenizer.tokenize(caption.lower()).split()
        print(tokens)
        print('=====================================')
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    # Remove punctuation from dictionary
    for punc in string.punctuation:
        if punc in words:
            words.remove(punc)
            
    # Add start sequence, end sequence
    words.append('<start>')
    words.append('<end>')

    # Create a vocab wrapper and add some special tokens.
    word2id = {}
    id2word = {}

    # Add the words to the vocabulary.
    for idx, word in enumerate(words):
        if word not in word2id:
            word2id[word] = idx
            id2word[idx] = word
            
    return word2id, id2word, max_len

def build():
    C = Config()
    train_data, _ = load_data()
    annotations = train_data['annotations']
    word2id, id2word, max_len = vocabulary(annotations, 5)

    vocab = {}
    vocab['word2id'] = word2id
    vocab['id2word'] = id2word
    vocab['max_len'] = max_len

    vocab_path = C.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print('Size of word2id: ', len(word2id))
    print('Max length of sequence: ', max_len)

if __name__ == '__main__':
    build()