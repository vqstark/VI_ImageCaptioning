import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications import ResNet152
from keras.utils import to_categorical, pad_sequences
import pickle, keras, copy, cv2

from config import Config

# Load configurations
C = Config()

# Load vocabulary
with open(C.vocab_path, 'rb') as f:
    vocab = pickle.load(f)
word2id = vocab['word2id']
id2word = vocab['id2word']
max_len = vocab['max_len']

# Load model
r_model = ResNet152(weights='imagenet')
resnet_model = Model(r_model.input, r_model.layers[-2].output)
model = keras.models.load_model(C.model_path)

def greedySearch(photo):
    in_text = '<start>'
    for i in range(max_len):
        sequence = [word2id[w] for w in in_text.split() if w in word2id]
        sequence = pad_sequences([sequence], maxlen=max_len)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = id2word[yhat]
        in_text += ' ' + word
        if word == '<end>':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beamSearch(photo, kmax=2, threshold=0.2, max_length=34):
    sentences = ['<start>']
    proba = [[1]]
    sen_result = []
    prob_result = []
    while len(sentences)>0:
        sentence = sentences[0]
        sentences = sentences[1:]
        cur_proba = proba[0]
        proba = proba[1:]
        in_sentence = copy.copy(sentence)
        
        # Split word in sentence
        sequence = sentence.split(" ")
        if sequence[-1] != '<end>' and len(sequence)<=max_length:
            sequence = [word2id[w] for w in sequence if w in word2id]
            sequence = pad_sequences([sequence], maxlen=max_len)
            yhat = model.predict([photo,sequence], verbose=0)
            
            next_words = [(id2word[idx],yhat[0][idx]) for idx in np.argsort(yhat[0])[::-1][:kmax] if yhat[0][idx]>=threshold]
            if len(next_words)==0:
                next_words = [(id2word[idx],yhat[0][idx]) for idx in np.argsort(yhat[0])[::-1][:1]]
#             print(next_words)
            
            for i in range(len(next_words)):
                next_sentence = in_sentence +' '+ next_words[i][0]
                sentences.append(next_sentence)
                next_proba = copy.copy(cur_proba)
                next_proba.append(next_words[i][1])
                proba.append(next_proba)

        else:
            sen_result.append(in_sentence)
            prob_result.append(cur_proba)
    # Calculate conditional proba
    probaAll = [np.prod(p) for p in prob_result]
    idx_probaAll = [idx for idx in np.argsort(probaAll)[::-1][:3]]
            
    # Remove <start> and <end> seg and return 3 sentences
    sen_result = np.array(sen_result)[idx_probaAll]
    return [' '.join(s.split()[1:-1]) for s in sen_result], np.array(probaAll)[idx_probaAll]

def export_result(img_path, search='beam'):
    if search == 'greedy':
        pass
    targeted_img = cv2.imread(img_path)
    plt.imshow(targeted_img)
    targeted_img = cv2.resize(targeted_img, (224, 224))
    x = np.expand_dims(targeted_img, axis=0)
    pred_x = resnet_model.predict(x)

    if search == 'greedy':
        print(greedySearch(pred_x))
    else:
        sen_result, prob_result = beamSearch(pred_x, kmax=5, threshold=0.3, max_length = 34)
        for idx,s in enumerate(sen_result):
            print(f"{idx+1}. {s}")
        for idx,s in enumerate(prob_result):
            print(f"{idx+1}. {s}")

export_result('./data/test_images/000000476045.jpg')