import pickle, os, urllib, cv2
import keras
from keras.models import Model
from keras.applications import ResNet152
from keras.models import Sequential
from keras.layers import Add, LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization, Bidirectional
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras import Input, layers
from keras.utils import to_categorical, pad_sequences
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from config import Config
from preprocessing_data import load_data, cleaning_data

def cnn_model():
    # Pre-trained model
    resnet_model = ResNet152(weights='imagenet')
    r_model = Model(resnet_model.input, resnet_model.layers[-2].output)
    return r_model


def img_emb(cnn_model, img):
    # Image embedding
    feature_vec = cnn_model.predict(img)
    feature_vec = np.reshape(feature_vec, feature_vec.shape[1])
    return feature_vec


def imgs_emb(path, image_embedding_path, cnn_model, train_data):
    img_embedding = {}
    for image in train_data['images']:
        url = image['coco_url']
        
        # Read image
        file_name = url.split('/')[-1]
        img_path = path+file_name
        
        #Read from url
        urllib.request.urlretrieve(url, img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        x = np.expand_dims(img, axis=0)
        
        embeded_img = img_emb(cnn_model, x)
        img_embedding[file_name] = embeded_img

    # Save image embedding
    with open(image_embedding_path, "wb") as embeded_pickle:
        pickle.dump(img_embedding, embeded_pickle)
    print('Save imgs embedding to: ', image_embedding_path)

    return img_embedding


def build_model(embedding_matrix, max_len, vocab_size, embedding_dim):
    # Build model
    # Bidirectional for LSTM model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)

    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Bidirectional(LSTM(256))(se1)

    decoder1 = Add()([fe2, se2])
    decoder2 = Dense(512, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.summary()
    # Set word embedding for Embedding layer
    model.layers[3].set_weights([embedding_matrix])
    model.layers[3].trainable = False

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

    return model

# Data generator
def data_generator(data, images_features, word2id, vocab_size, max_length, images_per_batch):
    """
    Args:
        data:
        images_features:
        word2id:
        max_length:
        images_per_batch:
    Return:
        Generate data for training process
    """
    X1 = []
    X2 = []
    y = []
    counter = 0
    for img in data:
        counter+=1
        file_name = img['url'].split('/')[-1]
        # Get image features
        fea_img = images_features[file_name]
        # Get 5 annotations of images
        desc_list = img['annotations']
        
        for desc in desc_list:
            # encode the sequence
            seq = [word2id[word] for word in desc.split(' ') if word in word2id]
            # split one sequence into multiple X, y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(fea_img)
                X2.append(in_seq)
                y.append(out_seq)
        if counter == images_per_batch:
            yield [[np.array(X1), np.array(X2)], np.array(y)]
            X1, X2, y = list(), list(), list()
            counter=0


def build():
    # Load configurations
    C = Config()

    # Load data
    train_data, val_data = load_data()

    # Load vocabulary
    with open(C.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    word2id = vocab['word2id']
    id2word = vocab['id2word']
    max_len = vocab['max_len']

    # Load resnet model
    r_model = cnn_model()

    # Load feature vectors
    if os.path.exists(C.image_embedding_path):
        img_features = pickle.load(open(C.image_embedding_path, "rb"))
    else:
        img_features = imgs_emb(C.images_path, C.image_embedding_path, r_model, train_data)
    

    # Load Word2Vec model (200 dim)
    w2v_model = Word2Vec.load(C.word2vec_path)

    # Embedding words
    vocab_size = len(word2id)+1
    embedding_dim = 200
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word2id.items():
        embedding_vector = None
        try:
            embedding_vector = w2v_model.wv[word]
        except Exception:
            embedding_vector = None
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    
    # Get model
    model = build_model(embedding_matrix, max_len, vocab_size, embedding_dim)

    # Get cleaning data to train
    data = cleaning_data(train_data)

    # Train
    epochs = C.epochs
    imgs_per_batch = C.imgs_per_batch
    step = len(data)//imgs_per_batch
    model_path = C.model_path
    train_history_path = C.train_history_path

    def train(model, train_history, start_epoch, epochs, imgs_per_batch, step):
        for epoch in range(start_epoch, epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            generator = data_generator(data, img_features, word2id, vocab_size, max_len, images_per_batch = imgs_per_batch)
            history = model.fit_generator(generator, epochs = 1, steps_per_epoch=step)

            # Save model checkpoint
            model.save(model_path, include_optimizer=True)
            cur_record = [history.history['loss'][0], history.history['accuracy'][0]]
            train_history = pd.concat([train_history, pd.DataFrame([cur_record], columns=['loss', 'accuracy'])])
            train_history.to_csv(train_history_path)


    
    if os.path.exists(train_history_path):
        print('==>Continue training..')
        # Load model and train_history
        model = keras.models.load_model(model_path)
        train_history = pd.read_csv(train_history_path)
        start_epoch = len(train_history)

        train(model, train_history, start_epoch, epochs, imgs_per_batch, step)
    else:
        print('==>Start training..')
        start_epoch = 0
        train_history = pd.DataFrame(columns=['loss', 'accuracy'])
        train(model, train_history, start_epoch, epochs, imgs_per_batch, step)

if __name__ == '__main__':
    build()