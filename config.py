class Config:
    def __init__(self):
        self.model_path = './model/model.h5'

        self.train_data_path = './data/UIT-ViIC/uitviic_captions_train2017.json'

        self.test_data_path = './data/UIT-ViIC/uitviic_captions_test2017.json'

        self.val_data_path = './data/UIT-ViIC/uitviic_captions_val2017.json'

        self.images_path = './data/images/'

        self.output_path = './output/'

        self.vocab_path = self.output_path + 'vocab.pkl'

        self.image_embedding_path = self.output_path + 'resnet152_imgEmbedding.pkl'

        self.word2vec_path = './word2vec model/gensim_word2vec.model'

        self.epochs = 2

        self.imgs_per_batch = 8

        self.model_path = './model/model_0_68_loss.h5'

        self.train_history_path = './model/history.csv'