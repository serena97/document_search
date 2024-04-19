from gensim.models import Word2Vec

def get_model(sentences):
    model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=1, sg=1, workers=5)  # sg=1 indicates Skip-Gram
    return model

def train_model(model, sentences, epochs=10):
    model.build_vocab(sentences, update=True, progress_per=10000)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    return model