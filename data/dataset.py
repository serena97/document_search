import torch
from torch.utils.data import Dataset
import random
from gensim.utils import simple_preprocess
import numpy as np
# Get document for query and doc tokens
class QueryDocumentDataset(Dataset):
    def __init__(self, data, embedding_model):
        self.data = data
        self.model = embedding_model
        self.indices = list(range(len(data)))

    def __len__(self):
        return len(self.data)

    def get_irrelevant_doc(self, index):
        numbers = self.indices.copy()
        numbers.remove(index)
        indexes = random.sample(numbers, 10)
        irrelevant = [self.data[i]['passages']['passage_text'][0] for i in indexes]
        return irrelevant

    def get_query_embedding(self, query):
        tokenized_query = simple_preprocess(query)
        embeddings = []
        for word in tokenized_query:
            if word in self.model.wv:
                embeddings.append(self.model.wv[word])
            else:
                raise ValueError(f"query {word} not in vocabulary")
        embeddings_array = np.array(embeddings, dtype=np.float32)
        return torch.tensor(embeddings_array)

    def get_doc_embedding(self, docs):
        embeddings = []
        for doc in docs:
            tokenized_doc = simple_preprocess(doc)
            for word in tokenized_doc:
                if word in self.model.wv:
                    embeddings.append(self.model.wv[word])
                else:
                    raise ValueError(f"doc {word} not in vocabulary")
        embeddings_array = np.array(embeddings, dtype=np.float32)
        return torch.tensor(embeddings_array)
    
    
    def __getitem__(self, index):
        query = self.data[index]['query']
        relevant = self.data[index]['passages']['passage_text']
        irrelevant = self.get_irrelevant_doc(index)
        
        query_embedding = self.get_query_embedding(query)
        relevant_embedding = self.get_doc_embedding(relevant)
        irrelevant_embedding = self.get_doc_embedding(irrelevant)

        return query_embedding, relevant_embedding, irrelevant_embedding
