import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
    
    def forward(self, x):
        _, h_n = self.rnn(x)
        return h_n.squeeze(0)


class DocumentTower(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(DocumentTower, self).__init__()
        self.doc_encoder = RNNEncoder(embedding_dim, hidden_dim)

    def forward(self, relevant_doc_emb, irrelevant_doc_emb):
        relevant_doc_encoding = self.doc_encoder(relevant_doc_emb)
        irrelevant_doc_encoding = self.doc_encoder(irrelevant_doc_emb)
        return relevant_doc_encoding, irrelevant_doc_encoding
    
    def encode_single_doc(self, doc_emb):
        # Assuming the same encoder can be used for individual documents
        doc_encoding = self.doc_encoder(doc_emb)
        return doc_encoding

class QueryTower(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(QueryTower, self).__init__()
        self.query_encoder = RNNEncoder(embedding_dim, hidden_dim)
    
    def forward(self, query_emb):
        query_encoding = self.query_encoder(query_emb)
        return query_encoding
