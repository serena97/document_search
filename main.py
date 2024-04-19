from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
from model.twotowermodel import DocumentTower, QueryTower
from data.dataset import QueryDocumentDataset
from data.marco import get_training_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.loss_function import triplet_loss_function
import torch

model_path = "./artifacts/word2vec-300.bin"
model = Word2Vec.load(model_path)

def custom_collate(batch):
    # Unzip the batch into lists of queries, relevant_docs, and irrelevant_docs
    query_emb_list, relevant_doc_emb_list, irrelevant_doc_emb_list = zip(*batch)
    
    # Pad sequences for relevant and irrelevant documents
    # This assumes each document is already a tensor of embeddings; adjust if the structure is different
    padded_relevant = pad_sequence([pad_sequence(docs, batch_first=True) for docs in relevant_doc_emb_list], batch_first=True)
    padded_irrelevant = pad_sequence([pad_sequence(docs, batch_first=True) for docs in irrelevant_doc_emb_list], batch_first=True)
    
    # Pad the sequences of query embeddings
    query_emb_tensor = pad_sequence(query_emb_list, batch_first=True)

    return query_emb_tensor, padded_relevant, padded_irrelevant


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)
# Initialize model and optimizer
document_model = DocumentTower(embedding_dim=300, hidden_dim=128).to(device)
query_model = QueryTower(embedding_dim=300, hidden_dim=128).to(device)
optimizer = Adam(list(document_model.parameters()) + list(query_model.parameters()), lr=0.001)
dataset_instance = QueryDocumentDataset(data=get_training_dataset(), embedding_model=model)
dataloader = DataLoader(dataset_instance, batch_size=32, shuffle=False, collate_fn=custom_collate)  # You can adjust batch_size as needed
counter = 0

print('Beginning training')
for query_emb, relevant_doc_emb, irrelevant_doc_emb in dataloader:
    # Convert the tokens to embeddings
    # Forward pass to get encodings for two tower

    query_emb = query_emb.to(device)
    relevant_doc_emb = relevant_doc_emb.to(device)
    irrelevant_doc_emb = irrelevant_doc_emb.to(device)
    
    relevant_doc_encoding, irrelevant_doc_encoding = document_model(relevant_doc_emb, irrelevant_doc_emb)
    query_encoding = query_model(query_emb)

    # Compute triplet loss
    loss = triplet_loss_function(query_encoding, relevant_doc_encoding, irrelevant_doc_encoding, margin=0.3)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    counter += 1

    if counter % 10000 == 0:
        print(f"Triplet loss: {loss.item()}")
