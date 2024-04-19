from datasets import load_dataset
from itertools import chain
import random

dataset = load_dataset("ms_marco",'v1.1')

def get_training_dataset():
    return dataset['train']

def get_sentences():
    queries = dataset['train']['query']
    passages = [text for passage in dataset['train']['passages'] for text in passage['passage_text']]
    return queries + passages

def get_irrelevant_doc(index):
    numbers = list(range(len(dataset['train'])))
    numbers.remove(index)
    indexes = random.sample(numbers, 10)
    irrelevant = [dataset['train'][i]['passages']['passage_text'][0] for i in indexes]
    return irrelevant

def build_query_doc_pairs(index):
    query = dataset['train'][index]['query']
    relevant = dataset['train'][index]['passages']['passage_text']
    irrelevant = get_irrelevant_doc(index)
    return query, relevant, irrelevant

def get_all_documents():
    train = [passage['passage_text'] for passage in dataset['train']['passages']]
    # validation = [passage['passage_text'] for passage in dataset['train']['passages']]
    # test = [passage['passage_text'] for passage in dataset['train']['passages']]
    return train