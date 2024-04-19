from torch import nn
import torch

def distance_function(x, y):
    return 1 - nn.functional.cosine_similarity(x, y)

def triplet_loss_function(query, relevant_document, irrelevant_document, margin):
    relevant_distance = distance_function(query, relevant_document) # should be small
    irrelevant_distance = distance_function(query, irrelevant_document) # should be big
    triplet_loss = torch.clamp(relevant_distance - irrelevant_distance + margin, min=0)
    triplet_loss = torch.mean(triplet_loss) # taking the mean for the batch
    return triplet_loss