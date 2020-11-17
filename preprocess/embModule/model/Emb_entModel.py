import torch
import torch.nn as nn
import torch.nn.functional as F

class Emb_entModel(nn.Module):
	def __init__(self, weight):
		super(Emb_entModel, self).__init__()
		self.ent_tot = weight.shape[0]
		self.dim =  weight.shape[1]
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		# load_weight
		self.ent_embeddings.weight.data = weight
		self.ent_embeddings.requires_grad_(False)

	def forward(self, data):
		emb = self.ent_embeddings(data)

		return emb


	def predict(self, data):
		score = self.forward(data)

		return score