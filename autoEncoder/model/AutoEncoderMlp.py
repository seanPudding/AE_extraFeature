import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModule import BaseModule
class AutoEncoderMlp(BaseModule):
	def __init__(self, orgin_dim, aim_dim):
		super(AutoEncoderMlp, self).__init__()
		
		self.orgin_dim = orgin_dim
		self.aim_dim = aim_dim

		#autocoder
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(self.orgin_dim, 100),
			torch.nn.ReLU(True),
			torch.nn.Linear(100, 80),
			torch.nn.ReLU(True),
			torch.nn.Linear(80, self.aim_dim),
			torch.nn.ReLU(True),

			# torch.nn.Linear(self.orgin_dim, self.aim_dim),
			# torch.nn.ReLU(True),

		)

		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(self.aim_dim, 80),
			torch.nn.ReLU(True),
			torch.nn.Linear(80, 100),
			torch.nn.ReLU(True),
			torch.nn.Linear(100, self.orgin_dim),
			torch.nn.Sigmoid(),

			# torch.nn.Linear(self.aim_dim, self.orgin_dim),
			# torch.nn.Sigmoid(),
		)


	def forward(self, data):
		x = data
		fea = self.encoder(x)
		x_ = self.decoder(fea)

		return fea, x_


	def predict(self, data):
		fea, x_ = format(data)
		return fea


# //check the value of emb_ent, then correct the activate function