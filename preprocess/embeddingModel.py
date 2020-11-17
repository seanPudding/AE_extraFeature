from preprocess.embModule.model import Emb_entModel
from preprocess.embModule.selectWeight import getWeight
import torch

preModel = 'transe_FB15K237'
ent_weight = getWeight(preModel,'ent')

emb_entModel =Emb_entModel(ent_weight)

res = emb_entModel.forward(torch.Tensor([[1]]))

pass
