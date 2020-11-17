import os
import json
import torch

para_dir = os.path.join(os.getcwd(),"/autoEncoder/data")

def getData(name,relOrent):
    f = open(name, "r")
    parameters = json.loads(f.read())
    f.close()
    if relOrent != 'ent' and relOrent != 'rel':
        raise Exception("relOrent should be 'rel' or 'ent'")
    for key in parameters:
        if relOrent == 'ent':
            if 'ent_embeddings.weight' == key:
                return torch.Tensor(parameters[key])
        if relOrent == 'rel':
            if 'rel_embeddings.weight' == key:
                return torch.Tensor(parameters[key])





# res = getWeight('transe_FB15K237','rel')
#
# pass







# def load_parameters(path):
#     f = open(path, "r")
#     parameters = json.loads(f.read())
#     f.close()
#     for key in parameters:
#         print(key)
#         parameters[key] = torch.Tensor(parameters[key])
#     # self.load_state_dict(parameters, strict=False)
#     # self.eval()