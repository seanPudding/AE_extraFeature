import os
import json
import torch
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf


def load_parameters(path):
    f = open(path, "r")
    parameters = json.loads(f.read())
    f.close()
    for key in parameters:
        print(key)
        parameters[key] = torch.Tensor(parameters[key])
    # self.load_state_dict(parameters, strict=False)
    # self.eval()

model_dir = "preprocess/embModule/savedparameters"
checkpoint_path = os.path.join(model_dir, "transe_FB15K237")
print(checkpoint_path)
load_parameters("/home/ubuntu/PycharmProjects/AE_extraFeature/preprocess/embModule/savedparameters/transe")
# load_parameters(checkpoint_path)