from preprocess.embModule.model import Emb_entModel
from preprocess.embModule.selectWeight import getWeight

preModel = 'transe_FB15K237'
ent_weight = getWeight(preModel,'ent')
emb_entModel =Emb_entModel(ent_weight)

triple = open("train2id.txt", "r")
valid = open("valid2id.txt", "r")
test = open("test2id.txt", "r")

tot = (int)(triple.readline())
for i in range(tot):
	content = triple.readline()
	h,t,r = content.strip().split()


tot = (int)(valid.readline())
for i in range(tot):
	content = valid.readline()
	h,t,r = content.strip().split()


tot = (int)(test.readline())
for i in range(tot):
	content = test.readline()
	h,t,r = content.strip().split()

	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

test.close()
valid.close()
triple.close()

