import pandas as pd
import numpy as np 
import os, gc, sys

from ml_metrics import mapk

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
reg = 10 # trying anokas idea of regularization
eval = True

def get_prob(k):
    if k not in cnt:
        return 0
    return cnt[k]/(float(cntall[k]) + reg)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse=True)
    return " ".join(map(str,ad_ids))

# ==============================================================================
train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train.csv"))
if eval:
	ids = train.display_id.unique()
	ids = np.random.choice(ids, size=len(ids)//10, replace=False)

	valid = train[train.display_id.isin(ids)]
	train = train[~train.display_id.isin(ids)]
	
	print (valid.shape, train.shape)

cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()

del train
gc.collect()
   
if eval:
	y = valid[valid.clicked==1].ad_id.values
	y = [[_] for _ in y]
	p = valid.groupby('display_id').ad_id.apply(list)
	print(p[:10])
	p = [sorted(x, key=get_prob, reverse=True) for x in p]
	print(p[:10])
	print (mapk(y, p, k=12))
else:
	subm = pd.read_csv(os.path.join(HOME_DIR,"input/sample_submission.csv")) 
	subm['ad_id'] = subm.ad_id.apply(lambda x: srt(x))
	subm.to_csv(os.path.join(HOME_DIR,"output/subm_reg_1.csv"), index=False)