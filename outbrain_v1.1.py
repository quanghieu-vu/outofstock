import pandas as pd
import numpy as np 
import os, gc, sys

from ml_metrics import mapk

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
reg = 10 # trying anokas idea of regularization

'''
return the second item of the list as the score
'''
def get_score(item):
    ad_id = int(item[1])
    
    #version 1.1
    sim_score = float(item[2])
    if ad_id not in cnt:
        return sim_score
    return 10 * cnt[ad_id]/(float(cntall[ad_id]) + reg) + sim_score
    
    #version 1
    '''
    if ad_id not in cnt:
        return 0
    return cnt[ad_id]/(float(cntall[ad_id]) + reg)
    '''
'''
list_items is a list of [ad_id, similarity_score]
'''
def get_sorted_items(list_items):
    sorted_items = sorted(list_items, key=get_score, reverse=True)
    res1 = [int(item[1]) for item in sorted_items]
    res2 = " ".join(["%s" % item[1] for item in sorted_items])
    return res1, res2

'''
Input file: display_id, ad_id, similarity_score
Output file: display_id, sorted list of ad_id
'''
def predict(is_validation=True):
    result = []
    if is_validation == True:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_train_content_similarity.csv"), "r")     
    else:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_content_similarity.csv"), "r")     
    
    #skip the title
    line = input_file.readline().strip()    
    
    #get the first row
    list_items = []
    line = input_file.readline().strip()
    params = line.split(",")
    current_display_id = params[0]
    ad_id = params[1]
    
    if is_validation==True:
        prob = float(params[3])
    else:
        prob = float(params[2])
    list_items.append([ad_id,prob])
    
    #initialize variables
    list_items = []
    list_items.append([current_display_id,ad_id,prob])
    current_display_id = params[0]
    
    #prepare output
    output_file = open(os.path.join(HOME_DIR,"output/submission_content_based.csv"), "w")
    output_file.write("display_id,ad_id\n")    
   
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
        
        #extract information of the current row
        params = line.split(",")
        display_id = params[0]
        ad_id = params[1]
        
        if is_validation==True:
            prob = float(params[3])
        else:
            prob = float(params[2])   
        
        if display_id != current_display_id:     
            output_file.write(current_display_id + ",")   
            res1, res2 = get_sorted_items(list_items) 
                
            #output current_display_id and reset values
            if is_validation==True:
                if int(current_display_id) in ids:
                    result.append(res1)
   
            output_file.write(res2 + "\n")
            list_items = []
            current_display_id = display_id
        
        #add information to list    
        list_items.append([current_display_id,ad_id,prob])

    #output the last one
    output_file.write(current_display_id + ",")   
    res1, res2 = get_sorted_items(list_items)   
   
    if is_validation==True: 
        if int(current_display_id) in ids: 
            result.append(res1)
    output_file.write(res2 + "\n")

    #closing files
    output_file.close()             
    input_file.close()             

    if is_validation == True:
        print("start evaluation")
        print(mapk(y, result, k=12))      

# ==============================================================================
eval = False

train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train_content_similarity.csv"))
if eval:
    ids = train.display_id.unique()
    
    np.random.seed(13)
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
	
predict(eval)