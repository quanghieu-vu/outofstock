import pandas as pd
import os, sys
import gc

from ml_metrics import mapk

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

def get_prob(item):
    return item[1]

def get_sorted_items(list_items):
    sorted_items = sorted(list_items, key=get_prob, reverse=True)
    res1 = [int(item[0]) for item in sorted_items]
    res2 = " ".join(["%s" % item[0] for item in sorted_items])
    return res1, res2

def predict(is_validation=True):
    result = []
    if is_validation==True:
        fin = open(os.path.join(HOME_DIR,"input/clicks_train_event_ad_mean.csv"), "r")     
    else:
        fin = open(os.path.join(HOME_DIR,"input/clicks_test_event_ad_mean.csv"), "r")     
    
    #skip the title
    line = fin.readline().strip()
    
    #get the first row
    list_items = []
    line = fin.readline().strip()
    params = line.split(",")
    current_display_id = params[0]
    ad_id = params[1]    
        
    if is_validation==True:
        prob = float(params[5])
    else:
        prob = float(params[2])
    list_items.append([ad_id,prob])
    
    #prepare output
    fout = open(os.path.join(HOME_DIR,"output/sub_v2.csv"), "w")
    fout.write("display_id,ad_id\n")
    
    while 1:
        line = fin.readline().strip()
        if line == '':
            break
        
        params = line.split(",")
        display_id = params[0]
        ad_id = params[1]
    
        if is_validation==True:
            prob = float(params[5])
        else:
            prob = float(params[2])    
        
        if display_id != current_display_id:     
            fout.write(current_display_id + ",")   
            res1, res2 = get_sorted_items(list_items) 
                
            #output current_display_id and reset values
            result.append(res1)
            fout.write(res2 + "\n")
            list_items = []
            current_display_id = display_id
            
        list_items.append([ad_id,prob])

    #output the last one
    fout.write(current_display_id + ",")   
    res1, res2 = get_sorted_items(list_items)    
    result.append(res1)
    fout.write(res2 + "\n")
    fout.close()               
    fin.close()      
    
    print(result[:5])
    if is_validation==True:
        train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train.csv")) 
        y = train[train.clicked==1].ad_id.values
        y = [[_] for _ in y]                                                
    
        del train
        gc.collect()        
        
        print("start evaluation")
        print(mapk(y, result, k=12))

#long running time
def validate():        
    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train.csv")) 
    y = train[train.clicked==1].ad_id.values
    y = [[_] for _ in y] 
    
    predict = pd.read_csv(os.path.join(HOME_DIR,"output/sub_v2.csv"))
    p = [[row["ad_id"].split()] for index, row in predict.iterrows()] 
    print(mapk(y, p, k=12))                                                                                                                                                                                                                                     

#===============================================================================
predict()
#validate()												
