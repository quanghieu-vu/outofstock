import pandas as pd
import os, sys
import gc

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

#add extra information from event and ad to the training and testing data sets
def merge_data():
    event_info = pd.read_csv(os.path.join(HOME_DIR,"input/events.csv"), usecols=["display_id", "document_id"])
    event_info.rename(columns={'document_id': 'event_doc_id'}, inplace=True)

    ad_info = pd.read_csv(os.path.join(HOME_DIR,"input/promoted_content.csv"), usecols=["ad_id", "document_id"])
    ad_info.rename(columns={'document_id': 'ad_doc_id'}, inplace=True)

    #update for 87141731 rows of train
    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train.csv"))
    print(train.shape)
    print(train.head(5))

    train = pd.merge(train, event_info, on="display_id", how="left")
    print(train.head(5))

    train = pd.merge(train, ad_info, on="ad_id", how="left")
    print(train.head(5))

    train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_update.csv"), index=False)
    
    del train
    gc.collect()

    #update for 32225162 of test
    test = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_test.csv"))
    print(test.shape)
    print(test.head(5))

    test = pd.merge(test, event_info, on="display_id", how="left")
    print(test.head(5))

    test = pd.merge(test, ad_info, on="ad_id", how="left")
    print(test.head(5))

    test.to_csv(os.path.join(HOME_DIR,"input/clicks_test_update.csv"), index=False)

    del test
    del event_info
    del ad_info
    gc.collect()

#preprocess
def preprocess():
    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train_update.csv"), usecols=["event_doc_id","ad_doc_id","clicked"])    
    print(train.head(5))
    
    train_dict = pd.DataFrame({'prob':train.groupby(["event_doc_id","ad_doc_id"])['clicked'].mean()}).reset_index()      
    print(train_dict.shape)   
    print(train_dict.head(5))

    del train
    gc.collect()    

    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train_update.csv"))    
    train = pd.merge(train, train_dict, on=["event_doc_id","ad_doc_id"], how="left")   
    train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_preprocessing.csv"), index=False)
    
    del train
    gc.collect()
        
    test = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_test_update.csv")) 
    test = pd.merge(test, train_dict, on=["event_doc_id","ad_doc_id"], how="left")   
    print("Number of NA prob %d" %len(test[(test["prob"].isnull())]))
    test.ix[test["prob"].isnull(),"prob"] = 0 
    test.to_csv(os.path.join(HOME_DIR,"input/clicks_test_preprocessing.csv"), index=False)
    
    del test    
    del train_dict
    gc.collect()

def get_prob(item):
    return item[1]

def get_sorted_items(lis_items):
    sorted_items = sorted(lis_items, key=get_prob, reverse=True)
    return " ".join(["%s" % item[0] for item in sorted_items])

def predict():
    fin = open(os.path.join(HOME_DIR,"input/clicks_test_preprocessing.csv"), "r")     
    
    #skip the title
    line = fin.readline().strip()
    
    #get the first row
    list_items = []
    line = fin.readline().strip()
    params = line.split(",")
    current_display_id = params[0]
    ad_id = params[1]
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
        prob = float(params[2])    
        
        if display_id != current_display_id:     
            fout.write(current_display_id + ",")   
            sorted_items = get_sorted_items(list_items) 
                
            #output current_display_id and reset values
            fout.write(sorted_items + "\n")
            list_items = []
            current_display_id = display_id
            
        list_items.append([ad_id,prob])

    #output the last one
    fout.write(current_display_id + ",")   
    sorted_items = get_sorted_items(list_items)    
    fout.write(sorted_items + "\n")
    fout.close()                      
                                                
#===============================================================================
#merge_data()
#preprocess()
predict()												
'''
TODO:
Join clicks_train|test.csv with events.csv and promoted_content.csv to obtain extra information
uuid, display_doc_id, ad_doc_id, clicked
a. using xgboost to predict the probability and then sort the result to output
b. using xgboost only for new users, existing users with matrix factorization

'''