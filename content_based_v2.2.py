import pandas as pd
import numpy as np
import os, sys
import gc
import json

from ml_metrics import mapk

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

'''
merge clicks_train.csv and clicks_test.csv with events.csv and promoted_content.csv
output files are:
    clicks_train_doc.csv(display_id, ad_id, clicked, event_doc_id, ad_doc_id)
    clicks_test_doc.csv(display_id, ad_id, event_doc_id, ad_doc_id)
'''
def merge_click_event_ad():
    event_info = pd.read_csv(os.path.join(HOME_DIR,"input/events.csv"), usecols=["display_id", "document_id"])
    event_info.rename(columns={'document_id': 'event_doc_id'}, inplace=True)

    ad_info = pd.read_csv(os.path.join(HOME_DIR,"input/promoted_content.csv"), usecols=["ad_id", "document_id"])
    ad_info.rename(columns={'document_id': 'ad_doc_id'}, inplace=True)

    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train.csv"))
    train = pd.merge(train, event_info, on="display_id", how="left")
    train = pd.merge(train, ad_info, on="ad_id", how="left")
    train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_doc.csv"), index=False)
    
    del train
    gc.collect()

    test = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_test.csv"))
    test = pd.merge(test, event_info, on="display_id", how="left")
    test = pd.merge(test, ad_info, on="ad_id", how="left")
    test.to_csv(os.path.join(HOME_DIR,"input/clicks_test_doc.csv"), index=False)

    del test
    del event_info
    del ad_info
    gc.collect()

'''
split training data into training and testing for validation
'''
def split_train_validate(file_name):
    train = pd.read_csv(os.path.join(HOME_DIR,file_name))
    ids = train.display_id.unique()
    
    np.random.seed(13)
    ids = np.random.choice(ids, size=len(ids)//10, replace=False)

    #valid needs to be set before train, otherwise, the valid will be empty after train is reset
    valid = train[train.display_id.isin(ids)]
    train = train[~train.display_id.isin(ids)]

    valid.to_csv(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), index=False)    
    train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_validation.csv"), index=False)

'''
build a dictionary for event_doc_id of clicked and unclicked ad_doc_id
input: name of input file which can be complete clicks_train_doc or a part of it for validation purpose
output: a dictionary  
'''
def build_dict_event_doc_clicked(file_name, is_validation):
    input_file = open(os.path.join(HOME_DIR,file_name), "r")
    input_file.readline()
    
    event_doc_dict = dict()
    total_row = 0                                    
    
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        total_row += 1
        if total_row % 1000000 == 0:
            print('Read {} lines...'.format(total_row))
                    
        params = line.split(",")
        clicked = int(params[2])
        event_doc_id = params[3]
        ad_doc_id = params[4]
                
        if event_doc_id in event_doc_dict:            
            #retrieve ad_doc_dict
            ad_doc_dict = event_doc_dict[event_doc_id]
            
            if ad_doc_id in ad_doc_dict:
                #increase total clicked of existing ad_doc_id by 1
                ad_doc_dict[ad_doc_id][1] += 1
            else:
                #set total occurences of ad_doc_id to 1
                ad_doc_dict[ad_doc_id] = [0, 1, 0.0]
 
            if clicked == 1:
                #increase total clicked of ad_doc_id by 1
                ad_doc_dict[ad_doc_id][0] += 1
        else:
            #create a new dict for the new key
            ad_doc_dict = dict()
            event_doc_dict[event_doc_id] = ad_doc_dict
            
            #set total occurences of ad_doc_id to 1
            ad_doc_dict[ad_doc_id] = [0, 1, 0.0]
            
            if clicked == 1:
                #increase total clicked of ad_doc_id by 1
                ad_doc_dict[ad_doc_id][0] += 1

    if is_validation == True:
        with open(os.path.join(HOME_DIR,"dict/event_doc_clicked_validation.dict"), "w") as f:
            json.dump(event_doc_dict, f) 
    else:                
        with open(os.path.join(HOME_DIR,"dict/event_doc_clicked.dict"), "w") as f:
            json.dump(event_doc_dict, f) 
    
    del event_doc_dict
    gc.collect()                                                                                                                                                                                                                                                            

'''
interaction score: #pos + alpha/#total + beta
final score = interaction score + similarity score
'''
def build_dict_event_doc_score(file_name, is_validation):    
    #count number of clicked ads and total number of occurences of ads
    train = pd.read_csv(os.path.join(HOME_DIR,file_name), usecols=["clicked","ad_doc_id"])
    beta_all = float(len(train)) / float(len(train[train.clicked==1]))
    print("beta_all %f" %(beta_all))
    
    cnt = train[train.clicked==1].ad_doc_id.value_counts()
    cntall = train.ad_doc_id.value_counts()
    
    del train
    gc.collect()
    
    if is_validation == True:
        with open(os.path.join(HOME_DIR,"dict/event_doc_ad_doc_clicked_validation.dict"), "r") as f:
            event_doc_dict = json.load(f) 
    else:
        with open(os.path.join(HOME_DIR,"dict/event_doc_ad_doc_clicked.dict"), "r") as f:
            event_doc_dict = json.load(f) 
    
    total_item = 0
    for event_doc_id in event_doc_dict:
        total_item += 1
        if total_item % 100000 == 0:
            print('Finished {} items...'.format(total_item))
            
        ad_doc_dict = event_doc_dict[event_doc_id]
        for ad_doc_id in ad_doc_dict:            
            ad_doc_id_value = int(ad_doc_id)
            if ad_doc_id_value in cnt:
                cnt_ad_id = cnt[ad_doc_id_value]
            else:
                cnt_ad_id = 0
            beta_ad = (cntall[ad_doc_id_value] + beta_all) / (cnt_ad_id + 1.0)
            ad_doc_dict[ad_doc_id][2] = (ad_doc_dict[ad_doc_id][0] + 1.0)/(ad_doc_dict[ad_doc_id][1] + beta_ad)            

    if is_validation == True:
        with open(os.path.join(HOME_DIR,"dict/event_doc_ad_doc_score_validation.dict"), "w") as f:
            json.dump(event_doc_dict, f)                                                                                                                            
    else:
        with open(os.path.join(HOME_DIR,"dict/event_doc_ad_doc_score.dict"), "w") as f:
            json.dump(event_doc_dict, f)                                                                                                                            

    del event_doc_dict
    gc.collect()
    
    return (1.0 / beta_all - 0.013)

'''
list1, list2 are lists of pairs(item, confidence)
'''                                
def get_similarity_score(list1, list2):
    score = 0.0   
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i][0] == list2[j][0]:
                score = score + float(list1[i][1]) * float(list2[j][1])
    return score

'''
Calculate score of each pair event_doc_id, ad_doc_id
Only consider probability
Not consier content_similarity
'''
def calculate_score(default_score, is_validation):
    #load probability score dictionary
    if is_validation == True:
        with open(os.path.join(HOME_DIR,"dict/event_doc_ad_doc_score_validation.dict"), "r") as f:
            event_doc_dict = json.load(f)    
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), "r")
        output_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation_score.csv"), "w")
    else:
        with open(os.path.join(HOME_DIR,"dict/event_doc_ad_doc_score.dict"), "r") as f:
            event_doc_dict = json.load(f)    
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_doc.csv"), "r")
        output_file = open(os.path.join(HOME_DIR,"input/clicks_test_score.csv"), "w")
    
    #load topics and categories dictionaries
    with open(os.path.join(HOME_DIR,"dict/doc_topics_conf.dict"), "r") as f:
        topics = json.load(f)    
    with open(os.path.join(HOME_DIR,"dict/doc_cats_conf.dict"), "r") as f:
        categories = json.load(f)      

    print("Dictionaries have been loaded")
                
    #update testing data set
    count = 0
    line = input_file.readline().strip()    
    output_file.write("display_id,ad_id,score\n")

    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        params = line.split(",")
        if is_validation == True:
            event_doc_id = params[3]
            ad_doc_id = params[4]
        else:
            event_doc_id = params[2]
            ad_doc_id = params[3]
            
        if event_doc_id in event_doc_dict:
            ad_doc_dict = event_doc_dict[event_doc_id]
            if ad_doc_id in ad_doc_dict:
                total_score = ad_doc_dict[ad_doc_id][2]
                count = count + 1
            else:
                total_score = default_score
        else:
            total_score = default_score
        
        #adding content-based score
        event_topic_ids = topics.get(event_doc_id)    
        event_cat_ids = categories.get(event_doc_id)
        ad_topic_ids = topics.get(ad_doc_id)
        ad_cat_ids = categories.get(ad_doc_id)                
        
        if event_topic_ids != None and ad_topic_ids != None:
            topic_score = get_similarity_score(event_topic_ids, ad_topic_ids)
        else:
            topic_score = 0.0
        if event_cat_ids != None and ad_cat_ids != None:
            cat_score = get_similarity_score(event_cat_ids, ad_cat_ids)                          
        else:
            cat_score = 0.0
        
        total_score = 5 * total_score + 0.45 * topic_score + 0.05 * cat_score                                                                                                                                                                        
        output_file.write("%s,%s,%f\n" %(params[0],params[1],total_score))    
       
    #closing files
    output_file.close()
    input_file.close()
    print("Number of learning scores %d" %count)
    
    del topics
    del categories
    del event_doc_dict
    gc.collect()

'''
return the second item of the list as the score
'''
def get_score(item):
    return item[1]

'''
list_items is a list of [ad_id, similarity_score]
'''
def get_sorted_items(list_items):
    sorted_items = sorted(list_items, key=get_score, reverse=True)
    res1 = [int(item[0]) for item in sorted_items]
    res2 = " ".join(["%s" % item[0] for item in sorted_items])
    return res1, res2

'''
Input file: display_id, ad_id, similarity_score
Output file: display_id, sorted list of ad_id
'''
def predict(is_validation=True):
    result = []
    if is_validation == True:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation_score.csv"), "r")     
    else:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_score.csv"), "r")     
    
    #skip the title
    line = input_file.readline().strip()    
    
    #get the first row
    line = input_file.readline().strip()
    params = line.split(",")
    current_display_id = params[0]
    ad_id = params[1]
    prob = float(params[2])
    
    #initialize variables
    list_items = []
    list_items.append([ad_id,prob])
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
        prob = float(params[2])   
        
        if display_id != current_display_id:     
            output_file.write(current_display_id + ",")   
            res1, res2 = get_sorted_items(list_items) 
                
            #output current_display_id and reset values
            if is_validation==True:
                result.append(res1)
   
            output_file.write(res2 + "\n")
            list_items = []
            current_display_id = display_id
        
        #add information to list    
        list_items.append([ad_id,prob])

    #output the last one
    output_file.write(current_display_id + ",")   
    res1, res2 = get_sorted_items(list_items)   
   
    if is_validation==True: 
        result.append(res1)
    output_file.write(res2 + "\n")

    #closing files
    output_file.close()             
    input_file.close()             
    
    if is_validation == True:
        train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_test_validation.csv")) 
        y = train[train.clicked==1].ad_id.values
        y = [[_] for _ in y]                                                
    
        del train
        gc.collect()        
        
        print("start evaluation")
        print(mapk(y, result, k=12))                                              

#main method
def main_method(is_validation=True):
    '''
    merge_click_event_ad()
    print("merge data has been done")
    '''
    if is_validation==True:
        '''
        split_train_validate("input/clicks_train_doc.csv")
        build_dict_event_doc_clicked("input/clicks_train_validation.csv", is_validation)
        print("dict_clicked has been built")
        '''
        default_score = build_dict_event_doc_score("input/clicks_train_validation.csv", is_validation)
    else:
        '''
        build_dict_event_doc_clicked("input/clicks_train_doc.csv", is_validation)
        print("dict_clicked has been built")
        '''
        default_score = build_dict_event_doc_score("input/clicks_train_doc.csv", is_validation)

    print("start calculate score")    
    calculate_score(default_score, is_validation)
    predict(is_validation)
    
#===============================================================================
#merge_click_event_ad()
#build_dict_event_doc_clicked()
#build_dict_event_doc_score()
#calculate_score()
#predict(is_validation=False)
main_method(is_validation=False)