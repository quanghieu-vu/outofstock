import pandas as pd
import os, sys
import gc
import json

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

    #update for 87141731 rows of train
    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train.csv"))
    print(train.shape)
    print(train.head(5))

    train = pd.merge(train, event_info, on="display_id", how="left")
    print(train.head(5))

    train = pd.merge(train, ad_info, on="ad_id", how="left")
    print(train.head(5))

    train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_doc.csv"), index=False)
    
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

    test.to_csv(os.path.join(HOME_DIR,"input/clicks_test_doc.csv"), index=False)

    del test
    del event_info
    del ad_info
    gc.collect()

'''
build topics and categories dictionary for documents
key is document_id
value is list of topics/categories
'''
def build_dictionaries():
    topic_info = pd.read_csv(os.path.join(HOME_DIR,"input/documents_topics.csv"), usecols=["document_id", "topic_id"])
    topic_grouped = topic_info.groupby("document_id")
    topics = {k: list(v) for k,v in topic_grouped["topic_id"]}    
    with open(os.path.join(HOME_DIR,"dict/topics.dict"), "w") as f:
        json.dump(topics, f)
    
    category_info = pd.read_csv(os.path.join(HOME_DIR,"input/documents_categories.csv"), usecols=["document_id", "category_id"])
    category_grouped = category_info.groupby("document_id")
    categories = {k: list(v) for k,v in category_grouped["category_id"]}
    with open(os.path.join(HOME_DIR,"dict/categories.dict"), "w") as f:
        json.dump(categories, f)

'''
merge clicks_train_doc.csv and clicks_test_doc.csv with documents_topics.csv and documents_categories.csv
output files are:
    clicks_train_doc_topic_cat.csv(display_id, ad_id, clicked, event_doc_id, ad_doc_id, event_topic, ad_topic, event_cat, doc_cat)
    clicks_test_doc_topic_cat.csv(display_id, ad_id, event_doc_id, ad_doc_id, event_topic, ad_topic, event_cat, ad_cat)
this version requires less memory
'''
def calculate_similarity_score():
    #load dictionary
    with open(os.path.join(HOME_DIR,"dict/topics.dict"), "r") as f:
        topics = json.load(f)    
    with open(os.path.join(HOME_DIR,"dict/categories.dict"), "r") as f:
        categories = json.load(f)        
    print("Dictionaries have been loaded")
                
    #update training data set
    file_list = [
        os.path.join(HOME_DIR,"input/clicks_test_doc.csv"), \
        os.path.join(HOME_DIR,"input/clicks_train_doc.csv"), \
        os.path.join(HOME_DIR,"input/clicks_test_content_similarity.csv"), \
        os.path.join(HOME_DIR,"input/clicks_train_content_similarity.csv"), \
        ]
    
    for i in range(2):
        count = 0
        input_file = open(file_list[i], "r")
        output_file = open(file_list[i + 2], "w")
        
        line = input_file.readline().strip()    
        if i == 0: #test
            output_file.write("display_id,ad_id,score\n")
        else: #train
            output_file.write("display_id,ad_id,clicked,score\n")

        while 1:
            line = input_file.readline().strip()
            if line == '':
                break
    
            params = line.split(",")

            event_doc_id = params[i+2]
            event_topic_ids = topics.get(event_doc_id)    
            event_cat_ids = categories.get(event_doc_id)
        
            ad_doc_id = params[i+3]
            ad_topic_ids = topics.get(ad_doc_id)
            ad_cat_ids = categories.get(ad_doc_id)
        
            topic_score = 0
            if event_topic_ids != None and ad_topic_ids != None:
                common_topics = list(set(event_topic_ids) & set(ad_topic_ids))
                topic_score = topic_score + 0.1 * len(common_topics)
        
            cat_score = 0
            if event_cat_ids != None and ad_cat_ids != None:
                common_cats = list(set(event_cat_ids) & set(ad_cat_ids))
                cat_score = cat_score + 0.1 * len(common_cats)
        
            total_score = topic_score * 0.45 + cat_score * 0.05    
            if total_score != 0:
                count = count + 1
                            
            if i == 0:
                output_file.write("%s,%s,%f\n" %(params[0],params[1],total_score))    
            else:
                output_file.write("%s,%s,%s,%f\n" %(params[0],params[1],params[2],total_score))    
        
        #closing files
        output_file.close()
        input_file.close()
        print("Number of non-zero scores %d" %count)
    
'''
get the average click for pairs of (event_doc_id, ad_doc_id)
'''
def get_mean_click_event_ad():
    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train_doc.csv"), usecols=["event_doc_id","ad_doc_id","clicked"])    
    print(train.head(5))
    
    train_dict = pd.DataFrame({'prob':train.groupby(["event_doc_id","ad_doc_id"])['clicked'].mean()}).reset_index()      
    print(train_dict.shape)   
    print(train_dict.head(5))

    del train
    gc.collect()    

    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train_doc.csv"))    
    train = pd.merge(train, train_dict, on=["event_doc_id","ad_doc_id"], how="left")   
    train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_event_ad_mean.csv"), index=False)
    
    del train
    gc.collect()
        
    test = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_test_doc.csv")) 
    test = pd.merge(test, train_dict, on=["event_doc_id","ad_doc_id"], how="left")   
    print("Number of NA prob %d" %len(test[(test["prob"].isnull())]))
    test.ix[test["prob"].isnull(),"prob"] = 0 
    test.to_csv(os.path.join(HOME_DIR,"input/clicks_test_event_ad_mean.csv"), index=False)
    
    del test    
    del train_dict
    gc.collect()
                                          
#===============================================================================
#merge_click_event_ad()
#get_mean_click_event_ad()
#build_dictionaries()
calculate_similarity_score()