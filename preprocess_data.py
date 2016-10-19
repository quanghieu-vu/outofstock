import pandas as pd
import os
import gc

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
merge clicks_train_doc.csv and clicks_test_doc.csv with documents_topics.csv and documents_categories.csv
output files are:
    clicks_train_doc_topic_cat.csv(display_id, ad_id, clicked, event_doc_id, ad_doc_id, event_topic, ad_topic, event_cat, doc_cat)
    clicks_test_doc_topic_cat.csv(display_id, ad_id, event_doc_id, ad_doc_id, event_topic, ad_topic, event_cat, ad_cat)
'''
def merge_click_event_ad_doc():
    topics = pd.read_csv(os.path.join(HOME_DIR,"input/documents_topics.csv"), usecols=["document_id", "topic_id"])
    categories = pd.read_csv(os.path.join(HOME_DIR,"input/documents_categories.csv"), usecols=["document_id", "category_id"])
    
    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train_doc.csv"))
    train = pd.merge(train, topics, left_on="event_doc_id", right_on="document_id", how="left")
    train = pd.merge(train, topics, left_on="ad_doc_id", right_on="document_id", how="left")
    train = pd.merge(train, categories, left_on="event_doc_id", right_on="document_id", how="left")
    train = pd.merge(train, categories, left_on="ad_doc_id", right_on="document_id", how="left")
    print(train.head(5))
    
    train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_doc_topic_cat.csv"), index=False)
    
    del train
    gc.collect()
    
    test = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_test_doc.csv"))
    test = pd.merge(test, topics, left_on="event_doc_id", right_on="document_id", how="left")
    test = pd.merge(test, topics, left_on="ad_doc_id", right_on="document_id", how="left")
    test = pd.merge(test, categories, left_on="event_doc_id", right_on="document_id", how="left")
    test = pd.merge(test, categories, left_on="ad_doc_id", right_on="document_id", how="left")
    print(test.head(5))
    
    test.to_csv(os.path.join(HOME_DIR,"input/clicks_test_doc_topic_cat.csv"), index=False)
    
    del test
    gc.collect()

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

def get_prob(item):
    return item[1]

def get_sorted_items(lis_items):
    sorted_items = sorted(lis_items, key=get_prob, reverse=True)
    return " ".join(["%s" % item[0] for item in sorted_items])
                                           
#===============================================================================
#merge_click_event_ad()
merge_click_event_ad_doc()
#get_mean_click_event_ad()