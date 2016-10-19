import pandas as pd
import os

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

def get_basic_statistics():
    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train.csv"))    
    print("Total number of rows %d" %train.shape[0])
    
    train_events = train["display_id"].unique()
    print("Total number of events %d" %len(train_events))
    train_ads = train["ad_id"].unique()
    print("Total number of ads %d" %len(train_ads))
    
    test = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_test.csv"))    
    print("Total number of rows %d" %test.shape[0])
    
    test_events = test["display_id"].unique()
    print("Total number of events %d" %len(test_events))
    test_ads = test["ad_id"].unique()
    print("Total number of ads %d" %len(test_ads))

    leakage_events = list(set(train_events) & set(test_events))
    if len(leakage_events) == 0:
        print("There is no leakage event")
    else:
        print(leakage_events)
        
    test_ads_new = list(set(test_ads) - set(train_ads))
    print("Number of new ads in the test set %d" %len(test_ads_new)) 

def get_topic_statistics():
    topics = pd.read_csv(os.path.join(HOME_DIR,"input/documents_topics.csv"))
    print('Number of unique topics: %d' %len(topics.topic_id.unique()))
    print("Min confidence: %f" %topics.confidence_level.min()) 
    print("Mean confidence: %f" %topics.confidence_level.mean()) 
    print("Max confidence: %f" %topics.confidence_level.max()) 
        
def get_category_statistics():
    categories = pd.read_csv(os.path.join(HOME_DIR,"input/documents_categories.csv"))
    print('Number of unique topics: %d' %len(categories.category_id.unique()))
    print("Min confidence: %f" %categories.confidence_level.min())
    print("Mean confidence: %f" %categories.confidence_level.mean()) 
    print("Max confidence: %f" %categories.confidence_level.max()) 

def get_extra_statistics():
    '''
    event_info = pd.read_csv(os.path.join(HOME_DIR,"input/events.csv"))
    event_info.rename(columns={'document_id': 'event_doc_id'}, inplace=True)
    
    train = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train.csv"))
    train = pd.merge(train, event_info, on="display_id", how="left")
    train_uuids = train["uuid"].unique()
    print("Total number of users %d" %len(train_uuids))
    
    test = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_test.csv"))
    test = pd.merge(test, event_info, on="display_id", how="left")
    test_uuids = test["uuid"].unique()
    print("Total number of users %d" %len(test_uuids))
    
    test_uuids_new = list(set(test_uuids) - set(train_uuids))
    print("Number of new users in the test set %d" %len(test_uuids_new))
    '''
    
    train_doc = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train_doc.csv"))    
    matched_train_docs = train_doc[(train_doc["event_doc_id"]==train_doc["ad_doc_id"])]
    if len(matched_train_docs) == 0:
        print("There is no matching document between events and ads in the training set")
    else:
        print(matched_train_docs)
    
    test_doc = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_test_doc.csv"))    
    matched_test_docs = test_doc[(test_doc["event_doc_id"]==test_doc["ad_doc_id"])]
    if len(matched_test_docs) == 0:
        print("There is no matching document between events and ads in the testing set")
    else:
        print(matched_test_docs)
            
#==============================================================================
if __name__ == '__main__':
    #get_basic_statistics()
    get_extra_statistics()
    #get_topic_statistics()
    #get_category_statistics()