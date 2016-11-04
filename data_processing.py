import pandas as pd
import numpy as np
import os, sys
import gc
import json

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

'''
split training data into training and testing for validation
'''
def split_train_validate(file_name):     
    train = pd.read_csv(os.path.join(HOME_DIR,"input/" + file_name), usecols=["display_id"])
    ids = train.display_id.unique()
    
    np.random.seed(13)
    ids = np.random.choice(ids, size=len(ids)//10, replace=False)
    
    del train
    gc.collect()
    
    print("A number of %d ids has been randomly selected " %len(ids))

    CHUNK_SIZE = 10 ** 6    
    chunks = pd.read_csv(os.path.join(HOME_DIR,"input/" + file_name), chunksize=CHUNK_SIZE)

    chunk_index = 0    
    for chunk in chunks:
        valid = chunk[chunk.display_id.isin(ids)]
        train = chunk[~chunk.display_id.isin(ids)]
        
        if chunk_index == 0:
            valid.to_csv(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), mode="w", header=True, index=False)
            train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_validation.csv"), mode="w", header=True, index=False)
        else:
            valid.to_csv(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), mode="a", header=False, index=False)
            train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_validation.csv"), mode="a", header=False, index=False)
        
        del valid
        del train
        gc.collect()
        
        chunk_index += 1
        print("Finish trunk %d" %chunk_index)
     
    ''' 
    #very slow           
    input_file = open(os.path.join(HOME_DIR,"input/" + file_name), "r")
    output_file_train = open(os.path.join(HOME_DIR,"input/clicks_train_validation.csv"), "w")
    output_file_test = open(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), "w")    
    
    line = input_file.readline().strip()    
    output_file_train.write("%s\n" %line)
    output_file_test.write("%s\n" %line)
    
    total_row = 0
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        total_row += 1
        if total_row % 1000000 == 0:
            print('Read {} lines...'.format(total_row))
            
        display_id = line.split(",")[0]
        if int(display_id) in ids:
            output_file_test.write("%s\n" %line)
        else:
            output_file_train.write("%s\n" %line)
    
    input_file.close()
    output_file_test.close()
    output_file_train.close()
    '''
    '''
    valid = train[train.display_id.isin(ids)]
    train = train[~train.display_id.isin(ids)]

    valid.to_csv(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), index=False)
    train.to_csv(os.path.join(HOME_DIR,"input/clicks_train_validation.csv"), index=False)
    
    del train
    del valid
    gc.collect()
    '''
    
'''
merge clicks_train.csv and clicks_test.csv with events.csv and promoted_content.csv
output files are:
    clicks_train_doc.csv(display_id, ad_id, clicked, event_doc_id, platform, ad_doc_id)
    clicks_test_doc.csv(display_id, ad_id, event_doc_id, platform, ad_doc_id)
'''
def merge_click_event_ad():
    '''
    event_info = pd.read_csv(os.path.join(HOME_DIR,"input/events.csv"), usecols=["display_id", "document_id", "platform"])
    event_info.rename(columns={'document_id': 'event_doc_id'}, inplace=True)
    '''
    event_info = pd.read_csv(os.path.join(HOME_DIR,"input/events_doc.csv"), usecols=["display_id", "document_id", "source_id", "publisher_id"])
    event_info.rename(columns={'document_id': 'event_doc_id'}, inplace=True)
    #event_info = pd.read_csv(os.path.join(HOME_DIR,"input/events_doc.csv"), usecols=["display_id", "source_id", "publisher_id", "platform"])
    '''
    ad_info = pd.read_csv(os.path.join(HOME_DIR,"input/promoted_content.csv"), usecols=["ad_id", "document_id"])
    ad_info.rename(columns={'document_id': 'ad_doc_id'}, inplace=True)
    '''
    ad_info = pd.read_csv(os.path.join(HOME_DIR,"input/promoted_content_doc.csv"), usecols=["ad_id", "document_id", "campaign_id"])
    ad_info.rename(columns={'document_id': 'ad_doc_id'}, inplace=True)
    #ad_info.rename(columns={'source_id': 'ad_source_id', 'publisher_id': 'ad_publisher_id'}, inplace=True)
    
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
merge event.csv and promoted_content.csv with documents_meta.csv
output files are:
    events_doc.csv(display_id, uuid, document_id, timestamp, platform, geo_location)
    promoted_content_doc.csv(ad_id, document_id, campaign_id, advertiser_id)
'''
def merge_event_doc():
    doc_info = pd.read_csv(os.path.join(HOME_DIR,"input/documents_meta.csv"), usecols=["document_id", "source_id", "publisher_id"])
    doc_info["source_id"].fillna(doc_info["document_id"] + 15000, inplace=True)
    doc_info["publisher_id"].fillna(doc_info["document_id"] + 1500, inplace=True)
    doc_info[["source_id", "publisher_id"]] = doc_info[["source_id", "publisher_id"]].astype(int)
    
    event_info = pd.read_csv(os.path.join(HOME_DIR,"input/events.csv"))
    event_info = pd.merge(event_info, doc_info, on="document_id", how="left")
    event_info.to_csv(os.path.join(HOME_DIR,"input/events_doc.csv"), index=False)
    
    del doc_info
    del event_info
    gc.collect()

def merge_ad_doc():
    doc_info = pd.read_csv(os.path.join(HOME_DIR,"input/documents_meta.csv"), usecols=["document_id", "source_id", "publisher_id"])
    doc_info["source_id"].fillna(doc_info["document_id"] + 15000, inplace=True)
    doc_info["publisher_id"].fillna(doc_info["document_id"] + 1500, inplace=True)
    doc_info[["source_id", "publisher_id"]] = doc_info[["source_id", "publisher_id"]].astype(int)
    
    ad_info = pd.read_csv(os.path.join(HOME_DIR,"input/promoted_content.csv"))
    ad_info = pd.merge(ad_info, doc_info, on="document_id", how="left")
    ad_info.to_csv(os.path.join(HOME_DIR,"input/promoted_content_doc.csv"), index=False)

    del doc_info
    del ad_info
    gc.collect()
    
'''
build topics and categories dictionary for documents
key is document_id
value is list of topics/categories
'''
def build_dictionaries(version=1):
    if version==1:
        topic_info = pd.read_csv(os.path.join(HOME_DIR,"input/documents_topics.csv"), usecols=["document_id", "topic_id"])
        topic_grouped = topic_info.groupby("document_id")
        topics = {k: list(v) for k,v in topic_grouped["topic_id"]}    
        with open(os.path.join(HOME_DIR,"dict/doc_topics.dict"), "w") as f:
            json.dump(topics, f)
    
        category_info = pd.read_csv(os.path.join(HOME_DIR,"input/documents_categories.csv"), usecols=["document_id", "category_id"])
        category_grouped = category_info.groupby("document_id")
        categories = {k: list(v) for k,v in category_grouped["category_id"]}
        with open(os.path.join(HOME_DIR,"dict/doc_cats.dict"), "w") as f:
            json.dump(categories, f)
            
    elif version==2:
        ifiles = [os.path.join(HOME_DIR,"input/documents_topics.csv"), os.path.join(HOME_DIR,"input/documents_categories.csv")]
        ofiles = [os.path.join(HOME_DIR,"dict/doc_topics_conf.dict"), os.path.join(HOME_DIR,"dict/doc_cats_conf.dict")]
                
        for i in range(2):
            if i == 0:
                continue
                
            f = open(ifiles[i], "r")
            f.readline()
            my_dict = dict()
            total_row = 0
        
            if i==1:
                cat_dict = {}
                cat_id = 0    
        
            while 1:
                line = f.readline().strip()
                if line == '':
                    break
        
                total_row += 1
                if total_row % 1000000 == 0:
                    print('Read {} lines...'.format(total_row))
        
                params = line.split(",")
                key = params[0]
                class_id = params[1] #class_id is either topic_id or category_id
                confidence = params[2]
                
                if i == 0:
                    if key in my_dict:
                        my_dict[key].append([class_id,confidence])
                    else:
                        my_dict[key] = [[class_id,confidence]]
                else:
                    if not cat_dict.has_key(class_id):
                        cat_dict[class_id] = cat_id
                        cat_id += 1   
                                                  
                    updated_cat_id = cat_dict.get(class_id)               
                    if key in my_dict:
                        my_dict[key].append([updated_cat_id,confidence])
                    else:
                        my_dict[key] = [[updated_cat_id,confidence]]                
                
            f.close()                                
            with open(ofiles[i], "w") as f:
                json.dump(my_dict, f)
            
            print(cat_id)

'''
list1, list2 are lists of pairs(item, confidence)
'''                                
def get_similarity_score(list1, list2):
    score = 0.0
    
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i][0] == list2[j][0]:
                score = score + float(list1[i][1]) * float(list2[j][1])
            '''
            elif list1[i][0] in :
            '''    
    return score
                                
'''
merge clicks_train_doc.csv and clicks_test_doc.csv with documents_topics.csv and documents_categories.csv
output files are:
    clicks_train_doc_topic_cat.csv(display_id, ad_id, clicked, event_doc_id, ad_doc_id, event_topic, ad_topic, event_cat, doc_cat)
    clicks_test_doc_topic_cat.csv(display_id, ad_id, event_doc_id, ad_doc_id, event_topic, ad_topic, event_cat, ad_cat)
this version requires less memory
'''
def calculate_similarity_score(version=1):
    #load dictionary
    if version == 1:
        with open(os.path.join(HOME_DIR,"dict/doc_topics.dict"), "r") as f:
            topics = json.load(f)    
        with open(os.path.join(HOME_DIR,"dict/doc_cats.dict"), "r") as f:
            categories = json.load(f)        
    elif version == 2:
        with open(os.path.join(HOME_DIR,"dict/doc_topics_conf.dict"), "r") as f:
            topics = json.load(f)    
        with open(os.path.join(HOME_DIR,"dict/doc_cats_conf.dict"), "r") as f:
            categories = json.load(f)        
    print("Dictionaries have been loaded")
                
    #update training data set
    file_list = [
        os.path.join(HOME_DIR,"input/clicks_test_doc.csv"), \
        os.path.join(HOME_DIR,"input/clicks_train_doc.csv"), \
        os.path.join(HOME_DIR,"input/clicks_test_content_similarity_v2.csv"), \
        os.path.join(HOME_DIR,"input/clicks_train_content_similarity_v2.csv"), \
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
                if version == 1:
                    common_topics = list(set(event_topic_ids) & set(ad_topic_ids))
                    topic_score = topic_score + 0.1 * len(common_topics)
                elif version == 2:
                    topic_score = topic_score + get_similarity_score(event_topic_ids, ad_topic_ids)
        
            cat_score = 0
            if event_cat_ids != None and ad_cat_ids != None:
                if version == 1:
                    common_cats = list(set(event_cat_ids) & set(ad_cat_ids))
                    cat_score = cat_score + 0.1 * len(common_cats)
                elif version == 2:
                    cat_score = cat_score + get_similarity_score(event_cat_ids, ad_cat_ids)
        
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

'''
build a dictionary for event_doc_id of clicked and unclicked ad_doc_id
'''
def build_dict_event_doc_clicked():
    input_file = open(os.path.join(HOME_DIR,"input/clicks_train_doc.csv"), "r")
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
        key = params[3]
        ad_doc_id = params[4]
        
        #clicked ad_doc_ids are in the first list and not-clicked ones are in the second list
        if key in event_doc_dict:
            if clicked == 1:
                event_doc_dict[key][0].append(ad_doc_id)
            else:
                event_doc_dict[key][1].append(ad_doc_id)
        else:
            if clicked == 1:
                event_doc_dict[key] = [[ad_doc_id],[]]
            else:
                event_doc_dict[key] = [[],[ad_doc_id]]

    with open(os.path.join(HOME_DIR,"dict/event_doc_clicked.dict"), "w") as f:
        json.dump(event_doc_dict, f)                                                                                                                            

'''
return the second item of the list as the score
'''
def get_score(item):
    return item[1]

'''
list_items is a list of [topic_id, confidence]
'''
def get_sorted_items(list_items):
    sorted_items = sorted(list_items, key=get_score, reverse=True)
    result = [(int(item[0]) + float(item[1])) for item in sorted_items]
    return result

def add_topics_cats(is_train_file=True, catK=3, topK=3):
    if is_train_file==True:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_train_doc.csv"), "r")
        output_file = open(os.path.join(HOME_DIR,"input/clicks_train_doc_sup.csv"), "w")
        output_file.write("display_id,ad_id,clicked,event_doc_id,source_id,publisher_id,ad_doc_id,campaign_id")
    else:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_doc.csv"), "r")
        output_file = open(os.path.join(HOME_DIR,"input/clicks_test_doc_sup.csv"), "w")
        output_file.write("display_id,ad_id,event_doc_id,source_id,publisher_id,ad_doc_id,campaign_id")
    
    for i in range(catK):        
        output_file.write(",event_cat_%d" %(i + 1))
    for i in range(topK):        
        output_file.write(",event_topic_%d" %(i + 1))        
    for i in range(catK):                        
        output_file.write(",ad_cat_%d" %(i + 1))
    for i in range(topK):        
        output_file.write(",ad_topic_%d" %(i + 1))
    output_file.write("\n")
    input_file.readline()    
    
    with open(os.path.join(HOME_DIR,"dict/doc_topics_conf.dict"), "r") as f:
        topics = json.load(f)
    with open(os.path.join(HOME_DIR,"dict/doc_cats_conf.dict"), "r") as f:
        categories = json.load(f)    
            
    total_line = 0
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break      
    
        if is_train_file == True:
            display_id, ad_id, clicked, event_doc_id, source_id, publisher_id, ad_doc_id, campaign_id = line.split(',')
        else:
            display_id, ad_id, event_doc_id, source_id, publisher_id, ad_doc_id, campaign_id = line.split(',')
        output_file.write(line)

        event_cat_ids = categories.get(event_doc_id)
        if event_cat_ids == None:
            event_cat_result = [97] * topK
        else:
            event_cat_result = get_sorted_items(event_cat_ids)
            while len(event_cat_result) < topK:
                event_cat_result.append(97)                
        for i in range(catK):
            output_file.write(",%f" %event_cat_result[i])
                                                
        event_topic_ids = topics.get(event_doc_id)
        if event_topic_ids == None:
            event_topic_result = [300] * topK
        else:
            event_topic_result = get_sorted_items(event_topic_ids)
            while len(event_topic_result) < topK:
                event_topic_result.append(300)
        for i in range(topK):
            output_file.write(",%f" %event_topic_result[i])
                
        ad_cat_ids = categories.get(ad_doc_id)        
        if ad_cat_ids == None:
            ad_cat_result = [97] * topK
        else:
            ad_cat_result = get_sorted_items(ad_cat_ids)
            while len(ad_cat_result) < topK:
                ad_cat_result.append(97)
        for i in range(catK):
            output_file.write(",%f" %ad_cat_result[i])

        ad_topic_ids = topics.get(ad_doc_id)        
        if ad_topic_ids == None:
            ad_topic_result = [300] * topK
        else:
            ad_topic_result = get_sorted_items(ad_topic_ids)
            while len(ad_topic_result) < topK:
                ad_topic_result.append(300)
        for i in range(topK):
            output_file.write(",%f" %ad_topic_result[i])
            
        output_file.write("\n")
        total_line += 1
        
        if total_line % 5000000 == 0:
            print('Finished {} lines...'.format(total_line))
            #sys.exit(0)
            
    input_file.close()
    output_file.close()

#===============================================================================
#merge_event_doc()
#merge_ad_doc()
#merge_click_event_ad()
#get_mean_click_event_ad()
#build_dictionaries(version=2)
#calculate_similarity_score(version=2)
#add_topics_cats(is_train_file=True, catK=3, topK=3)
#add_topics_cats(is_train_file=False, catK=3, topK=3)
split_train_validate("clicks_train_doc_sup.csv")
#build_dict_event_doc_clicked()