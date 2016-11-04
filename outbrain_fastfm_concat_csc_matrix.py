import pandas as pd
import numpy as np
import os, gc
import json

from fastFM import sgd
from scipy.sparse import csc_matrix
from scipy.sparse import vstack
from ml_metrics import mapk

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

'''
get statistics of the input file
'''
def get_num_of_event_doc_id(is_validation=True):
    if is_validation == True:
        train_doc = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train_validation.csv"), usecols=["source_id","publisher_id","ad_id","campaign_id"])
    else:
        train_doc = pd.read_csv(os.path.join(HOME_DIR,"input/clicks_train_doc_sup.csv"), usecols=["source_id","publisher_id","ad_id","campaign_id"])
    
    #get number of publisher_id and ad doc id        
    num_of_source_id = len(train_doc["source_id"].unique())
    num_of_publisher_id = len(train_doc["publisher_id"].unique())
    num_of_ad_id = len(train_doc["ad_id"].unique())
    num_of_campaign_id = len(train_doc["campaign_id"].unique())
   
    del train_doc
    gc.collect()
    
    print("Total number of unique num_of_source_id: %d" %num_of_source_id)
    print("Total number of unique num_of_publisher_id: %d" %num_of_publisher_id)
    print("Total number of unique ad_id: %d" %num_of_ad_id)
    print("Total number of unique campaign id: %d" %num_of_campaign_id)   
     
    return num_of_source_id, num_of_publisher_id, num_of_ad_id, num_of_campaign_id

'''
load train data into sparse matrix
'''
def load_train_data(is_validation=True):
    num_of_cat_id = 97
    num_of_topic_id = 300
    
    #statistics of validation file
    if is_validation==True:
        num_of_source_id, num_of_publisher_id, num_of_ad_doc_id, num_of_campaign_id = 4809, 526, 467402, 32511
    else:
        num_of_source_id, num_of_publisher_id, num_of_ad_doc_id, num_of_campaign_id = get_num_of_event_doc_id(is_validation)
      
    data = []
    row_ind = []
    col_ind = []
    y = []
    
    if is_validation == True:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_train_validation.csv"), "r")
    else:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_train_doc_sup.csv"), "r")
    input_file.readline()
    total_line = 0

    user_source_id    = 0
    user_publisher_id = user_source_id + num_of_source_id    
    item_id           = user_publisher_id + num_of_publisher_id
    ad_campaign_id    = item_id + num_of_ad_doc_id        
    event_cat_id_1    = ad_campaign_id + num_of_campaign_id
    event_cat_id_2    = event_cat_id_1 + num_of_cat_id
    ad_cat_id_1       = event_cat_id_2 + num_of_cat_id
    ad_cat_id_2       = ad_cat_id_1 + num_of_cat_id    
    total_column      = ad_cat_id_2 + num_of_cat_id
    
    print("event_cat_id_1 starts at %d" %event_cat_id_1)
    print("Total column %d" %total_column)
    
    source_dict = {}
    publisher_dict = {}
    item_dict = {}
    campaign_dict = {}
    sparse_matrices = []
    
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        display_id, ad_id, clicked, event_doc_id, source_id, publisher_id, ad_doc_id, campaign_id, \
            event_cat_1, event_cat_2, event_cat_3, event_topic_1, event_topic_2, event_topic_3, \
            ad_cat_1, ad_cat_2, ad_cat_3, ad_topic_1, ad_topic_2, ad_topic_3 = line.split(',')

        #basic features
        if not source_dict.has_key(source_id):
            source_dict[source_id] = user_source_id
            user_source_id += 1        
        if not publisher_dict.has_key(publisher_id):
            publisher_dict[publisher_id] = user_publisher_id
            user_publisher_id += 1
        if not item_dict.has_key(ad_id):
            item_dict[ad_id] = item_id
            item_id += 1
        if not campaign_dict.has_key(campaign_id):
            campaign_dict[campaign_id] = ad_campaign_id
            ad_campaign_id += 1

        data.extend([1, 1, 1, 1])
        row_ind.extend([total_line, total_line, total_line, total_line])
        col_ind.extend([source_dict.get(source_id), publisher_dict.get(publisher_id), item_dict.get(ad_id), campaign_dict.get(campaign_id)])

        #content features
        event_cat_1_value = int(float(event_cat_1))
        if event_cat_1_value != 97:
            data.append(1)
            row_ind.append(total_line)
            col_ind.append(event_cat_id_1 + event_cat_1_value)
        
        event_cat_2_value = int(float(event_cat_2))
        if event_cat_2_value != 97:
            data.append(1)
            row_ind.append(total_line)
            col_ind.append(event_cat_id_2 + event_cat_2_value)
        
        ad_cat_1_value = int(float(ad_cat_1))
        if ad_cat_1_value != 97:
            data.append(1)
            row_ind.append(total_line)
            col_ind.append(ad_cat_id_1 + ad_cat_1_value)
            
        ad_cat_2_value = int(float(ad_cat_2))
        if ad_cat_2_value != 97:
            data.append(1)
            row_ind.append(total_line)
            col_ind.append(ad_cat_id_2 + ad_cat_2_value)
            
        '''
        if platform != "\\N":
            data.append(1)
            row_ind.append(total_line)
            col_ind.append(total_column + int(platform) - 1)
        '''
        y.append(int(clicked))
        
        total_line += 1
        if total_line % 5000000 == 0:
            print('Build csc_matric for %d lines: %d times...' %(total_line, len(sparse_matrices) + 1))     
            x_train = csc_matrix((data, (row_ind, col_ind)), shape=(total_line, total_column), dtype=bool)
            sparse_matrices.append(x_train)

            #clear data
            del data
            del row_ind
            del col_ind
            gc.collect()            
            
            #reset parameters
            total_line = 0 
            data = []
            row_ind = []
            col_ind = []
    
    input_file.close()    
    
    print("build csc_matrix")        
    x_train_rest = csc_matrix((data, (row_ind, col_ind)), shape=(total_line, total_column + 600), dtype=bool)
    sparse_matrices.append(x_train_rest)
    
    print("clear data")
    del data
    del row_ind
    del col_ind
    gc.collect()
    
    print("concat sparse matrices: len %d" %len(sparse_matrices))
    x_train = vstack(sparse_matrices) 
    y_train = np.array(y, dtype=np.int8)

    del sparse_matrices
    del y
    gc.collect()
    
    return x_train, y_train, source_dict, publisher_dict, item_dict, campaign_dict, total_column

'''
load test data into sparse matrix
'''        
def load_test_data(source_dict, publisher_dict, item_dict, campaign_dict, total_column, is_validation=True):
    
    if is_validation == True:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), "r")
    else:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_doc_sup.csv"), "r")
    input_file.readline()
    
    data = []
    row_ind = []
    col_ind = []
    record_line = 0
    sparse_matrices = []
    
    event_cat_id_1    = total_column - (97 * 4)
    event_cat_id_2    = event_cat_id_1 + num_of_cat_id
    ad_cat_id_1       = event_cat_id_2 + num_of_cat_id
    ad_cat_id_2       = ad_cat_id_1 + num_of_cat_id   
    print("event_cat_id_1 starts at %d" %event_cat_id_1)
    
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        if is_validation == True:
            display_id, ad_id, clicked, event_doc_id, source_id, publisher_id, ad_doc_id, campaign_id, \
                event_cat_1, event_cat_2, event_cat_3, event_topic_1, event_topic_2, event_topic_3, \
                ad_cat_1, ad_cat_2, ad_cat_3, ad_topic_1, ad_topic_2, ad_topic_3 = line.split(',')
        else:
            display_id, ad_id, event_doc_id, source_id, publisher_id, ad_doc_id, campaign_id, \
                event_cat_1, event_cat_2, event_cat_3, event_topic_1, event_topic_2, event_topic_3, \
                ad_cat_1, ad_cat_2, ad_cat_3, ad_topic_1, ad_topic_2, ad_topic_3 = line.split(',')
        
        if source_dict.has_key(source_id) and publisher_dict.has_key(publisher_id) and item_dict.has_key(ad_id) and campaign_dict.has_key(campaign_id):            
            #basic features
            data.extend([1, 1, 1, 1])
            row_ind.extend([record_line, record_line, record_line, record_line])
            col_ind.extend([source_dict.get(source_id), publisher_dict.get(publisher_id), item_dict.get(ad_id), campaign_dict.get(campaign_id)])
            
            #content features
            event_cat_1_value = int(float(event_cat_1))
            if event_cat_1_value != 97:
                data.append(1)
                row_ind.append(record_line)
                col_ind.append(event_cat_id_1 + event_cat_1_value)
        
            event_cat_2_value = int(float(event_cat_2))
            if event_cat_2_value != 97:
                data.append(1)
                row_ind.append(record_line)
                col_ind.append(event_cat_id_2 + event_cat_2_value)
        
            ad_cat_1_value = int(float(ad_cat_1))
            if ad_cat_1_value != 97:
                data.append(1)
                row_ind.append(record_line)
                col_ind.append(ad_cat_id_1 + ad_cat_1_value)
            
            ad_cat_2_value = int(float(ad_cat_2))
            if ad_cat_2_value != 97:
                data.append(1)
                row_ind.append(record_line)
                col_ind.append(ad_cat_id_2 + ad_cat_2_value)
    
            '''
            if platform != "\\N":
                data.append(1)
                row_ind.append(record_line)
                col_ind.append(total_column + int(platform) - 1)
            '''
            
            record_line += 1
            if record_line % 3000000 == 0:
                print('Build csc_matric for %d lines: %d times...' %(record_line, len(sparse_matrices) + 1))     
                x_test = csc_matrix((data, (row_ind, col_ind)), shape=(record_line, total_column + 600), dtype=bool)
                sparse_matrices.append(x_test)

                #clear data
                del data
                del row_ind
                del col_ind
                gc.collect()            
            
                #reset parameters
                record_line = 0 
                data = []
                row_ind = []
                col_ind = []
    
    input_file.close()

    print("build csc_matrix")    
    x_test_rest = csc_matrix((data, (row_ind, col_ind)), shape=(record_line, total_column + 600), dtype=bool)
    sparse_matrices.append(x_test_rest)
    
    print("clear data")
    del data
    del row_ind
    del col_ind
    gc.collect()
    
    print("concat sparse matrices: len %d" %len(sparse_matrices))
    x_test = vstack(sparse_matrices) 
  
    del sparse_matrices  
    gc.collect()    
    
    return x_test   

'''
get score of pairs of (event_doc_id, ad_doc_id) and export to a file
'''
def export_score(source_dict, publisher_dict, item_dict, campaign_dict, preds, is_validation=True):
    
    if is_validation == True:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), "r")
        output_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation_score.csv"), "w")
    else:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_doc_sup.csv"), "r")
        output_file = open(os.path.join(HOME_DIR,"input/clicks_test_score.csv"), "w")

    input_file.readline()
    output_file.write("display_id,ad_id,score\n")

    record_line = 0
    
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        if is_validation == True:
            display_id, ad_id, clicked, event_doc_id, source_id, publisher_id, ad_doc_id, campaign_id, \
                event_cat_1, event_cat_2, event_cat_3, event_topic_1, event_topic_2, event_topic_3, \
                ad_cat_1, ad_cat_2, ad_cat_3, ad_topic_1, ad_topic_2, ad_topic_3 = line.split(',')
        else:
            display_id, ad_id, event_doc_id, source_id, publisher_id, ad_doc_id, campaign_id, \
                event_cat_1, event_cat_2, event_cat_3, event_topic_1, event_topic_2, event_topic_3, \
                ad_cat_1, ad_cat_2, ad_cat_3, ad_topic_1, ad_topic_2, ad_topic_3 = line.split(',')
            
        if source_dict.has_key(source_id) and publisher_dict.has_key(publisher_id) and item_dict.has_key(ad_id) and campaign_dict.has_key(campaign_id):            

            output_file.write("%s,%s,%f\n" %(display_id, ad_id, preds[record_line]))            
            record_line += 1
            
            if record_line % 1000000 == 0:
                print('Finished {} lines...'.format(record_line))        
        else:
            output_file.write("%s,%s,0\n" %(display_id, ad_id))
    
    input_file.close()
    output_file.close()

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
def predict(is_validation):
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
    output_file = open(os.path.join(HOME_DIR,"output/submission_fm.csv"), "w")
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
                                                                                                            
#============================================================================== 
is_validation = True

print("Load train")   
x_train, y_train, source_dict, publisher_dict, item_dict, campaign_dict, total_column = load_train_data(is_validation)

print("Start training")
fm = sgd.FMRegression(n_iter=1000000000, init_stdev=0.01, l2_reg_w=0.5, l2_reg_V=0.5, rank=2, step_size=0.0001)
fm.fit(x_train, y_train)

print("Load test")
x_test = load_test_data(source_dict, publisher_dict, item_dict, campaign_dict, total_column, is_validation)

print("Predict")
preds = fm.predict(x_test)

print("Record score")
export_score(source_dict, publisher_dict, item_dict, campaign_dict, preds, is_validation)

del x_train
del x_test
del source_dict
del publisher_dict
del item_dict
del campaign_dict
del preds
gc.collect()

print("Predict and evaluate")
predict(is_validation)