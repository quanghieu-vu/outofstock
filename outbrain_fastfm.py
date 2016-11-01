import pandas as pd
import numpy as np
import os, gc

from fastFM import sgd
from scipy.sparse import csc_matrix
from ml_metrics import mapk

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

def get_num_of_event_doc_id(file_name):
    train_doc = pd.read_csv(os.path.join(HOME_DIR,"input/" + file_name))
    num_of_event_doc_id = len(train_doc["event_doc_id"].unique())
    num_of_ad_doc_id = len(train_doc["ad_doc_id"].unique())
    
    del train_doc
    gc.collect()
    
    print("Total number of unique event_doc_id: %d" %num_of_event_doc_id)
    print("Total number of unique ad_doc_id: %d" %num_of_ad_doc_id)    
    return num_of_event_doc_id, num_of_ad_doc_id

def load_train_data(file_name):
    num_of_event_doc_id, num_of_ad_doc_id = get_num_of_event_doc_id(file_name)
    data = []
    row_ind = []
    col_ind = []
    y = []
    
    input_file = open(os.path.join(HOME_DIR,"input/" + file_name), "r")
    input_file.readline()
    total_line = 0
    
    user_id = 0
    item_id = num_of_event_doc_id
    user_dict = {}
    item_dict = {}
    
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        display_id, ad_id, clicked, event_doc_id, ad_doc_id = line.split(',')
        
        if not user_dict.has_key(event_doc_id):
            user_dict[event_doc_id] = user_id
            user_id += 1
        if not item_dict.has_key(ad_doc_id):
            item_dict[ad_doc_id] = item_id
            item_id += 1
        
        data.extend([1, 1])
        row_ind.extend([total_line, total_line])
        col_ind.extend([user_dict.get(event_doc_id), item_dict.get(ad_doc_id)])        
        y.append(int(clicked))

        total_line += 1
        if total_line % 5000000 == 0:
            print('Finished {} lines...'.format(total_line))
    
    input_file.close()
    print(data[:10])
    print(row_ind[:10])
    print(col_ind[:10])
        
    total_column = num_of_event_doc_id + num_of_ad_doc_id
    x_train = csc_matrix((data, (row_ind, col_ind)), shape=(total_line, total_column), dtype=np.int8)
    
    del data
    del row_ind
    del col_ind
    gc.collect()
    
    y_train = np.array(y, dtype=np.int8)
    del y
    gc.collect()
    
    return x_train, y_train, user_dict, item_dict, total_column
    
def load_test_data(file_name, user_dict, item_dict, total_column):
    
    input_file = open(os.path.join(HOME_DIR,"input/" + file_name), "r")
    input_file.readline()
    
    data = []
    row_ind = []
    col_ind = []
    record_line = 0
    
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        display_id, ad_id, clicked, event_doc_id, ad_doc_id = line.split(',')
        if user_dict.has_key(event_doc_id) and item_dict.has_key(ad_doc_id):
            data.extend([1, 1])
            row_ind.extend([record_line, record_line])
            col_ind.extend([user_dict.get(event_doc_id), item_dict.get(ad_doc_id)])
        
            record_line += 1
            if record_line % 1000000 == 0:
                print('Recorded {} lines...'.format(record_line))        
    
    input_file.close()
    print(data[:10])
    print(row_ind[:10])
    print(col_ind[:10])
    
    x_test = csc_matrix((data, (row_ind, col_ind)), shape=(record_line, total_column), dtype=np.int8)
    
    del data
    del row_ind
    del col_ind
    gc.collect()
    
    return x_test   

def export_score(user_dict, item_dict, preds):
    
    input_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), "r")
    input_file.readline()
    
    output_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation_score.csv"), "w")
    output_file.write("display_id,ad_id,score\n")

    record_line = 0
    
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        display_id, ad_id, clicked, event_doc_id, ad_doc_id = line.split(',')
        if user_dict.has_key(event_doc_id) and item_dict.has_key(ad_doc_id):
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
                                                                                                            
#============================================================================== 
print("Load train")   
x_train, y_train, user_dict, item_dict, total_column = load_train_data("clicks_train_validation.csv")

print("Start training")
fm = sgd.FMRegression(n_iter=10000, init_stdev=0.01, l2_reg_w=0.5, l2_reg_V=50.5, rank=2, step_size=0.0001)
fm.fit(x_train, y_train)

print("Load test")
x_test = load_test_data("clicks_test_validation.csv", user_dict, item_dict, total_column)

print("Predict")
preds = fm.predict(x_test)
print(preds[:10])
print(preds[11])

print("Record score")
export_score(user_dict, item_dict, preds)

del x_train
del x_test
del user_dict
del item_dict
gc.collect()

print("Predict and evaluate")
predict(is_validation=True)