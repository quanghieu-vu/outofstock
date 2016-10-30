import os, gc
import pandas as pd

from my_utils import load_data_from_csv
from ml_metrics import mapk

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

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
training_data, users_to_index, items_to_index = load_data_from_csv(os.path.join(HOME_DIR,"input/clicks_train_validation.csv"))

from my_bpr import BPR
bpr = BPR(100, len(users_to_index.keys()), len(items_to_index.keys()))
bpr.train(training_data, epochs=5)

input_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation.csv"), "r")
output_file = open(os.path.join(HOME_DIR,"input/clicks_test_validation_score.csv"), "w")

line = input_file.readline().strip()    
output_file.write("display_id,ad_id,score\n")
total_item = 0
current_event_doc_id = ""
current_event_pred = None

while 1:
    line = input_file.readline().strip()
    if line == '':
        break
        
    total_item += 1
    if total_item % 1000000 == 0:
        print('Finished {} items...'.format(total_item))
    
    display_id, ad_id, clicked, event_doc_id, ad_doc_id = line.split(",")
    
    if event_doc_id != current_event_doc_id:
        current_event_doc_id = event_doc_id
        if current_event_doc_id in users_to_index:
            current_event_pred = bpr.predictions(users_to_index.get(current_event_doc_id))
        else:
            current_event_pred = None

    if current_event_pred == None:
        output_file.write("%s,%s,0\n" %(display_id,ad_id))
    else:
        if ad_doc_id in items_to_index:
            score = current_event_pred[items_to_index.get(ad_doc_id)]
            output_file.write("%s,%s,%f\n" %(display_id,ad_id,score))
        else:
            output_file.write("%s,%s,0\n" %(display_id,ad_id))
        
output_file.close()
input_file.close()

predict(is_validation=True)