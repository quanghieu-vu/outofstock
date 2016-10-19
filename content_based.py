import pandas as pd
import os, sys
import gc

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

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
    return " ".join(["%s" % item[0] for item in sorted_items])

'''
Input file: display_id, ad_id, similarity_score
Output file: display_id, sorted list of ad_id
'''
def predict(is_validation=True):
    if is_validation == True:
        input_file = open(os.path.join(HOME_DIR,"input/trains_test_preprocessing.csv"), "r")     
    else:
        input_file = open(os.path.join(HOME_DIR,"input/clicks_test_preprocessing.csv"), "r")     
    
    #skip the column title and get the first row
    line = input_file.readline().strip()    
    line = input_file.readline().strip()
    
    #extract information of the first row
    params = line.split(",")
    ad_id = params[1]
    score = float(params[2])    
    
    #initialize variables
    list_items = []
    list_items.append([ad_id,score])
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
            sorted_items = get_sorted_items(list_items) 
                            
            #output current_display_id and reset values
            output_file.write(current_display_id + ",")   
            output_file.write(sorted_items + "\n")
            
            #reset variables
            list_items = []
            current_display_id = display_id
        
        #add information to list    
        list_items.append([ad_id,prob])

    #output the last one
    output_file.write(current_display_id + ",")   
    sorted_items = get_sorted_items(list_items)    
    output_file.write(sorted_items + "\n")
    output_file.close()                      
                                                
#===============================================================================
#merge_data()
#preprocess()
predict()												
