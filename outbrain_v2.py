import pandas as pd
import os, sys
import gc

HOME_DIR = os.path.dirname(os.path.abspath(__file__))

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
predict()												
