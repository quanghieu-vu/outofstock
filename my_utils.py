def load_data_from_csv(csv_file, users_to_i = {}, items_to_i = {}):
    """
      Loads data from a CSV file located at `csv_file` 
      where each line is of the form:
        display_id, ad_id, clicked, event_doc_id, ad_doc_id
      Initial mappings from user and item identifiers
      to integers can be passed using `users_to_i`
      and `items_to_i` respectively.
      This function will return a data array consisting
      of (user, item) tuples, a mapping from user ids to integers
      and a mapping from item ids to integers.
    """    
    
    data = []
    if len(users_to_i.values()) > 0:
        u = max(users_to_i.values()) + 1
    else:
        u = 0
    if len(items_to_i.values()) > 0:
        i = max(items_to_i.values()) + 1
    else:
        i = 0
    
    total_item = 0
    input_file = open(csv_file, "r")
    line = input_file.readline().strip()    
    
    while 1:
        line = input_file.readline().strip()
        if line == '':
            break
    
        total_item += 1
        if total_item % 5000000 == 0:
            print('Finished {} items...'.format(total_item))
         
        display_id, ad_id, clicked, event_doc_id, ad_doc_id = line.split(",")
        if int(clicked) == 1:
            if not users_to_i.has_key(event_doc_id):
                users_to_i[event_doc_id] = u
                u += 1
            if not items_to_i.has_key(ad_doc_id):
                items_to_i[ad_doc_id] = i
                i += 1
            data.append((users_to_i[event_doc_id], items_to_i[ad_doc_id]))
    input_file.close()
    return data, users_to_i, items_to_i
