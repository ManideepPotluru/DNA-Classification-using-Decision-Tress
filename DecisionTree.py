#!/usr/bin/env python
# coding: utf-8

# In[1675]:


import pandas
import numpy as np
#for chisquare
from scipy.stats import chi2

#we are reading the csv
totaldataset = pandas.read_csv("training.csv", header=None)


# In[1676]:


#coverting the DataFrame to a Numpy array
dataset_numpy = totaldataset.to_numpy()
#deleting the first column 
matrix = np.delete(dataset_numpy, 0, 1)
#splitting the dna into 60 and appending labels to that list forming a list of 61 elements
reshape_numpy = []
for row in matrix:
    dna_seq = row[0]
    dna_list = list(dna_seq)
    dna_list.append(row[1])
    reshape_numpy.append(dna_list)

np_dataset = np.array(reshape_numpy)


# In[1677]:


def gini_index(column): #function to find impurity via gini_index at parent
    # we find all the unique lables and their counts
    labels, num_labels = np.unique(column,return_counts = True)
    total_gini = 0
    gini_label = 0
    #for each label we find the gini, sum them up and do 1 - the sum
    for i in range(len(num_labels)):
        percentage_of_label = num_labels[i] / len(column)
        gini_label += percentage_of_label**2
    total_gini = 1 - gini_label
    return total_gini


# In[1678]:


def misclassification_error(column): #function to find impurity via misclassification_error at parent
    # we find all the unique lables and their counts
    labels, num_labels = np.unique(column,return_counts = True)
    total_misclass = 0
    percentage_of_label=[]
    #for each label we find misclassication_error and do 1 - the maximum 
    for i in range(len(num_labels)):
        ind_percentage_of_label = num_labels[i] / len(column)
        percentage_of_label.append(ind_percentage_of_label)
    misclass_label = np.max(percentage_of_label)
    total_misclass = 1 - misclass_label
    return total_misclass


# In[1679]:


def entropy(column): #function to find impurity via entropy at parent
    # we find all the unique lables and their counts 
    labels, num_labels = np.unique(column,return_counts = True)
    total_entropy = 0
    #for each label we find entropy and sum them up
    for i in range(len(num_labels)):
        percentage_of_label = num_labels[i] / len(column)
        entropy_label = (-1) * percentage_of_label * np.log2(percentage_of_label)
        total_entropy += entropy_label
    return total_entropy


# In[1680]:


def InformationGain(current_dataset, column_name, class_label, method="gini"): #function to find information gain via diff methods.
    
    #We created a nested dictionary that holds the values and counts of the char_val i.e.., 'A','G'...and the char_class_label i.e.., 'N', 'IE', 'EI'
    
    hash_map = {}
    for index, char_val in np.ndenumerate(current_dataset[:, column_name]):
        value_index = index[0]
        char_class_label = current_dataset[:, class_label][value_index]
        if char_val not in hash_map:
            hash_map[char_val] = {}
            if char_class_label not in hash_map[char_val]:
                hash_map[char_val][char_class_label] = 1
            else:
                hash_map[char_val][char_class_label] += 1
        else:
            if char_class_label not in hash_map[char_val]:
                hash_map[char_val][char_class_label] = 1
            else:
                hash_map[char_val][char_class_label] += 1
    
    #Based on the method called one of entropy, gini, or misclass will be executed.
    if method == "gini":
        total_gini = gini_index(current_dataset[:, class_label])
        sum_gini =0
        # for each char in the map, we find the gini and sum it all up
        for char in hash_map.keys():
            labels_counts = hash_map[char].values()
            char_gini = 0
            gini_char_label = 0
            for char_label_count in labels_counts:
                percentage_char_label = char_label_count / sum(labels_counts)
                gini_char_label+= percentage_char_label**2 
            char_gini = 1- gini_char_label
            sum_gini += sum(labels_counts)/ len(current_dataset[:, column_name]) * char_gini
        Information_Gain = total_gini - sum_gini
        return Information_Gain
    
    elif method == "entropy":
        total_entropy = entropy(current_dataset[:, class_label])
        sum_entropy = 0
        # for each char in the map, we find the entropy and sum it all up
        for char in hash_map.keys():
            labels_counts = hash_map[char].values()
            char_entropy = 0
            for char_label_count in labels_counts:
                percentage_char_label = char_label_count / sum(labels_counts)
                entropy_char_label = (-1) * percentage_char_label * np.log2(percentage_char_label)
                char_entropy += entropy_char_label
            sum_entropy += sum(labels_counts)/ len(current_dataset[:, column_name]) * char_entropy
        Information_Gain = total_entropy - sum_entropy
        return Information_Gain
    
    else:
        if method =="misclass":
            total_misclass = misclassification_error(current_dataset[:, class_label])
            sum_misclass =0
            # for each char in the map, we find the misclass and sum it all up
            for char in hash_map.keys():
                labels_counts = hash_map[char].values()
                char_misclass = 0
                percentage_char_label=[]
                for char_label_count in labels_counts:
                    ind_percentage_char_label = char_label_count / sum(labels_counts)
                    percentage_char_label.append(ind_percentage_char_label)
                misclass_char_label = np.max(percentage_char_label)
                char_misclass =1-misclass_char_label
                sum_misclass += sum(labels_counts)/ len(current_dataset[:, column_name]) * char_misclass
            Information_Gain = total_misclass - sum_misclass
            return Information_Gain


# In[1681]:


def ChiSquare(current_dataset, column_name, class_label=60): #find chiSquare for a column
    result_chi = 0
    unique_chars = np.unique(current_dataset[:, column_name]) #find unique chars ('A','G'..) 
    for char in unique_chars:
        branch_dataset = current_dataset[current_dataset[:, column_name] == char] #take a certain char to loop through
        branch_class_names, branch_class_counts = np.unique(branch_dataset[:, class_label], return_counts=True) #collect all the counts of unique labels of that char
        for cls in branch_class_names:
            cls_index = list(branch_class_names).index(cls)
            cls_count = branch_class_counts[cls_index]
            real_n_count = cls_count #get the real count of labels that belong to char
            left_expected = len(branch_dataset) # total branch length
            parent_cls_names, parent_cls_count = np.unique(current_dataset[:, class_label], return_counts=True)
            parent_n_index = list(parent_cls_names).index(cls)
            right_expected = parent_cls_count[parent_n_index] / len(current_dataset)
            expected = left_expected * right_expected
            chi = (real_n_count - expected) ** 2 / expected
            result_chi += chi
    
    freedom = (len(unique_chars) - 1 ) * (len(np.unique(current_dataset[:, class_label])) -1 )
    critical_value = chi2.ppf(0.99, freedom) #calc crit value based on freedom and confidence
    if result_chi > critical_value:
        keep_building = True
    else:
        keep_building = False
    return keep_building


# In[1682]:


def ID3Algorithm(current_dataset, completedataset, characteristics, labels_column=60, upper_node_label = "N"): #we are considering each column positon as a characteristics
    #for the current data if there is only one label in the 60th column, then return that
    if len(np.unique(current_dataset[:, labels_column])) <= 1:
        return np.unique(current_dataset[:, labels_column])[0]
    #if the len of the data is zero then return label which has the highest count in the complete data
    elif len(current_dataset)==0:
        return np.unique(completedataset[:, labels_column])[
            np.argmax(np.unique(completedataset[:, 
                                             labels_column],return_counts=True)[1])]
    #if there are no more features, then we return the label of the parent node
    elif len(characteristics) ==0:
        return upper_node_label
    else:
        upper_node_label = np.unique(current_dataset[:, labels_column])[
            np.argmax(np.unique(current_dataset[:, labels_column],return_counts=True)[1])]
        #ig_set has the list of all IG of all characteristics
        ig_set = [InformationGain(current_dataset,characteristic,labels_column,"misclass") for characteristic in characteristics]
        #top_characteristic_index has the index of the highest value of ig_set
        top_characteristic_index = np.argmax(ig_set)
        #we find the top_characteristic from the list using index
        top_characteristic = characteristics[top_characteristic_index]
        #we add that characteristic to a dictionary
        tree = {top_characteristic:{}}
        #A list characteristics puts in the characteristics as long as it's not equal to the best characteristic
        characteristics = [i for i in characteristics if i != top_characteristic]
        #Check is it is noteworthy to keep building the tree by running the chisquare test
        if (ChiSquare(current_dataset, top_characteristic)):
            for col_value in np.unique(current_dataset[:, top_characteristic]):
                col_value = col_value
                sub_dataset = current_dataset[current_dataset[:, top_characteristic] == col_value] 
                subtree = ID3Algorithm(sub_dataset, completedataset, 
                                  characteristics, labels_column, upper_node_label)            
                tree[top_characteristic][col_value] = subtree
            return(tree)
        else:
            upper_label_node = np.unique(current_dataset[:, labels_column])[
            np.argmax(np.unique(current_dataset[:, 
                                             labels_column],return_counts=True)[1])]
            return(upper_label_node)


# In[1683]:


def predict(instance, tree):
    attribute = list(tree.keys())[0]
    if instance[attribute] in list(tree[attribute].keys()):
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):
            return predict(instance, result)
        elif result is not None:
            return result
        else:
            return "N"
    else:
        return "N"


# In[1684]:


tree = ID3Algorithm(np_dataset, np_dataset, [i for i in range(0, 60)])


# In[1685]:


test_dataset = pd.read_csv("testing.csv", header=None).to_numpy()

result = [[row[0], predict(row[1], tree)] for row in test_dataset]

pandas_frame = pd.DataFrame(result)

pandas_frame.to_csv("cs529_dna_test_misclass_edited_chi2_2_final.csv")
    


# In[ ]:




