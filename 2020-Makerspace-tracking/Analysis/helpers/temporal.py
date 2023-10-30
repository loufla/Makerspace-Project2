# Libraries
import numpy as np


def compute_transitions(actions):
    ''' compute the number of transitions between actions '''
    transitions = 0
    for i,action in enumerate(actions):
        if i+1 >= len(actions): break
        if actions[i+1] != action: transitions +=1
    return transitions

def compute_transitions_matrix(labels, diag=True):
    ''' takes a list of actions / labels and returns a probability matrix
    note: labels have to be numbers'''
    
    # get the size of the matrix
    set_labels = list(set(labels))
    num_clusters = len(set_labels)
    labels = [set_labels.index(x) for x in labels]
    
    # build the numpy matrix
    transitions_matrix = np.zeros((num_clusters, num_clusters))
    prev = None
    
    # count the transitions
    for lab in labels:
        if prev != None:
            if not diag:
                if prev != lab:
                    transitions_matrix[prev,lab] += 1
            else: 
                transitions_matrix[prev,lab] += 1
        prev = lab

    # make it a probability matrix
    for i in range(num_clusters):
        row = transitions_matrix[i,:]
        if sum(row) > 0: transitions_matrix[i,:] = row / sum(row)
        for j in range(num_clusters):
            transitions_matrix[i,j] = round(transitions_matrix[i,j],2)
  
    return set_labels,transitions_matrix