import csv
import random
from math import log, e


# Used to load in the file data into the dataset
# Also returns the number of entries in the file

def load_data(file):
    d = []
    with open(file, 'r') as f:
        r = csv.reader(f, delimiter='\t')
        i = 0
        for row in r:
            d.append(row)
            i += 1
    return d, i


# Used to shuffle and split the data into 2/3s training data and 1/3 testing data

def split_shuffle(ds, parts):
    random.shuffle(ds)
    p = (len(ds)) // parts
    test = ds[:p]
    train = ds[p:]
    return test, train


# Used to returns all values for a set column

def get_column(r, c):
    cols = []
    for row in r:
        cols.append(row[c])
    return cols


# Used to count the number of each label/feature (beer_style) in the rows passed to the function

def y_count(r):
    y_num = {}
    for row in r:
        y = row[3]
        if y not in y_num:
            y_num[y] = 0
        y_num[y] += 1
    return y_num


# Used to compare and test if the current row is greater than or equal to the test value
# in order to split up the data

def compare(r, test_c, test_val):
    if r[test_c].is_numeric:
        return r[test_c] == test_val

    elif float(r[test_c]) >= float(test_val):
        return True

    else:
        return False


# Splits the data into two lists for the true/false results of the compare test

def fork(r, c, test_val):
    true = []
    false = []

    for row in r:

        if compare(row, c, test_val):
            true.append(row)
        else:
            false.append(row)

    return true, false


# Used to calculate the Gini Index of the rows inputted (of beer style)

def gini_index(r):
    stylesNum = y_count(r)
    impurity = 1

    for style in stylesNum:
        style_prob = stylesNum[style] / float(len(r))
        impurity -= style_prob ** 2
    return impurity


# Used to calculate the Information gain

def gain(left, right, impurity):
    p = float(len(left)) / (len(left) + len(right))
    return impurity - p * gini_index(left) \
           - (1 - p) * gini_index(right)


# Used to find the best split for data among all attributes

def split(r):
    max_ig = 0
    max_att = 0
    max_att_val = 0
    i = 0

    curr_gini = gini_index(r)
    n_att = len(attributes)

    for c in range(n_att):
        if c == 3:
            continue

        c_vals = get_column(r, c)

        while i < len(c_vals):
            # Value of the current attribute that is being tested
            curr_att_val = r[i][c]
            true, false = fork(r, c, curr_att_val)
            gain = gain(true, false, curr_gini)

            if gain > max_ig:
                max_ig = gain
                max_att = c
                max_att_val = r[i][c]
            i += 1

    return max_ig, max_att, max_att_val


# Defines the classifications of leaf

class Leaf:
    def __init__(self, rows):
        self.predictions = y_count(rows)


# class to define a split node - contains the feature and feature value and two child branches

class Node:
    def __init__(self, feature, feature_value, true_branch, false_branch):
        self.feature = feature
        self.feature_value = feature_value
        self.true_branch = true_branch
        self.false_branch = false_branch


# Main function

if __name__ == '__main__':
    filename = 'beer.txt'
    data, classes = load_data(filename)
    attributes = ['calorific_value', 'nitrogen', 'turbidity', 'beer_style', 'alcohol', 'sugars', 'bitterness',
                  'beer_id', 'colour', 'degree_of_fermentation']

    print(attributes)

    testing, training = split_shuffle(data, 3)

    print(testing), print(training), print(data), print(classes)
