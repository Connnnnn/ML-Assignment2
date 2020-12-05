import csv
import random
from math import log, e


# Used to load in the file data into the dataset
# Also returns the number of entries in the file
# Conor
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
# Conor
def split_shuffle(ds, parts):
    random.shuffle(ds)
    p = (len(ds)) // parts
    test = ds[:p]
    train = ds[p:]
    return test, train


# Used to returns all values for a set column
# Conor
def get_column(r, c):
    cols = []
    for row in r:
        cols.append(row[c])
    return cols


# Used to count the number of each label/feature (beer_style) in the rows passed to the function
# Conor
def y_count(r):
    y_num = {}
    for row in r:
        y = row[3]
        if y not in y_num:
            y_num[y] = 0
        y_num[y] += 1
    return y_num


# Used to calculate the Gini Index of the rows inputted (of beer style)
# Conor
def gini_index(r):
    stylesNum = y_count(r)
    impurity = 1

    for style in stylesNum:
        style_prob = stylesNum[style] / float(len(r))
        impurity -= style_prob ** 2
    return impurity


# Used to calculate the Information gain
# Conor
def gain(left, right, impurity):
    p = float(len(left)) / (len(left) + len(right))
    return impurity - p * gini_index(left) \
           - (1 - p) * gini_index(right)


# class to define leaf node reached - contains the classifications at that leaf
# Conor
class Leaf:
    def __init__(self, rows):
        self.predictions = y_count(rows)


# class to define a split node - contains the feature and feature value and two child branches
# Conor
class Split_Node:
    def __init__(self, feature, feature_value, true_branch, false_branch):
        self.feature = feature
        self.feature_value = feature_value
        self.true_branch = true_branch
        self.false_branch = false_branch


# Main function
# Conor
if __name__ == '__main__':
    filename = 'beer.txt'
    data, classes = load_data(filename)
    attributes = ['calorific_value', 'nitrogen', 'turbidity', 'beer_style', 'alcohol', 'sugars', 'bitterness',
                  'beer_id', 'colour', 'degree_of_fermentation']

    print(attributes)

    testing, training = split_shuffle(data, 3)

    print(testing), print(training), print(data), print(classes)
