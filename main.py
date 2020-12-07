import csv
import random
from math import log, e


# Used to load in the file data into the dataset
# Also returns the number of entries in the file

def load_data(file):
    d = []
    att = []
    with open(file, 'r') as f:
        r = csv.reader(f, delimiter='\t')
        i = 0
        for row in r:
            if i == 0:
                att = row
                i += 1
            else:
                d.append(row)
                i += 1
    return d, i, att


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
    if r[test_c].isdigit():
        return r[test_c] == test_val

    elif float(r[test_c]) >= float(test_val):
        return True

    else:
        return False

def check_forValue(value):
    try:
        float(value)
        return True
    except ValueError:
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
    ig = impurity - p * gini_index(left) - (1 - p) * gini_index(right)
    return ig


# Used to find the best split for data among all attributes

def split(r):
    max_ig = 0
    max_att = 0
    max_att_val = 0
    i = 0

    curr_gini = gini_index(r)
    n_att = len(attribute)

    for c in range(n_att):
        if c == 3:
            continue

        c_vals = get_column(r, c)

        while i < len(c_vals):

            # Value of the current attribute that is being tested
            curr_att_val = r[i][c]
            true, false = fork(r, c, curr_att_val)
            ig = gain(true, false, curr_gini)

            if ig > max_ig:
                max_ig = ig
                max_att = c
                max_att_val = r[i][c]
            i += 1

    return max_ig, max_att, max_att_val


# Used to recursively go through the tree in order to find the optimal attribute to split the tree with
def rec_tree(r):
    ig, att, curr_att_val = split(r)

    if ig == 0:
        return Leaf(r)

    true_rows, false_rows = fork(r, att, curr_att_val)

    true_branch = rec_tree(true_rows)
    false_branch = rec_tree(false_rows)

    return Node(att, curr_att_val, true_branch, false_branch)


# Defines the classifications of leaf

class Leaf:
    def __init__(self, rows):
        self.predictions = y_count(rows)


# Defines a split node - contains the primary attribute its value and the two child branches

class Node:
    def __init__(self, att, att_value, true_branch, false_branch):
        self.att = att
        self.att_value = att_value
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(node, spacing=""):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing + "Is " + attribute[node.att] + " > " + str(node.att_value) + " ?")

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    build_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    build_tree(node.false_branch, spacing + "  ")


def print_leaf(counts):
    total = sum(counts.values())
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


# method used to classify test data once tree is formed
def classify(r, node):
    if isinstance(node, Leaf):
        return node.predictions

    c = node.att
    att_value = node.att_value

    if compare(r, c, att_value):
        return classify(r, node.true_branch)
    else:
        return classify(r, node.false_branch)


# Main function

if __name__ == '__main__':
    filename = 'beer.txt'
    data, classes, attribute = load_data(filename)

    testing, training = split_shuffle(data, 3)

    # print(testing), print(training), print(data), print(classes)
    print(attribute)

    tree = rec_tree(training)
    build_tree(tree)

    right = 0
    wrong = 0
    for r in testing:
        # classify(row, tree)
        print("Actual: %s. Predicted: %s" % (r[3], print_leaf(classify(r, tree))))
        for key, value in classify(r, tree).items():
            if r[3] == key:
                right += 1
            else:
                wrong += 1
    print('Percentage Correctly Classified')
    print(right / (right + wrong) * 100)
    print('Percentage Incorrectly Classified')
    print(wrong / (right + wrong) * 100)
