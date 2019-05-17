#!/usr/bin/env python3

import numpy as np


n = 2160
#m = 24
m = 360
training_ratio = 0.5
validation_ratio = 0.25

# Reproducible results
np.random.seed(1234567)


num_train = int(m * training_ratio)
num_validation = int(num_train +  m * validation_ratio)

#problems = 1 + np.arange(n)
problems = np.arange(n)
subsets = np.split(problems, n/m)

training_set =  []
validation_set = []
test_set = []
for subset in subsets:
    train, validation, test = np.split(np.random.permutation(subset),
                                       [num_train,num_validation])
    training_set.append(train)
    validation_set.append(validation)
    test_set.append(test)

def write(filename, input_list):
    x = np.concatenate(input_list).ravel()
    with open(filename,'w') as f:
        for i in x:
            f.write(str(i)+'\n')
    print("{} problems written to {}\n".format(len(x), filename))

write("training_set.txt", training_set)
write("validation_set.txt", validation_set)
write("test_set.txt", test_set)
