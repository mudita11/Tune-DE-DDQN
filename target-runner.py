#!/usr/bin/python
###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import datetime
import os.path
import os
import subprocess
import sys

#! /usr/bin/env python3

exe = "python3 ../bin/train_dqn.py"

fixed_params = " "

if len(sys.argv) < 5:
    print ("\nUsage: ./target-runner.py <candidate_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
    sys.exit(1)

def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)

# Get the parameters as command line arguments.
candidate_id = sys.argv[1]
instance_id = sys.argv[2]
seed = sys.argv[3]
instance = sys.argv[4]; #print("inst1",instance)
cand_params = sys.argv[5:]

# Define the stdout and stderr files.
prefix = "c{}".format(candidate_id)
out_file = "c" + str(candidate_id) + "-" + str(instance_id) + ".stdout"
err_file = "c" + str(candidate_id) + "-" + str(instance_id) + ".stderr"
weight_file = prefix + "_dqn_ea_weights.h5f"
training_set = "training_set.txt"
# Build the command, run it and save the output to a file,
# to parse the result from it.
# 
# Stdout and stderr files have to be opened before the call().
#
# Exit with error if something went wrong in the execution.

outf = open(out_file, "w")
errf = open(err_file, "w")
command = " ".join([exe, "--trainingInstance_file", training_set, "--weight_file ", weight_file, "--instance", instance] + cand_params)
# print(command)
return_code = subprocess.call(command, shell=True,stdout = outf, stderr = errf)
outf.close()
errf.close()

if return_code != 0:
    now = datetime.datetime.now()
    print(str(now) + " error: command returned code " + str(return_code))
    sys.exit(1)

if not os.path.isfile(out_file):
    now = datetime.datetime.now()
    print(str(now) + " error: output file "+ out_file  +" not found.")
    sys.exit(1)

#cost=[line.rstrip('\n') for line in open(out_file)][-8]

cost = [line.rstrip('\n') for line in open(out_file)][-1]

# This is an example of reading a number from the output.
# It assumes that the objective value is the first number in
# the first column of the last line of the output.
# from http://stackoverflow.com/questions/4703390

# print("Cost:= ",cost)
print(cost)

sys.exit(0)
#print("End of target-runner")
