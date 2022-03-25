# Readme: DEBARUS

DEBARUS is a tool for testing and debugging the numerical failures in DNN architectures.

## Usage

All scripts run on Python 3.6.9 + PyTorch 1.8.1 with CPU environment. GPU support only requires minor changes in the code (place `tensor.cuda()` at necessary places).
For following commands, the working directory is the root directory of this project.
All results are output to `results/` folder.

## Result Reproduction

To evaluate the DEBARUS and other baseline approaches, we create a few scripts to run all GRIST dataset cases for 10 random seeds with one seed.
You can use the following script to obtain all results. It takes roughly 2 days.

```
~/anaconda3/bin/ipython evaluate/bug_verifier.py

~/anaconda3/bin/ipython evaluate/precond_generator.py debarus all
~/anaconda3/bin/ipython evaluate/precond_generator.py debarus weight
~/anaconda3/bin/ipython evaluate/precond_generator.py debarus input
~/anaconda3/bin/ipython evaluate/precond_generator.py gd all
~/anaconda3/bin/ipython evaluate/precond_generator.py gd weight
~/anaconda3/bin/ipython evaluate/precond_generator.py gd input
~/anaconda3/bin/ipython evaluate/precond_generator.py debarusexpand all
~/anaconda3/bin/ipython evaluate/precond_generator.py debarusexpand weight
~/anaconda3/bin/ipython evaluate/precond_generator.py debarusexpand input

~/anaconda3/bin/ipython evaluate/robust_inducing_inst_generator.py
~/anaconda3/bin/ipython experiments/unittest/onestep_gd.py

~/anaconda3/bin/ipython experiments/unittest/err_trigger.py debarus
~/anaconda3/bin/ipython experiments/unittest/err_trigger.py gd
~/anaconda3/bin/ipython experiments/unittest/err_trigger.py random

~/anaconda3/bin/ipython evaluate/train_inst_generator.py debarus
~/anaconda3/bin/ipython evaluate/train_inst_generator.py random
```

Specifically, we show the individual commands for each stage task.

### 0. Static Defect Detection

DEBARUS: `python3 evaluate/bug_verifier.py`

The running results of DEBAR are from GRIST (Yan et al) paper and DEBAR (Zhang et al) repository.

### 1. Unit Test Generation

#### DEBARUS
    
1. `python3 evaluate/robust_inducing_inst_generator.py` - generate failure-inducing intervals
2. `python3 experiments/unittest/err_trigger.py debarus` - generate 1000 distinct unit tests from these intervals

#### Gradient Descent
    
1. `python3 experiments/unittest/onestep_gd.py` - generate failure-inducing intervals
2. `python3 experiments/unittest/err_trigger.py gd` - generate 1000 distinct unit tests from these intervals
    
##### Random
    
`python3 experiments/unittest/err_trigger.py random` - generate 1000 distinct unit tests from these intervals

### 2. System Test Generation

System test generation relies on the generated unit tests, so please run unit test generation first

#### DEBARUS

`python3 evaluate/train_inst_generator.py debarus`

#### Random

`python3 evaluate/train_inst_generator.py random`

### 3. Precondition Generation

#### a. Precondition on Weight + Input Nodes

##### DEBARUS

`python3 evaluate/precond_generator.py debarus all`

##### DEBARUS Expand

`python3 evaluate/precond_generator.py debarusexpand all`

##### Gradient Descent

`python3 evaluate/precond_generator.py gd all`

#### b. Precondition on Weight Nodes

##### DEBARUS

`python3 evaluate/precond_generator.py debarus weight`

##### DEBARUS Expand

`python3 evaluate/precond_generator.py debarusexpand weight`

##### Gradient Descent

`python3 evaluate/precond_generator.py gd weight`

#### c. Precondition on Input Nodes

##### DEBARUS

`python3 evaluate/precond_generator.py debarus input`

##### DEBARUS Expand

`python3 evaluate/precond_generator.py debarusexpand input`

##### Gradient Descent

`python3 evaluate/precond_generator.py gd input`

