# Static Defect Detection
ipython evaluate/bug_verifier.py

# Precondition-Fix Generation
ipython evaluate/precond_generator_immediate.py
ipython evaluate/precond_generator.py ranum all
ipython evaluate/precond_generator.py ranum weight
ipython evaluate/precond_generator.py ranum input
ipython evaluate/precond_generator.py gd all
ipython evaluate/precond_generator.py gd weight
ipython evaluate/precond_generator.py gd input
ipython evaluate/precond_generator.py ranumexpand all
ipython evaluate/precond_generator.py ranumexpand weight
ipython evaluate/precond_generator.py ranumexpand input

# Unit Test Generation
ipython evaluate/robust_inducing_inst_generator.py
ipython experiments/unittest/err_trigger.py random

# System Test (Training-Time Test) Generation
# Warning: time-consumping: 10 hrs on raw environment, 2 days on Docker
ipython evaluate/train_inst_generator.py ranum
ipython evaluate/train_inst_generator.py random
ipython evaluate/train_inst_generator.py ranum_p_random
ipython evaluate/train_inst_generator.py random_p_ranum

# Result Printing
ipython evaluate/texify/final_texify.py
ipython evaluate/texify/final_texify.py >results/results.txt # results/results.txt will contain all key tables in the paper