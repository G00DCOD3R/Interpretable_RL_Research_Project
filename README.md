# Interpretable_RL_Research_Project

### Structure

**dagger.py** - script that runs the dagger algorithm, then saves the tree it found and plots the result. Can also detect performance drops 

**learn_expert.py** - learns a TD3 model for the given environment 

**debug.py** - Checks two policies' performances (good and bad) on the training dataset, also produces visualisations of those policies (as pdf) 

**validate_dt.py** - runs given policy and gives some debug insight to it (what the reason of termination, what was the last action etc.)

**run_experiments.py** - runs multiple experiments in parallel 

**results/** - folder containing results for the three environments, trained decision policies, and validation scores on all dagger iterations.

### Paper 
Research paper for this research can be found at [...]

