# Interpretable_RL_Research_Project

### Structure

**dagger.py** - script that runs the dagger algorithm, then saves the tree it found and plots the result. Can also detect performance drops 

**learn_expert.py** - learns a TD3 model for the given environment 

**debug.py** - Checks two policies performances (good and bad) on the training dataset, also produces visualisations of those policies (as pdf) 

**validate_dt.py** - runs given policy and gives some debug insight to it (what the reason of termination, what was the last action etc.)

### What did I discover:

The weird learning curve is probably because of the continuous actions. 
The structure of the tree that achieves max performance and the next one that has terrible performance are almost identical. 
There are only small changes in split values, which results in the correct classification (left/right) but wrong value: 
bad policy got: 0.3, good policy got: 0.73, expert got: 3

Because of too weak reaction, bad policy was already in a unhealthy (terminating) state at the next iteration. 
Which means no time to fix the mistake

How to prevent: 

- manipulate beta on significant drops 
- expand more at bad performances --> Tested, didn't improve
- maybe there is some way to add more significance to those extreme conditions? We don't really care what action we should take when the pole is upright, we should only care when it's falling 