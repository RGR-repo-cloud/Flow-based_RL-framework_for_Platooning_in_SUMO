# Extension of Flow for Longitudinal Vehicle Platooning

Flow (https://github.com/flow-project/flow) is a framework for building RL-compatible environments for traffic microsimulators like SUMO (https://github.com/eclipse-sumo/sumo). In this repository, a simple platooning environment is added to the original version as well as functionality for logging and evaluating. It is compatible with an RL-module that can be found here: https://github.com/RGR-repo-cloud/RL-based_Longitudinal_Vehicle_Platooning_in_SUMO . 


## Setup Instructions
If you want to use this extension of Flow, you need to rename it to "flow" after cloning.

## Running Instructions
A training or evaluation run is started via the RL-module, where most of the configuration is done.
Still, some parameters must be specified in the upper section of the "platoon_exp.py" file. Note, that the environment must match the running specifications in the RL-module.
