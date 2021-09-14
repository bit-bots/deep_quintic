## DeepQuintic: Learning Omnidirectional Walking

This repository contains the source code to train omnidirectional walk policies and run them with ROS.

### Usage

#### Training

Use the `train.py` from rl-baselines3-zoo. A lot of options can be set using the `--env-kwargs` argument, e.g. if Cartesian or joint space should be used. Have a look at the `env.py` to see which arguments can be provided.

#### Running

A learned policy can be either executed with the `enjoy.py` script from rl-baseline3-zoo to run it in PyBullet or by starting the `ros_runner.py` to launch it with a ROS stack. 
 

### Dependencies

The code currently requires multiple dependencies on specific branches and forks:

- wolfgang_robot (https://github.com/bit-bots/wolfgang_robot/tree/walk_optim)
- bitbots_quintic_walk (https://github.com/bit-bots/bitbots_motion/tree/walk_cartesian_python)
- parallel_parameter_optimization (https://github.com/bit-bots/parallel_parameter_search/tree/feature/new_walk_objectives)
- stable-baselines3 (https://github.com/SammyRamone/stable-baselines3)
- rl-baselines3-zoo (https://github.com/SammyRamone/rl-baselines3-zoo)

Additionally, some standard libraries are required:
- ROS melodic
- PyTorch
- Optuna
- BioIK (https://github.com/TAMS-Group/bio_ik)

Hint: PlotJuggler (https://github.com/facontidavide/PlotJuggler) is helpful vor visualizing states, actions, and reference motions.

### Todos
- Refactoring
- Cleaning up dependencies
- More documentation