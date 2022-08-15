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

### Using Different Robot Models

TODO add links to existing webots models

If you want to use a different robot model for training in Webots you will need to perform the following steps.
1. Create URDF and MoveIt configuration as it described [here](https://github.com/bit-bots/humanoid_robots_ros2/blob/master/README.md)
2. Optimize the quintic_walk parameters as described [here](https://bit-bots.github.io/quintic_walk/)
3. Adapt the .proto model file of the robot
    1. Add following fields to the header of the proto
        <pre><code>field  SFBool    enableBoundingObject TRUE
       field  SFBool    enablePhysics        TRUE          
       field  MFNode    externalCamera       NULL
       </code></pre>
    2. Make bounding objects optional by encapsulating every bounding object in the .proto like this:
        <pre><code>
        %{if fields.enableBoundingObject.value then}%
        boundingObject Box{...}
        %{ end }%
        </code></pre>
    3. Make physics optional by encapsulating every physics node in the .proto like this:
        <pre><code>
        %{if fields.enablePhysics.value then}%
        physics Physics {...}
        %{ end }%
        </code></pre>
    4. Add an optional node which can be used as an external camera by adding the following as a children to your base node:
        <pre><code>
        Group {
            children IS externalCamera
        }
        </code></pre>
    5. Name all joints by adding "DEF NAMEJoint" like in the following example:
        <pre><code>
        DEF NeckYawJoint HingeJoint{...}
        </code></pre>
4. Copy another ROBOT_NAME-optimization.proto and adapt it (this is necessary to avoid some weird webots bugs and maybe not necessary for all robot types. Propably something related to the HingeJointsWithBacklash)
    1. Change the name of the file to the correct robot name
    2. Change the name of the robot in the .proto    
5. Add a world file for the robot.
    1. Copy the deep_quintic_ROBOTNAME.wbt file of another robot into the worlds folder of your robot package
    2. Replace the robot name in the file name
    3. Replace the name of the robot proto for the learnbot and refbot in the .wbt file
6. Define necessary informations in robot.py
    1. Add a new robot in the __init__() of the Robot class, similar to the existing ones.
    2. Define the initial state positions. Here the arms are especially important as they are not set to the reference.
    3. Define which joints belong to the legs of the robot and which belong to the head
    4. You may need to define an additional height offset for resetting your robot, based on where your base_link is. Just see if the robot is placed at the correct height during resets and adapt this value accordingly.
    5. Define the Cartesian limits for the action space. Not all poses in this space need to be solvable, as approximation will be used otherwise. Still, at least half of the poses should be solvable as the policy may otherwise run into to many IK issues during training. If the number is too high, you may restrict your action space too much. You can use the find_cartesian_limits.py script to check how many poses are solvable. Not all poses in the action space need to be solvable. In our experience, having around 2/3 of the poses solvable is a good compromise between having a solvable action space and not restricting the action space too much. You can also run the reference motion as action and see if it stays in the action bounds of [-1,1]
    6. Define the command velocity limits. You can use the test_solvable_speeds.py script in the bitbots_quinitic_walk package to find these. This only defines the area from which the command values are randomly sampled during training. The sampled velocity is then tested if it is kinematically solvable and only used if this is the case. Otherwise it is resampled until a valid velocity is found. Therefore, it is not necessary to limit this space too much.
7. Verify that everthing is correct
    1. Start with using the reference action as action
    2. Start PlotJuggler and visualize state entries and rewards. Check if all values are correct. Especially check if IMU is correctly oriented.

### Todos
- Refactoring
- Cleaning up dependencies
- More documentation