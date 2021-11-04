from bitbots_quintic_walk import PyWalk


class AbstractEngine:

    def __init__(self, namespace):
        self.namespace = namespace

    def special_reset(self, engine_state, phase, goal, reset_odometry):
        raise NotImplementedError

    def step_open_loop(self, timestep, goal):
        raise NotImplementedError

    def step(self, timestep, goal, imu_msg, joint_state_msg, left_pressure_msg, right_pressure_msg):
        raise NotImplementedError

    def get_phase(self):
        raise NotImplementedError

    def get_odom(self):
        raise NotImplementedError


class WalkEngine(PyWalk, AbstractEngine):

    def __init__(self, namespace):
        AbstractEngine.__init__(self, namespace)
        PyWalk.__init__(self, namespace)
