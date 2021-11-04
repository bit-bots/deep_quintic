# code is python version of
# https:#github.com/ccny-ros-pkg/imu_tools/blob/indigo/imu_complementary_filter/src/complementary_filter.cpp
import math


class ComplementaryFilter:
    def __init__(self):
        self.kGravity = 9.81
        self.gamma_ = 0.01
        # Bias estimation steady state thresholds
        self.kAngularVelocityThreshold = 0.2
        self.kAccelerationThreshold = 0.1
        self.kDeltaAngularVelocityThreshold = 0.01
        self.gain_acc_ = 0.01
        self.bias_alpha_ = 0.05
        self.do_bias_estimation_ = False
        self.do_adaptive_gain_ = False
        self.initialized_ = False
        self.steady_state_ = False

        self.q0_ = 1
        self.q1_ = 0
        self.q2_ = 0
        self.q3_ = 0
        self.wx_prev_ = 0
        self.wy_prev_ = 0
        self.wz_prev_ = 0
        self.wx_bias_ = 0
        self.wy_bias_ = 0
        self.wz_bias_ = 0

    def reset(self, quat):
        # if we know the correct orientation we can set the filter directly to these values so that it does not need
        # a couple of seconds to restore itself after a reset
        self.initialized_ = True
        self.q0_ = quat[0]
        self.q1_ = quat[1]
        self.q2_ = quat[2]
        self.q3_ = quat[3]
        self.wx_prev_ = 0
        self.wy_prev_ = 0
        self.wz_prev_ = 0
        self.wx_bias_ = 0
        self.wy_bias_ = 0
        self.wz_bias_ = 0

    def setDoBiasEstimation(self, do_bias_estimation):
        self.do_bias_estimation_ = do_bias_estimation

    def getDoBiasEstimation(self):
        return self.do_bias_estimation_

    def setDoAdaptiveGain(self, do_adaptive_gain):
        self.do_adaptive_gain_ = do_adaptive_gain

    def getDoAdaptiveGain(self):
        return self.do_adaptive_gain_

    def setGainAcc(self, gain):
        if 0 <= gain <= 1.0:
            self.gain_acc_ = gain
            return True
        else:
            return False

    def getGainAcc(self):
        return self.gain_acc_

    def getSteadyState(self):
        return self.steady_state_

    def setBiasAlpha(self, bias_alpha):
        if 0 <= bias_alpha <= 1.0:
            self.bias_alpha_ = bias_alpha
            return True
        else:
            return False

    def getBiasAlpha(self):
        return self.bias_alpha_

    def setOrientation(self, q0, q1, q2, q3):
        # Set the state to inverse (state is fixed wrt body).
        self.q0_, self.q1_, self.q2_, self.q3_ = self.invertQuaternion(q0, q1, q2, q3)

    def getAngularVelocityBiasX(self):
        return self.wx_bias_

    def getAngularVelocityBiasY(self):
        return self.wy_bias_

    def getAngularVelocityBiasZ(self):
        return self.wz_bias_

    def update(self, ax, ay, az, wx, wy, wz, dt):
        if not self.initialized_:
            # First time - ignore prediction:
            self.q0_, self.q1_, self.q2_, self.q3_ = self.getMeasurement(ax, ay, az)
            self.initialized_ = True
            return
        # Bias estimation.
        if self.do_bias_estimation_:
            self.updateBiases(ax, ay, az, wx, wy, wz)

        # Prediction.
        q0_pred, q1_pred, q2_pred, q3_pred = self.getPrediction(wx, wy, wz, dt)

        # Correction (from acc):
        # q_ = q_pred * [(1-gain) * qI + gain * dq_acc]
        # where qI = identity quaternion
        dq0_acc, dq1_acc, dq2_acc, dq3_acc = self.getAccCorrection(ax, ay, az, q0_pred, q1_pred, q2_pred, q3_pred)

        if self.do_adaptive_gain_:
            gain = self.getAdaptiveGain(self.gain_acc_, ax, ay, az)
        else:
            gain = self.gain_acc_

        dq0_acc, dq1_acc, dq2_acc, dq3_acc = self.scaleQuaternion(gain, dq0_acc, dq1_acc, dq2_acc, dq3_acc)

        self.q0_, self.q1_, self.q2_, self.q3_ = self.quaternionMultiplication(q0_pred, q1_pred, q2_pred, q3_pred,
                                                                               dq0_acc, dq1_acc, dq2_acc, dq3_acc)

        self.q0_, self.q1_, self.q2_, self.q3_ = self.normalizeQuaternion(self.q0_, self.q1_, self.q2_, self.q3_)

    def checkState(self, ax, ay, az, wx, wy, wz):
        acc_magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        if abs(acc_magnitude - self.kGravity) > self.kAccelerationThreshold:
            return False

        if abs(wx - self.wx_prev_) > self.kDeltaAngularVelocityThreshold or abs(
                wy - self.wy_prev_) > self.kDeltaAngularVelocityThreshold or abs(
            wz - self.wz_prev_) > self.kDeltaAngularVelocityThreshold:
            return False

        if abs(wx - self.wx_bias_) > self.kAngularVelocityThreshold or abs(
                wy - self.wy_bias_) > self.kAngularVelocityThreshold or abs(
            wz - self.wz_bias_) > self.kAngularVelocityThreshold:
            return False

        return True

    def updateBiases(self, ax, ay, az, wx, wy, wz):
        self.steady_state_ = self.checkState(ax, ay, az, wx, wy, wz)

        if self.steady_state_:
            self.wx_bias_ += self.bias_alpha_ * (wx - self.wx_bias_)
            self.wy_bias_ += self.bias_alpha_ * (wy - self.wy_bias_)
            self.wz_bias_ += self.bias_alpha_ * (wz - self.wz_bias_)

        self.wx_prev_ = wx
        self.wy_prev_ = wy
        self.wz_prev_ = wz

    def getPrediction(self, wx, wy, wz, dt):
        wx_unb = wx - self.wx_bias_
        wy_unb = wy - self.wy_bias_
        wz_unb = wz - self.wz_bias_

        q0_pred = self.q0_ + 0.5 * dt * (wx_unb * self.q1_ + wy_unb * self.q2_ + wz_unb * self.q3_)
        q1_pred = self.q1_ + 0.5 * dt * (-wx_unb * self.q0_ - wy_unb * self.q3_ + wz_unb * self.q2_)
        q2_pred = self.q2_ + 0.5 * dt * (wx_unb * self.q3_ - wy_unb * self.q0_ - wz_unb * self.q1_)
        q3_pred = self.q3_ + 0.5 * dt * (-wx_unb * self.q2_ + wy_unb * self.q1_ - wz_unb * self.q0_)

        return self.normalizeQuaternion(q0_pred, q1_pred, q2_pred, q3_pred)

    def getMeasurement(self, ax, ay, az):
        # q_acc is the quaternion obtained from the acceleration vector representing
        # the orientation of the Global frame wrt the Local frame with arbitrary yaw
        # (intermediary frame). q3_acc is defined as 0.

        # Normalize acceleration vector.
        ax, ay, az = self.normalizeVector(ax, ay, az)

        if az >= 0:
            q0_meas = math.sqrt((az + 1) * 0.5)
            q1_meas = -ay / (2.0 * q0_meas)
            q2_meas = ax / (2.0 * q0_meas)
            q3_meas = 0
        else:
            X = math.sqrt((1 - az) * 0.5)
            q0_meas = -ay / (2.0 * X)
            q1_meas = X
            q2_meas = 0
            q3_meas = ax / (2.0 * X)
        return q0_meas, q1_meas, q2_meas, q3_meas

    def getAccCorrection(self, ax, ay, az, p0, p1, p2, p3):
        # Normalize acceleration vector.
        ax, ay, az = self.normalizeVector(ax, ay, az)

        # Acceleration reading rotated into the world frame by the inverse predicted
        # quaternion (predicted gravity):
        gx, gy, gz = self.rotateVectorByQuaternion(ax, ay, az, p0, -p1, -p2, -p3)

        # Delta quaternion that rotates the predicted gravity into the real gravity:
        dq0 = math.sqrt((gz + 1) * 0.5)
        dq1 = -gy / (2.0 * dq0)
        dq2 = gx / (2.0 * dq0)
        dq3 = 0.0
        return dq0, dq1, dq2, dq3

    def getOrientation(self):
        # Return the inverse of the state (state is fixed wrt body).
        return self.invertQuaternion(self.q0_, self.q1_, self.q2_, self.q3_)

    def getAdaptiveGain(self, alpha, ax, ay, az):
        a_mag = math.sqrt(ax * ax + ay * ay + az * az)
        error = abs(a_mag - self.kGravity) / self.kGravity
        error1 = 0.1
        error2 = 0.2
        m = 1.0 / (error1 - error2)
        b = 1.0 - m * error1
        if error < error1:
            factor = 1.0
        elif error < error2:
            factor = m * error + b
        else:
            factor = 0.0
        # printf("FACTOR: %f \n", factor)
        return factor * alpha

    def normalizeVector(self, x, y, z):
        norm = math.sqrt(x * x + y * y + z * z)
        x /= norm
        y /= norm
        z /= norm
        return x, y, z

    def normalizeQuaternion(self, q0, q1, q2, q3):
        norm = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        q0 /= norm
        q1 /= norm
        q2 /= norm
        q3 /= norm
        return q0, q1, q2, q3

    def invertQuaternion(self, q0, q1, q2, q3):
        # Assumes quaternion is normalized.
        q0_inv = q0
        q1_inv = -q1
        q2_inv = -q2
        q3_inv = -q3
        return q0_inv, q1_inv, q2_inv, q3_inv

    def scaleQuaternion(self, gain, dq0, dq1, dq2, dq3):
        if dq0 < 0.0:  # 0.9
            # Slerp (Spherical linear interpolation):
            angle = math.acos(dq0)
            A = math.sin(angle * (1.0 - gain)) / math.sin(angle)
            B = math.sin(angle * gain) / math.sin(angle)
            dq0 = A + B * dq0
            dq1 = B * dq1
            dq2 = B * dq2
            dq3 = B * dq3
        else:
            # Lerp (Linear interpolation):
            dq0 = (1.0 - gain) + gain * dq0
            dq1 = gain * dq1
            dq2 = gain * dq2
            dq3 = gain * dq3
        return self.normalizeQuaternion(dq0, dq1, dq2, dq3)

    def quaternionMultiplication(self, p0, p1, p2, p3, q0, q1, q2, q3):
        # r = p q
        r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
        r1 = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2
        r2 = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1
        r3 = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0
        return r0, r1, r2, r3

    def rotateVectorByQuaternion(self, x, y, z, q0, q1, q2, q3):
        vx = (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * x + 2 * (q1 * q2 - q0 * q3) * y + 2 * (q1 * q3 + q0 * q2) * z
        vy = 2 * (q1 * q2 + q0 * q3) * x + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * y + 2 * (q2 * q3 - q0 * q1) * z
        vz = 2 * (q1 * q3 - q0 * q2) * x + 2 * (q2 * q3 + q0 * q1) * y + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * z
        return vx, vy, vz
