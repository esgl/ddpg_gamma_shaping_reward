import numpy as np

class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_actin_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_actin_stddev:
            self.current_stddev /= self.adoption_coefficient
        else:
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            "param_noise_stddev": self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = "AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaption_coefficient={})"
        return fmt.format(self.initial_stddev, self.desired_actin_stddev, self.adoption_coefficient)

class ActionNoise(object):
    def reset(self):
        pass

class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, *args, **kwargs):
        return np.random.normal(self.mn, self.sigma)

    def __repr__(self):
        return "NormalActionNoise(mu={}, sigma={})".format(self.mu, self.sigma)

class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    def __call__(self, *args, **kwargs):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(self.mu, self.sigma)
