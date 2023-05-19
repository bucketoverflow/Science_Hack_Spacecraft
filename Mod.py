import numpy as np
from gymnasium.vector.utils import spaces

from battery import Battery
from orbit_propagator import Orbit_Propagator
from propellant_tank import Propellant_Tank
from renderer import Renderer
from spacecraft import Spacecraft
from thruster import Thruster


class Data_Storage:

    def __init__(self, render_mode=None):

        self.propellant_tank = Propellant_Tank(self.INITIAL_PROPELLANT_MASS)
        self.thruster = Thruster()
        self.orbit_propagator = Orbit_Propagator(self.COM_ORBIT, self.OBSERVER_ORBIT)

        self.battery = Battery(self.INITIAL_POWER)  ##############
        self.data_storage = Data_Storage(self.INITIAL_DATA)  ################

        self.state = None

        self.action_space = spaces.Discrete(4)

        self._ai_action_to_spacecraft_action = {
            0: np.array([-1, 0]),  # Thruster PROGRADE, Communications OFF
            1: np.array([1, 0]),  # Thruster RETROGRADE, Communications OFF
            2: np.array([0, 0]),  # Thruster OFF, Communications OFF
            3: np.array([0, 1])  # Thruster OFF, Communications ON
        }

        self.mass_spacecraft = self.MASS_SPACECRAFT_INITIAL

        self.thrust_vector = []
        self.position_vector_x_com = []
        self.position_vector_y_com = []
        self.position_vector_x_obs = []
        self.position_vector_y_obs = []
        self.time_vector = [0]

        self.steps_to_truncate = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.renderer = Renderer()
            self.orbit_propagator.reset_orbits()
            self.com_orbit_points = self.orbit_propagator.get_orbit_points(self.orbit_propagator.orb_com)
            self.obs_orbit_points = self.orbit_propagator.get_orbit_points(self.orbit_propagator.orb_obs)

    def reset(self):

        """ Resets the simulation. Sets the orbit and mass stored propellant tank to its initial values 

            Generally run at the beginning of a new episode
        """

        self.propellant_tank.reset_propellant_tank(self.INITIAL_PROPELLANT_MASS)
        self.battery.reset_battery(self.INITIAL_POWER)  ##############################
        self.data_storage.reset_data_storage(self.INITIAL_DATA_STORAGE)  ###############################
        self.orbit_propagator.reset_orbits()

        self.state = self.get_state()

        self.mass_spacecraft = self.MASS_SPACECRAFT_INITIAL

        self.steps_to_truncate = 0

        self.thrust_vector = []
        self.position_vector_x_com = [self.orbit_propagator.orb_com.r[0].value]
        self.position_vector_y_com = [self.orbit_propagator.orb_com.r[1].value]
        self.position_vector_x_obs = [self.orbit_propagator.orb_obs.r[0].value]
        self.position_vector_y_obs = [self.orbit_propagator.orb_obs.r[1].value]
        self.time_vector = [0]

        return self.state, {}

    def step(self, action):

        """ Takes one time step of the entire simulation 

            :param action: Integer (From 0 to action_space - 1) that translates to spacecraft action (Example: 0 = Thruster ON/Electrolyser OFF)
        """

        s = self.state
        assert s is not None, "Call reset before using Spacecraft object."

        self.thruster_action = self._ai_action_to_spacecraft_action[action][0]
        communication_on_off = self._ai_action_to_spacecraft_action[action][1]

        # Propellant Tank
        propellant_used = abs(self.thruster_action) * 0.01  # [Kg] ; 0.01 Kg used per Thrust manoeuvre
        self.propellant_tank.remove_mass(propellant_used)
        # print(self.propellant_tank.current_mass)

        # Computing burn duration
        t_burn = 5 * abs(self.thruster_action)  # self.compute_burn_duration() * self.thruster_on_off

        # Thruster
        thrust = self.thruster_action * self.thruster.get_thrust(propellant_used, t_burn)  # self.MASS_FLOW, t_burn)
        acceleration = thrust / self.mass_spacecraft
        # print('Thrust: ', thrust)
        if self.propellant_tank.current_mass <= 0:
            acceleration = 0
        # Orbit_Propagator
        self.orbit_propagator.acceleration_com = acceleration
        self.orbit_propagator.propagate_orbits(self.dt, t_burn)

        self.mass_spacecraft = self.mass_spacecraft - propellant_used

        self.ignition_on = thrust > 0

        # Battery
        power_used = self.communication_on_off * 0.01 * distance  ########################
        self.battery.remove_power(power_used)

        # Data_Storage
        data_pack_dimension = 1
        data_sent = self.communication_on_off * data_pack_dimension  ########################
        self.data_storage.remove_data(data_sent)

        # 
        self.state = self.get_state()
        reward = self.get_reward()
        terminated = self._terminal()
        truncated = self._truncated()

        self.thrust_vector.append(getattr(thrust, "tolist", lambda: value)())
        self.position_vector_x_com.append(self.orbit_propagator.orb_com.r[0].value)
        self.position_vector_y_com.append(self.orbit_propagator.orb_com.r[1].value)
        self.position_vector_x_obs.append(self.orbit_propagator.orb_obs.r[0].value)
        self.position_vector_y_obs.append(self.orbit_propagator.orb_obs.r[1].value)
        self.time_vector.append(self.time_vector[-1] + self.dt)

        # if self.ignition_on:
        #     print('Thrust', self.steps_to_truncate, self.thruster.get_isp(t_burn))
        # print(self.orbit_propagator.orb.a)

        if self.render_mode == "human":
            self.render()

        self.steps_to_truncate += 1

        return self.state, reward, terminated, truncated, {}

    def get_state(self):
        return self._get_state_new()

    def get_reward(self):
        return self._get_reward3()

    def _get_state_new(self):
        r_com = self.orbit_propagator.orb_com.r.value / self.COM_ORBIT['a']
        # r_obs = self.orbit_propagator.orbit_compare() / self.INITIAL_ORBIT['a']

        h_com = self.orbit_propagator.orb_com.a.value - self.R_NEPTUNE
        h_obs = self.orbit_propagator.orb_obs.a.value - self.R_NEPTUNE
        theta_com = self.orbit_propagator.orb_com.nu.value

        ecc_com = self.orbit_propagator.orb_com.ecc.value
        ecc_obs = self.orbit_propagator.orb_obs.ecc.value

        argp_com = self.orbit_propagator.orb_com.argp.value
        argp_obs = self.orbit_propagator.orb_obs.argp.value

        # diff = r_current[0:2] - r_final
        theta_obs = self.orbit_propagator.orb_obs.nu.value

        propellant_mass = self.propellant_tank.current_mass

        # gas_tank_pressure = self.hydrogen_gas_tank.pressure / 50
        # print(gas_tank_pressure)

        # return np.array([diff[0], diff[1], h_current / h_final, ecc_current, cos(argp_current), sin(argp_current), cos(theta), sin(theta), gas_tank_pressure])
        return np.array(
            [r_com[0], r_com[1], h_com / h_obs, ecc_com, np.cos(argp_com), np.sin(argp_com), np.cos(theta_com), np.sin(theta_obs),
             propellant_mass])

    # def _get_reward(self):

    #     h_current = self.orbit_propagator.orb.a.value - self.R_NEPTUNE
    #     h_final   = self.orbit_propagator.final_orb.a.value - self.R_NEPTUNE

    #     # reward = 0

    #     # if (h_current >= 0.95 * h_final) and (h_current <= 1.05 * h_final):
    #     #     reward = 1

    #     reward = 1 - (abs(h_final - h_current) / 250)

    #     return reward

    # def _get_reward2(self):

    #     h_current = self.orbit_propagator.orb.a.value - self.R_NEPTUNE
    #     h_final   = self.orbit_propagator.final_orb.a.value - self.R_NEPTUNE
    #     scale_h = max([h_current, h_final])

    #     theta     = self.orbit_propagator.orb.nu.value

    #     ecc_current = self.orbit_propagator.orb.ecc.value
    #     ecc_final   = self.orbit_propagator.final_orb.ecc.value
    #     scale_ecc = max([ecc_current, ecc_final])

    #     argp_current = self.orbit_propagator.orb.argp.value
    #     argp_final   = self.orbit_propagator.final_orb.argp.value
    #     scale_argp = max([argp_current, argp_final])

    #     reward_h = 1 - (abs(h_final - h_current) / 250)
    #     reward_ecc = 1 - (abs(ecc_final - ecc_current) / 0.0223)
    #     # reward_argp = 1 - abs(cos((argp_final - argp_current)))
    #     reward_argp = 0
    #     if h_current < 1.2 * h_final:
    #         reward_argp = cos(argp_final - argp_current)
    #         if reward_argp < 0:
    #             reward_argp = 0
    #     # reward_argp = 0

    #     reward = reward_h + reward_ecc + reward_argp

    #     return reward

    def _get_reward3(self):

        # distance = self.orbit_propagator.compute_distance_to_final_orbit()
        # print(distance)
        # scaling_factor = self.INITIAL_ORBIT['a'] - self.FINAL_ORBIT['a']
        # print(scaling_factor)

        # reward = 0
        # if distance < scaling_factor * 0.05:
        #     reward = 1

        reward = 1  # - distance/scaling_factor
        # print(reward)

        # print(reward)
        return reward

    def _terminal(self):

        # h_current = self.orbit_propagator.orb.a.value - self.R_NEPTUNE
        # h_final   = self.orbit_propagator.final_orb.a.value - self.R_NEPTUNE

        terminal = False

        # if h_current < 0.80 * h_final:
        #     terminal = True
        # if self.propellant_tank.current_mass <= 0:
        #     terminal = True

        return terminal

    def _truncated(self):

        truncated = False

        if self.steps_to_truncate >= self.MAX_STEPS:
            truncated = True

        return truncated

    def render(self):
        r = self.orbit_propagator.orbit_compare()
        l = len(self.orbit_propagator.positions_com)

        # th = self.ignition_on

        if len(self.orbit_propagator.positions_com) < 101:
            self.renderer.render(self.com_orbit_points, self.obs_orbit_points, self.orbit_propagator.positions_com,
                                 self.orbit_propagator.positions_obs, r)
        else:
            self.renderer.render(self.com_orbit_points, self.obs_orbit_points,
                                 self.orbit_propagator.positions_com[l - 100:l],
                                 self.orbit_propagator.positions_obs[l - 100:l], r)


if __name__ == "__main__":
    from itertools import count

    env = Spacecraft(render_mode="human")
    # print(env.action_space.n)
    env.reset()
    # a,b,c,d = env.step(3)
    # print(a)
    # print(b)
    for t in count():
        observation, reward, terminated, truncated, _ = env.step(2)
        done = terminated or truncated
        if done:
            break
