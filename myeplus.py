from pyenergyplus.api import EnergyPlusAPI
import threading
import utils
import graph
from functor import Leaf, fmap

api = EnergyPlusAPI()


class Env:
    def __init__(self, building, weather, variables, actuators):
        self.state = api.state_manager.new_state()

        self.obs_chan = utils.Queue()
        self.act_chan = utils.Queue()

        self.building_file = building
        self.weather_file = weather
        self.variables = variables
        self.actuators = actuators

        def eplus_thread():
            """Thread running the energyplus simulation."""
            exit_code = api.runtime.run_energyplus(
                self.state, ["-w", weather, building]
            )

        def cb_end_timestep(state):
            # obs = utils.AttrDict()

            obs = fmap(
                lambda han: api.exchange.get_variable_value(state, han),
                self.variable_handlers,
            )
            obs["time_elapsed"] = Leaf(api.exchange.current_sim_time(state))

            # We also read every actuator
            # for k, i in self.actuator_handlers.items():
            #     obs[k] = api.exchange.get_actuator_value(state, i)

            self.obs_chan.put(obs)
            act = self.act_chan.get()
            if type(act) != dict:
                return
            for k, v in act.items():
                api.exchange.set_actuator_value(state, self.actuator_handlers[k], v)

        fmap(
            lambda var: api.exchange.request_variable(self.state, *var),
            self.variables,
        )

        def cb_after_warmup(state):
            api.runtime.callback_end_zone_timestep_after_zone_reporting(
                self.state,
                cb_end_timestep,
            )

            self.variable_handlers = fmap(
                lambda var: api.exchange.get_variable_handle(state, *var),
                self.variables,
            )

            self.actuator_handlers = {}
            for k, act in self.actuators.items():
                self.actuator_handlers[k] = api.exchange.get_actuator_handle(
                    state, *act
                )

        api.runtime.callback_after_new_environment_warmup_complete(
            self.state,
            cb_after_warmup,
        )

        self.thread = threading.Thread(target=eplus_thread)

    def reset(self):
        self.thread.start()
        return self.obs_chan.get()

    def step(self, action):
        self.act_chan.put(action)
        return self.obs_chan.get()
