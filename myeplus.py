from pyenergyplus.api import EnergyPlusAPI
import threading
import utils
import graph
import collections
from dataclasses import dataclass

api = EnergyPlusAPI()

StepResult = collections.namedtuple("StepResult", ["observation", "finished"])

DoneResult = collections.namedtuple("DoneResult", ["exit_code"])


@dataclass
class Variable:
    """To add a variable to the observation space, add

    `{
        "outdoor": Variable(("SITE OUTDOOR AIR DRYBULB TEMPERATURE","ENVIRONMENT")),
        ...
    }`

    somewhere in the observation_template object. When an observation is received,
    the value will replace the tuple like this:

    `{
        "outdoor": Variable(1.2345),
        ...
    }`

    """

    inner: any


@dataclass
class Actuator:
    """..."""

    inner: any


@dataclass
class Meter:
    """Works like variable, but inner contains only a str."""

    inner: any


@dataclass
class Function:
    """Function(f) will be replaced by f(state) at each timestep"""

    inner: any


class EnergyPlus:

    """A PyTree containing `Variable` and `Meter` leaves."""

    observation_template: any

    """ A PyTree of the same shape, but containing the associated handles"""
    observation_handles: any

    def __init__(
        self,
        building,
        weather,
        observation_template,
        actuators,
        instance="default_instance_name",
        max_steps=10_000,
    ):
        self.instance = instance
        self.obs_chan = utils.Channel()
        self.act_chan = utils.Channel()

        self.done = False

        self.building_file = building
        self.weather_file = weather
        self.observation_template = observation_template
        self.actuators = actuators
        self.max_steps = max_steps
        self.n_steps = 0

    def cb_end_timestep(self, state) -> None:
        # print("running one timestep...")
        try:

            # We replace each Variable handle by its value
            var_replaced = utils.fmap(
                lambda han: api.exchange.get_variable_value(state, han),
                self.observation_handles,
                LeafType=Variable,
            )

            # We replace each Meter handle by its value
            meter_replaced = utils.fmap(
                lambda han: api.exchange.get_meter_value(state, han),
                var_replaced,
                Meter,
            )

            actuator_replaced = utils.fmap(
                lambda han: api.exchange.get_actuator_value(state, han),
                meter_replaced,
                Actuator,
            )
            # We replace each Function(f) by f(state)
            function_replaced = utils.fmap(
                lambda fn: fn(self.state),
                actuator_replaced,
                Function,
            )
            obs = function_replaced
            # Here, we assume `variable_handles` takes the shape of a dict
            # which might not be the case.

            if not (self.n_steps < self.max_steps):
                api.runtime.stop_simulation(self.state)
                return
            self.obs_chan.put(
                StepResult(
                    observation=obs,
                    finished=False,
                )
            )
            # self.obs_chan.put(
            #     StepResult(
            #         observation=obs,
            #         finished=True,
            #     )
            # )
            # self.obs_chan.put(
            #     StepResult(
            #         observation=obs,
            #         finished=True,
            #     )
            # )
            # early return, because after receiving a StopStep signal, the
            # caller will not give other actuator values to set.
            act = self.act_chan.get()
            if type(act) != dict:
                return

            for k, v in act.items():
                api.exchange.set_actuator_value(state, self.actuator_handles[k], v)
            self.n_steps += 1
        except Exception as e:
            api.runtime.stop_simulation(self.state)
            raise e

    def cb_after_warmup(self, state):
        try:
            # print("done warming up...")

            def var_handle(var):
                han = api.exchange.get_variable_handle(state, *var)
                assert han >= 0, f"invalid variable: {var}"
                return han

            with_variable_handles = utils.fmap(
                var_handle,
                self.observation_template,
                Variable,
            )

            def meter_handle(met):
                han = api.exchange.get_meter_handle(state, met)
                # assert han >= 0, f"invalid meter: {met}"
                return han

            with_meter_handles = utils.fmap(
                meter_handle,
                with_variable_handles,
                Meter,
            )

            def actuator_handle(act):
                han = api.exchange.get_actuator_handle(state, *act)
                assert han >= 0, f"invalid actuator: {act}"
                return han

            with_actuator_handles = utils.fmap(
                actuator_handle,
                with_meter_handles,
                Actuator,
            )
            self.observation_handles = with_actuator_handles

            self.actuator_handles = {}
            for k, act in self.actuators.items():
                self.actuator_handles[k] = actuator_handle(act)

            def cb_progress(progress):
                print(f"progress: {progress/100:0.2f}")

            api.runtime.callback_progress(self.state, cb_progress)

            # api.runtime.callback_end_zone_timestep_after_zone_reporting(
            #     self.state,
            #     self.cb_end_timestep,
            # )
            api.runtime.callback_end_system_timestep_after_hvac_reporting(
                self.state, self.cb_end_timestep
            )
        except Exception as e:
            api.runtime.stop_simulation(self.state)
            raise e

    def start(self):
        state = api.state_manager.new_state()
        self.state = state

        def eplus_thread():
            """Thread running the energyplus simulation."""
            exit_code = api.runtime.run_energyplus(
                self.state,
                [
                    "-d",
                    f"output/{self.instance}",
                    "-w",
                    self.weather_file,
                    self.building_file,
                ],
            )
            print(f"energyplus exited with code {exit_code}")
            api.state_manager.delete_state(state)
            print(f"freed E+ memory")
            self.obs_chan.put(DoneResult(exit_code))
            del self.state
            self.done = True

        # Like forEach
        utils.fmap(
            lambda var: api.exchange.request_variable(self.state, *var),
            self.observation_template,
            Variable,
        )

        api.runtime.callback_after_new_environment_warmup_complete(
            self.state,
            self.cb_after_warmup,
        )

        self.thread = threading.Thread(target=eplus_thread, daemon=True)
        self.thread.start()

    def filter(self, result: StepResult | DoneResult):
        if type(result) == StepResult:
            return result
        elif type(result) == DoneResult:
            if result.exit_code != 0:
                raise Exception(
                    "energyplus exited with an error. See the eplusout.err file"
                )
            else:
                return StepResult({}, True)
        else:
            raise Exception(
                "this channel should receive only a `StepResult` or a `DoneResult`."
            )

    def get_obs(self):
        assert not self.done, "This simulation is done. Create another one."
        return self.filter(self.obs_chan.get())

    def step(self, action):
        assert not self.done, "This simulation is done. Create another one."
        self.act_chan.put(action)
        return self.get_obs()

    def get_api_endpoints(self):
        exchange_points = api.exchange.get_api_data(self.state)

        def extract(v):
            return [v.key, v.name, v.type, v.what]

        t = list(map(extract, exchange_points))
        return t
