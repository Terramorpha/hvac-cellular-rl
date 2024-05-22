import gymnasium as gym
import numpy as np
import sinergym
from sinergym.utils.common import get_ids
from sinergym.envs import EplusEnv
import profile
import os
import json
import graph
import rdflib

# env = EplusEnv(
#     os.getcwd() +  "/converted.json", "COL_Bogota.802220_IWEC.epw",
#     reward_kwargs={
#         "temperature_variables": [],
#         "energy_variables": [],
#         "range_comfort_winter": [],
#         "range_comfort_summer": [],
#     })
env = gym.make('Eplus-datacenter-mixed-continuous-stochastic-v1')
# env = gym.make('Eplus-demo-v1')

obs, info = env.reset()

def get_trucs(env):
    actuators = env.unwrapped.actuators
    json_file_name = env.unwrapped.building_path
    with open(json_file_name, "rb") as file:
        def ob_pairs_hook(pairs):
            return {
                k.upper(): v for (k, v) in pairs
            }
        epjson = json.load(file, object_pairs_hook=ob_pairs_hook)
    return {
        "meters": {
        },
        "actuators": {
            k: {
                "original_name": original_name,
                "contents": epjson[type.upper()][original_name.upper()],
            }
            for k, (type, value, original_name) in actuators.items()
        },
    }

rdf = graph.json_to_rdf(env.unwrapped.building_path)



# <custom_actuator_name> : (<actuator_type>,<actuator_value>,<actuator_original_name>),
for k, (type_name, _, name) in env.unwrapped.actuators.items():
    print(f"doing {k} {type_name} {name}...")
    q = """# -*- mode: sparql -*-
SELECT ?zonename
WHERE {
  {
    ?setpoint idf:cooling_setpoint_temperature_schedule_name ?actuator .
  } UNION {
    ?setpoint idf:heating_setpoint_temperature_schedule_name ?actuator .
  }
    ?setpoint ^idf:control_1_name ?zonecontrol .
    ?zonecontrol idf:zone_or_zonelist_name ?zonename
}
"""
    print(list(rdf.query(q, initBindings={"actuator": rdflib.Literal(name)})))
