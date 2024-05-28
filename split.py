import rdflib
import graph
from dataclasses import dataclass


@dataclass
class FeaturePartition:
    """Used to separates component of the action space (and observation space)
    into bins corresponding to each zone.

    """

    @dataclass
    class Zone:
        """Contains information about a zone's associated actuators and
        variables."""

        actuator_variables: list[str]
        variables: list[str]

        def __init__(self):
            self.actuator_variables = []
            self.variables = []

    zones: dict[str, Zone]

    observation_variables: list[str]

    rdf: rdflib.Graph

    def __init__(self, env):
        self.zones = {}
        self.observation_variables = env.unwrapped.observation_variables
        self.rdf = graph.json_to_rdf(env.unwrapped.building_path)

        # First, we attach zone of each actuator t
        for k, (type_name, _, name) in env.unwrapped.actuators.items():
            q = """# -*- mode: sparql -*-
SELECT
# ?setpoint_a
# ?control
?zone
WHERE {
  FILTER( LCASE(?actuator) = LCASE(?actuator_real))
  ?actuator_real ^idf:setpoint_temperature_schedule_name ?setpoint_a .
  FILTER( LCASE(?setpoint_a) = LCASE(?setpoint_b))
  ?setpoint_b ^idf:control_1_name|^idf:control_2_name|^idf:control_3_name ?control .
  ?control idf:zone_or_zonelist_name ?zone .
}"""
            actuator_zones = set(
                res.value
                for (res,) in self.rdf.query(
                    q, initBindings={"actuator": rdflib.Literal(name)}
                )
            )
            for z in actuator_zones:
                self.add_actuator(z, k)

        for k, (_, zone) in env.unwrapped.variables.items():
            self.add_variable(zone, k)

    def add_actuator(self, zone, actuator):
        """Associate an actuator to a zone. The actuator value will be read from
        the specified zone when actuator values are collected into a vector.

        """
        if zone not in self.zones:
            self.zones[zone] = self.Zone()
        self.zones[zone].actuator_variables.append(actuator)

    def add_variable(self, zone, variable):
        """Associate a variable to a zone. The variable value will be written to
        the specified zone when observation values are received from a vector.

        """
        if zone not in self.zones:
            self.zones[zone] = self.Zone()

        self.zones[zone].variables.append(variable)

    def split_observation(self, vect):
        """Return a dictionnary that associates to each zones the values of the
        associated variables.

        """
        obs = {}
        for i, k in enumerate(self.observation_variables):
            obs[k] = vect[i]

        out = {
            zone: {
                # "actuator_variables": [
                #     obs[k] for k in self.zones[zone]["actuator_variables"]
                # ],
                "variables": [obs[k] for k in self.zones[zone].variables],
            }
            for zone in self.zones
        }
        return out

    def join_action(self, thing):
        """Return a vector composed of the output of each zone."""
        pass

    def zones_graph(self) -> list[tuple[str, str]]:
        """Compute the edge list of zones graph."""

        g = graph.rdf_to_adjacency(self.rdf)
        return list(
            (
                (a.value, b.value)
                for (a, b) in g.query("SELECT ?a ?c WHERE { ?a ?b ?c . }")
            )
        )
