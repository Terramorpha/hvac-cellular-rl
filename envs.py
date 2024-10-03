import graph
import os
from collections import namedtuple
import myeplus

EnvConfig = namedtuple(
    "EnvConfig",
    [
        "building_file",
        "weather_file",
        "observation_template",
        "actuators",
    ],
)


def auto_add_actuators(rdf, actuators):
    """Add all actuators listed in the graph. This is probably not what you
    want, since actuators that are not heating/cooling setpoints will be added
    too."""
    for name in graph.rdf_schedules(rdf):
        # for name in zones_with_cooling
        actuators[name] = ("Schedule:Compact", "Schedule Value", name)


def auto_add_actuators_observation(rdf, variables):
    act = {}
    for name in graph.rdf_schedules(rdf):
        # for name in zones_with_cooling
        act[name] = myeplus.Actuator(
            (
                "Schedule:Compact",
                "Schedule Value",
                name,
            )
        )
    variables["actuators"] = act


def auto_add_temperature_variables(rdf, variables):
    """Add a "ZONE AIR TEMPERATURE" for each zone in the graph."""
    temps = {}
    temps["environment"] = myeplus.Variable(
        (
            "SITE OUTDOOR AIR DRYBULB TEMPERATURE",
            "ENVIRONMENT",
        )
    )
    for z in graph.rdf_zones(rdf):
        temps[z] = myeplus.Variable(("ZONE AIR TEMPERATURE", z))
        variables["temperature"] = temps


def auto_add_setpoint_variables(rdf, variables):
    setpoints = {}
    variables["setpoints"] = setpoints

    heating = {}
    setpoints["heating"] = heating

    cooling = {}
    setpoints["cooling"] = cooling

    for z in graph.rdf_zones(rdf):
        heating[z] = myeplus.Variable(
            ("Zone Thermostat Heating Setpoint Temperature", z)
        )
        cooling[z] = myeplus.Variable(
            ("Zone Thermostat Cooling Setpoint Temperature", z)
        )


def auto_add_comfort_variables(rdf, variables):
    if "comfort" not in variables:
        variables["comfort"] = {}

    comfort = variables["comfort"]
    for z in graph.rdf_zones(rdf):
        comfort[z + "_comfort"] = myeplus.Variable(
            ("Zone Thermal Comfort Pierce Model Thermal Sensation Index", z)
        )
        comfort[z + "_discomfort"] = myeplus.Variable(
            ("Zone Thermal Comfort Pierce Model Discomfort Index", z)
        )


def auto_add_energy_variables(rdf, variables):
    if "reward" not in variables:
        variables["energy"] = {}

    r = variables["energy"]
    for z in graph.rdf_zones(rdf):
        r[z + "_cooling"] = myeplus.Variable(
            ("Zone Air System Sensible Cooling Energy", z)
        )
        r[z + "_heating"] = myeplus.Variable(
            ("Zone Air System Sensible Heating Energy", z)
        )


def auto_add_base_variables(rdf, variables):
    """Add ubiquitous variables."""

    time = {}
    variables["time"] = time
    time["current_sim_time"] = myeplus.Function(myeplus.api.exchange.current_sim_time)
    time["time_of_day"] = myeplus.Function(myeplus.api.exchange.current_time)
    time["day_of_month"] = myeplus.Function(myeplus.api.exchange.day_of_month)
    time["day_of_week"] = myeplus.Function(myeplus.api.exchange.day_of_week)
    time["day_of_year"] = myeplus.Function(myeplus.api.exchange.day_of_year)

    # Not universal
    #
    # variables["demand_rate"] = Leaf(
    #     (
    #         "Facility Total HVAC Electricity Demand Rate",
    #         "Whole Building",
    #     )
    # )


def alburquerque():
    buildingfile = "./buildings/alburquerque.epJSON"
    weatherfile = "./honolulu.epw"

    rdf = graph.json_to_rdf(buildingfile)
    actuators = {}
    variables = {}

    auto_add_energy_variables(rdf, variables)
    auto_add_actuators(rdf, actuators)
    auto_add_actuators_observation(rdf, variables)
    auto_add_base_variables(rdf, variables)
    auto_add_temperature_variables(rdf, variables)
    return EnvConfig(buildingfile, weatherfile, variables, actuators)


def crawlspace():
    buildingfile = "./buildings/crawlspace.epJSON"
    weatherfile = "./miami.epw"

    rdf = graph.json_to_rdf(buildingfile)
    actuators = {}
    variables = {}

    auto_add_energy_variables(rdf, variables)
    auto_add_actuators(rdf, actuators)
    auto_add_actuators_observation(rdf, variables)
    auto_add_base_variables(rdf, variables)
    auto_add_temperature_variables(rdf, variables)
    # auto_add_comfort_variables(rdf, variables)
    # auto_add_setpoint_variables(rdf, variables)
    return EnvConfig(buildingfile, weatherfile, variables, actuators)


def add_all_variables(template):
    vs = [
        ["ENVIRONMENT", "Site Outdoor Air Drybulb Temperature", "", "OutputVariable"],
        [
            "LIVING HARDWIRED LIGHTING1_BACKROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING1_BACKROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING1_BACKROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING1_FRONTROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING1_FRONTROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING1_FRONTROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING2_BACKROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING2_BACKROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING2_BACKROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING2_FRONTROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING2_FRONTROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING2_FRONTROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING3_BACKROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING3_BACKROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING3_BACKROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING3_FRONTROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING3_FRONTROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING HARDWIRED LIGHTING3_FRONTROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING1_BACKROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING1_BACKROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING1_BACKROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING1_FRONTROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING1_FRONTROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING1_FRONTROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING2_BACKROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING2_BACKROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING2_BACKROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING2_FRONTROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING2_FRONTROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING2_FRONTROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING3_BACKROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING3_BACKROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING3_BACKROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING3_FRONTROW_BOTTOMFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING3_FRONTROW_MIDDLEFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING PLUG-IN LIGHTING3_FRONTROW_TOPFLOOR",
            "Lights Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ1_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ1_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ1_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ1_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ1_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ1_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ2_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ2_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ2_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ2_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ2_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ2_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ3_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ3_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ3_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ3_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ3_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "IECC_ADJ3_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER1_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER1_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER1_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER1_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER1_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER1_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER2_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER2_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER2_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER2_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER2_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER2_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER3_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER3_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER3_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER3_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER3_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "CLOTHESWASHER3_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER1_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER1_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER1_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER1_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER1_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER1_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER2_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER2_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER2_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER2_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER2_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER2_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER3_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER3_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER3_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER3_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER3_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "DISHWASHER3_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER1_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER1_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER1_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER1_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER1_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER1_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER2_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER2_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER2_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER2_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER2_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER2_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER3_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER3_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER3_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER3_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER3_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_DRYER3_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS1_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS1_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS1_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS1_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS1_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS1_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS2_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS2_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS2_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS2_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS2_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS2_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS3_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS3_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS3_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS3_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS3_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_MELS3_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE1_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE1_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE1_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE1_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE1_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE1_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE2_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE2_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE2_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE2_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE2_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE2_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE3_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE3_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE3_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE3_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE3_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "ELECTRIC_RANGE3_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR1_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR1_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR1_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR1_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR1_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR1_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR2_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR2_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR2_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR2_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR2_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR2_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR3_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR3_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR3_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR3_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR3_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "REFRIGERATOR3_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION1_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION1_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION1_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION1_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION1_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION1_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION2_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION2_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION2_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION2_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION2_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION2_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION3_BACKROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION3_BACKROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION3_BACKROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION3_FRONTROW_BOTTOMFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION3_FRONTROW_MIDDLEFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "TELEVISION3_FRONTROW_TOPFLOOR",
            "Electric Equipment Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_BACKROW_BOTTOMFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_BACKROW_MIDDLEFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_BACKROW_TOPFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_BOTTOMFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_MIDDLEFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_TOPFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_BOTTOMFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_MIDDLEFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_TOPFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_BOTTOMFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_MIDDLEFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_TOPFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_BOTTOMFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_MIDDLEFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_TOPFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_BOTTOMFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_MIDDLEFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_TOPFLOOR",
            "Zone Ventilation Fan Electricity Energy",
            "",
            "OutputVariable",
        ],
        ["BREEZEWAY", "Zone Air Temperature", "", "OutputVariable"],
        ["BREEZEWAY", "Zone Air System Sensible Heating Energy", "", "OutputVariable"],
        ["BREEZEWAY", "Zone Air System Sensible Cooling Energy", "", "OutputVariable"],
        ["ATTIC", "Zone Air Temperature", "", "OutputVariable"],
        ["ATTIC", "Zone Air System Sensible Heating Energy", "", "OutputVariable"],
        ["ATTIC", "Zone Air System Sensible Cooling Energy", "", "OutputVariable"],
        ["CRAWLSPACE", "Zone Air Temperature", "", "OutputVariable"],
        ["CRAWLSPACE", "Zone Air System Sensible Heating Energy", "", "OutputVariable"],
        ["CRAWLSPACE", "Zone Air System Sensible Cooling Energy", "", "OutputVariable"],
        [
            "LIVING_UNIT1_BACKROW_BOTTOMFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_BACKROW_BOTTOMFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_BACKROW_BOTTOMFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_BACKROW_MIDDLEFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_BACKROW_MIDDLEFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_BACKROW_MIDDLEFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        ["LIVING_UNIT1_BACKROW_TOPFLOOR", "Zone Air Temperature", "", "OutputVariable"],
        [
            "LIVING_UNIT1_BACKROW_TOPFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_BACKROW_TOPFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_BOTTOMFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_BOTTOMFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_BOTTOMFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_MIDDLEFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_MIDDLEFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_MIDDLEFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_TOPFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_TOPFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT1_FRONTROW_TOPFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_BOTTOMFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_BOTTOMFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_BOTTOMFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_MIDDLEFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_MIDDLEFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_MIDDLEFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        ["LIVING_UNIT2_BACKROW_TOPFLOOR", "Zone Air Temperature", "", "OutputVariable"],
        [
            "LIVING_UNIT2_BACKROW_TOPFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_BACKROW_TOPFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_BOTTOMFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_BOTTOMFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_BOTTOMFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_MIDDLEFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_MIDDLEFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_MIDDLEFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_TOPFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_TOPFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT2_FRONTROW_TOPFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_BOTTOMFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_BOTTOMFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_BOTTOMFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_MIDDLEFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_MIDDLEFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_MIDDLEFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        ["LIVING_UNIT3_BACKROW_TOPFLOOR", "Zone Air Temperature", "", "OutputVariable"],
        [
            "LIVING_UNIT3_BACKROW_TOPFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_BACKROW_TOPFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_BOTTOMFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_BOTTOMFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_BOTTOMFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_MIDDLEFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_MIDDLEFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_MIDDLEFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_TOPFLOOR",
            "Zone Air Temperature",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_TOPFLOOR",
            "Zone Air System Sensible Heating Energy",
            "",
            "OutputVariable",
        ],
        [
            "LIVING_UNIT3_FRONTROW_TOPFLOOR",
            "Zone Air System Sensible Cooling Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT1_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT1_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT1_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT1_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT1_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT1_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT1_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT1_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT1_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT1_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT1_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT1_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT1_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT1_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT1_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT1_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT1_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT1_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT1_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT1_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT1_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT1_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT1_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT1_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT1_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT1_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT1_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT1_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT1_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT1_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT2_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT2_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT2_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT2_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT2_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT2_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT2_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT2_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT2_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT2_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT2_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT2_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT2_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT2_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT2_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT2_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT2_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT2_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT2_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT2_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT2_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT2_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT2_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT2_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT2_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT2_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT2_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT2_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT2_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT2_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT3_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT3_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT3_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT3_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT3_BACKROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT3_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT3_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT3_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT3_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT3_BACKROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT3_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT3_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT3_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT3_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT3_BACKROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT3_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT3_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT3_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT3_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT3_FRONTROW_BOTTOMFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT3_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT3_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT3_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT3_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT3_FRONTROW_MIDDLEFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SINKS_UNIT3_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW SHOWERS_UNIT3_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW CLOTHESWASHER_UNIT3_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW DISHWASHER_UNIT3_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "DHW BATHS_UNIT3_FRONTROW_TOPFLOOR",
            "Water Use Connections Plant Hot Water Energy",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Total Purchased Electricity Rate",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Total Purchased Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Total Surplus Electricity Rate",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Total Surplus Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Net Purchased Electricity Rate",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Net Purchased Electricity Energy",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Total Building Electricity Demand Rate",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Total HVAC Electricity Demand Rate",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Total Electricity Demand Rate",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Total Produced Electricity Rate",
            "",
            "OutputVariable",
        ],
        [
            "WHOLE BUILDING",
            "Facility Total Produced Electricity Energy",
            "",
            "OutputVariable",
        ],
    ]

    all_vars = {}
    for (z, v, _, _) in vs:
        all_vars[z + v] = myeplus.Variable((v, z))

    template["all_variables"] = all_vars
