#!/usr/bin/env python3
import os
import sys
import eppy
from eppy.modeleditor import IDF
from networkx.drawing import nx_pydot
from networkx import Graph, DiGraph, draw, all_simple_paths

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Initialize a directed graph

# Add nodes and edges based on some relationships
# Here we're adding zones and their associated HVAC systems as an example
def equipment_graph(idf, graph=DiGraph()):
    for obj in idf.idfobjects["ZONEHVAC:EQUIPMENTCONNECTIONS"]:
        zone = obj.Zone_Name
        graph.add_node(zone, type='Zone')
        equipment = obj.Zone_Conditioning_Equipment_List_Name
        graph.add_node(equipment, type='Equipment')
        graph.add_edge(zone, equipment)
    return graph

def quote(n):
    if ":" in n:
        return "\""+n+"\""
    return n

def zone_graph(idf, graph=Graph()):
    graph.add_node("Outdoors", type="Outdoors")
    for z in idf.idfobjects["Zone"]:
        graph.add_node(z.Name, type="Zone")
    for sur in idf.idfobjects["BuildingSurface:Detailed"]:
        sur_name = quote(sur.Name)
        zone = sur.Zone_Name
        # On associe la surface à sa zone.
        graph.add_node(zone, type="Zone")
        graph.add_node(sur_name, type="Surface")
        graph.add_edge(zone, sur_name)
        obc = sur.Outside_Boundary_Condition
        # Et on associe aussi la surface à la chose qui est "outside".
        if obc == "Zone":
            zone_to = sur.Outside_Boundary_Condition_Object
            graph.add_node(zone_to, type="Zone")
            graph.add_edge(sur_name, zone_to)
        elif obc == "Surface":
            sur_to = quote(sur.Outside_Boundary_Condition_Object)
            graph.add_node(sur_to, type="Surface")
            graph.add_edge(sur_name, sur_to)
        elif obc == "Outdoors":
            zone_from = sur.Zone_Name
            graph.add_node(zone_from, type="Zone")
            graph.add_edge(sur_name, "Outdoors")
    return graph

def remove_surfaces(graph):

    # Here, zones (and outdoors) are potentially only connected through
    # surfaces. We want to connect zones that are connected through a path
    # containing exclusively surfaces, so that we can remove surfaces and keep a
    # topology that makes sense
    all_zones = [n for n, d in graph.nodes(data=True) if d["type"] in ["Zone", "Outdoors"]]
    n = len(all_zones)
    for i in range(n):
        zi = all_zones[i]
        for j in range(i):
            zj = all_zones[j]
            for p in list(all_simple_paths(graph, all_zones[i], all_zones[j])):
                if zi == "West Plenum" and zj == "West Zone":
                    eprint(p)
                # On veut savoir si deux zones ne sont reliées que par des
                # surfaces. Si c'est le cas, on voudrait les relier directement.
                is_only_surfaces = True
                for n in p[1:-1]:
                    if graph.nodes[n]["type"] != "Surface":
                        is_only_surfaces = False
                        break
                if is_only_surfaces:
                    graph.add_edge(all_zones[i], all_zones[j])

    for n, d in list(graph.nodes(data=True)):
        if d["type"] == "Surface":
            graph.remove_node(n)
    return graph

def color_graph(graph):
    for name, data in graph.nodes(data=True):
        if data["type"] == "Outdoors":
            data["color"] = "blue"
        if data["type"] == "Equipment":
            data["shape"] = "rectangle"
        if data["type"] == "Surface":
            data["color"] = "green"

def draw_surfaces(idf):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    surfaces = []
    for sur in idf.idfobjects["BuildingSurface:Detailed"]:
        xs = []
        ys = []
        zs = []
        for i in range(1, int(sur["Number_of_Vertices"]+1)):
            xs.append(sur[f"Vertex_{i}_Xcoordinate"])
            ys.append(sur[f"Vertex_{i}_Ycoordinate"])
            zs.append(sur[f"Vertex_{i}_Zcoordinate"])
        xs.append(xs[0])
        ys.append(ys[0])
        zs.append(zs[0])

        surfaces.append({
            "xs": xs,
            "ys": ys,
            "zs": zs,
            "name": sur["Name"]
        })

    ax = plt.figure().add_subplot(projection='3d')
    # Draw the graph of the zones
    for sur in surfaces:
        xs = sur["xs"]
        ys = sur["ys"]
        zs = sur["zs"]
        ax.plot(xs, ys, zs, label=sur["name"])

    ax.legend()
    plt.show()

def parseargs():
    import argparse
    parser = argparse.ArgumentParser(
        prog="graph.py",
        description="Create a graph of the zones from an idf file",
    )
    parser.add_argument("filename")
    parser.add_argument(
        "--draw",
        action="store_true",
        help="instead of outputting graphviz code, show a model of the building",
    )
    parser.add_argument("--idd", help="the path to the idd file to be used")
    parser.add_argument("--output", "-o", help="the output file to write to", default="/dev/stdout")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseargs()

    iddfile = args.idd or "./V9-5-0-Energy+.idd"
    IDF.setiddname(iddfile)

    idffile = args.filename # prof + "/US+MF+CZ1AWH+elecres+crawlspace+IECC_2006.idf"
    idf = IDF(idffile)
    # Draw the graph

    if args.draw:
        draw_surfaces(idf)
    else:
        graph = zone_graph(idf)
        color_graph(graph)
        graph = equipment_graph(idf, graph=graph)
        remove_surfaces(graph)
        # pos = draw(graph, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold')
        # plt.show()
        nx_pydot.write_dot(graph, args.output)
