import os
import eppy
from eppy.modeleditor import IDF
from networkx.drawing import nx_pydot
from networkx import DiGraph, draw
import matplotlib
import matplotlib.pyplot as plt
# Initialize a directed graph

# Add nodes and edges based on some relationships
# Here we're adding zones and their associated HVAC systems as an example
def equipment_graph(idf, graph=DiGraph()):
    for obj in idf.idfobjects["ZONEHVAC:EQUIPMENTCONNECTIONS"]:
        zone = obj.Zone_Name
        graph.add_node(zone, type='Zone')
        equipment = obj.Zone_Conditioning_Equipment_List_Name
        graph.add_node(equipment, type='Equipment', shape="rectangle")
        graph.add_edge(zone, equipment)
    return graph


def zone_graph(idf, graph=DiGraph()):
    graph.add_node("Outdoors", shape="rectangle", color="blue")
    for sur in idf.idfobjects["BuildingSurface:Detailed"]:
        obc = sur.Outside_Boundary_Condition
        if obc == "Zone":
            zone_from = sur.Zone_Name
            zone_to = sur.Outside_Boundary_Condition_Object
            graph.add_node(zone_from, type="Zone")
            graph.add_node(zone_to, type="Zone")
            graph.add_edge(zone_from, zone_to)
        elif obc == "Outdoors":
            zone_from = sur.Zone_Name
            graph.add_node(zone_from, type="Zone")
            graph.add_edge(zone_from, "Outdoors")

    return graph

def draw_surfaces(idf):
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

prof = str(os.getenv("GUIX_LOAD_PROFILE"))

matplotlib.use("TkAgg")
# prof + "/Energy+.idd"
iddfile = "./V9-5-0-Energy+.idd"

# Load the IDF file
IDF.setiddname(iddfile)
# idffile = prof + "/US+SF+CZ1AWH+elecres+crawlspace+IECC_2006.idf"
idffile = prof + "/US+MF+CZ1AWH+elecres+crawlspace+IECC_2006.idf"

idf = IDF(idffile)

if __name__ == "__main__":

    # Draw the graph

    # pos = draw(graph, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold')
    # plt.show()
    graph = zone_graph(idf)
    graph = equipment_graph(idf, graph=graph)
    nx_pydot.write_dot(graph, "graph.dot")
