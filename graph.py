#!/usr/bin/env python3
import sys
import profile
import rdflib
import rdflib.namespace
import json


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def quote(n):
    if ":" in n:
        return '"' + n + '"'
    return n


def remove_surfaces(graph):

    # Here, zones (and outdoors) are potentially only connected through
    # surfaces. We want to connect zones that are connected through a path
    # containing exclusively surfaces, so that we can remove surfaces and keep a
    # topology that makes sense
    all_zones = [
        n for n, d in graph.nodes(data=True) if d["type"] in ["Zone", "Outdoors"]
    ]
    n = len(all_zones)
    for i in range(n):
        for j in range(i):
            for p in list(nx.all_simple_paths(graph, all_zones[i], all_zones[j])):
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


def test():
    with open("buildings/crawlspace.epJSON", "rb") as f:
        obj = json.load(f)

    def draw_surfaces_epjson(obj):

        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        surfaces = []
        for name, val in obj["BuildingSurface:Detailed"].items():
            print(name)
            xs = []
            ys = []
            zs = []

            for vertex in val["vertices"]:
                xs.append(vertex["vertex_x_coordinate"])
                ys.append(vertex["vertex_y_coordinate"])
                zs.append(vertex["vertex_z_coordinate"])

            xs.append(xs[0])
            ys.append(ys[0])
            zs.append(zs[0])
            surfaces.append({"xs": xs, "ys": ys, "zs": zs, "name": name})
        ax = plt.figure().add_subplot(projection="3d")
        # Draw the graph of the zones
        for sur in surfaces:
            xs = sur["xs"]
            ys = sur["ys"]
            zs = sur["zs"]
            ax.plot(xs, ys, zs, label=sur["name"])

        # ax.legend()
        # plt.show()
        plt.savefig("building.svg")

    draw_surfaces_epjson(obj)


def draw_surfaces(idf):
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    surfaces = []
    for sur in idf.idfobjects["BuildingSurface:Detailed"]:
        xs = []
        ys = []
        zs = []
        for i in range(1, int(sur["Number_of_Vertices"] + 1)):
            xs.append(sur[f"Vertex_{i}_Xcoordinate"])
            ys.append(sur[f"Vertex_{i}_Ycoordinate"])
            zs.append(sur[f"Vertex_{i}_Zcoordinate"])
        xs.append(xs[0])
        ys.append(ys[0])
        zs.append(zs[0])

        surfaces.append({"xs": xs, "ys": ys, "zs": zs, "name": sur["Name"]})

    ax = plt.figure().add_subplot(projection="3d")
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
    parser.add_argument(
        "--idd",
        help="the path to the idd file to be used",
        default=profile.PROFILE + "/Energy+.idd",
    )
    parser.add_argument(
        "--output", "-o", help="the output file to write to", default="/dev/stdout"
    )
    args = parser.parse_args()
    return args


def intern_object(
    g: rdflib.Graph,
    subject,
    value,
    ns=rdflib.Namespace("http://terramorpha.org/"),
):
    """Take a rdf graph, a rdf subject and a python dict/array and intern
    it into the ontology.

    Dictionnaries are interned using the keys as predicates, lists use
    has-elem.
    """

    if type(value) == list:
        haselem = ns.haselem
        hasindex = ns.hasindex
        for i, elem in enumerate(value):
            if type(elem) in [str, int, float]:
                name = rdflib.Literal(elem)
            elif type(elem) in [dict, list]:
                name = rdflib.BNode()
                intern_object(g, name, elem)
            else:
                raise BaseException("erreur!")

            g.add((subject, haselem, name))
            g.add((name, hasindex, rdflib.Literal(i)))

    elif type(value) == dict:
        for k, elem in value.items():
            if type(elem) in [str, int, float]:
                name = rdflib.Literal(elem)
                g.add((subject, ns[k], name))
            elif type(elem) in [dict, list]:
                name = rdflib.BNode()
                intern_object(g, name, elem)
            else:
                raise BaseException("erreur!")
            g.add((subject, ns[k], name))


def json_to_rdf(jsonfile):
    """Take an epJSON file path, read it and transform it into an RDF
    representation that can be queried (through .query(q: str)) in SPARQL.
    """
    n = rdflib.Namespace("http://terramorpha.org/")
    isa = rdflib.namespace.RDF.type
    g = rdflib.Graph()
    g.bind("idf", n)

    with open(jsonfile, "rb") as f:
        j = json.load(f)

    for typename, elems in j.items():
        for name, keyvals in elems.items():
            rTypename = rdflib.Literal(typename)
            rName = rdflib.Literal(name)
            # rTypename = n[typename]
            # rName = n[name]
            g.add((rName, isa, rTypename))
            for key, val in keyvals.items():
                if type(val) in [str, float, int]:
                    name = rdflib.Literal(val)
                    g.add((rName, n[key], name))
                if type(val) == list:
                    name = rdflib.BNode()
                    g.add((rName, n[key], name))
                    intern_object(g, name, val)

    return g


def rdf_to_adjacency(g: rdflib.Graph):
    """Take a rdf representation of an idf file and return a new graph of zones
    idf:is_connected_to each other through surfaces'
    outside_boundary_condition_object properties."""

    query = """# -*- mode: sparql -*-
CONSTRUCT {
  ?src idf:is_connected_to ?dst .
} WHERE {
  ?surface1 idf:zone_name ?src .

  {
    ?surface1  (idf:outside_boundary_condition_object | ^idf:outside_boundary_condition_object)+  ?surface2 .
    ?surface2 idf:zone_name ?dst .
  } UNION {
    ?surface1 idf:outside_boundary_condition "Outdoors" .
    BIND ("Outdoors" as ?dst)
  }
  FILTER (?src < ?dst)
}
"""
    resp = g.query(query)
    return resp.graph


def rdf_to_dot(rdf: rdflib.Graph):
    """Take an rdflib graph and return its representation in the graphviz dot
    format."""

    o = ""
    o += "digraph G {\n"
    for (a, b, c) in rdf.query("""SELECT ?a ?b ?c WHERE { ?a ?b ?c . }"""):
        o += f'"{a}" -> "{c}" [label="{b}"];\n'
    o += "}\n"
    return o


def rdf_zones(rdf: rdflib.Graph):
    """Return a list of all the zones in the building"""
    q = """# -*- mode: sparql -*-
SELECT ?name WHERE {
  ?name a "Zone" .
}"""
    return list(set(x.value for (x,) in rdf.query(q)))


def rdf_schedules(rdf: rdflib.Graph):
    """Return a list of all the scheduler names in the building"""
    q = """# -*- mode: sparql -*-
SELECT ?name WHERE {
  ?name a "Schedule:Compact" .
}"""
    return list(set(x.value for (x,) in rdf.query(q)))
