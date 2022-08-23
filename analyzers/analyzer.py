class CrossingNode:
    def __init__(self, endpoint=False):
        # by conventions, endpoints will have only plus edges
        self.plus_edges = []
        self.minus_edges = []
        self.endpoint = endpoint

    def continuing_edge(self, incoming_edge):
        if incoming_edge in self.plus_edges:
            # return the other positive edge
            return self.plus_edges[1 - self.plus_edges.index(incoming_edge)]
        elif incoming_edge in self.minus_edges:
            # return the other negative edge
            return self.minus_edges[1 - self.minus_edges.index(incoming_edge)]

class Edge:
    def __init__(self, node1, node1plus, node2, node2plus):
        self.node1 = node1
        self.node2 = node2
        self.node1plus = node1plus
        self.node2plus = node2plus
        if node1plus:
            node1.plus_edges.append(self)
        else:
            node1.minus_edges.append(self)

        if node2plus:
            node2.plus_edges.append(self)
        else:
            node2.minus_edges.append(self)

    def other_node(self, node):
        if node == self.node1:
            return self.node2
        else:
            return self.node1

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
    
    def get_endpoint(self):
        for node in self.nodes:
            if node.endpoint:
                return node

    def get_next_node(self, node, prev_edge=None):
        if prev_edge is None:
            # we are at an endpoint
            edge_from_endpoint = node.plus_edges[0]
            return edge_from_endpoint.other_node(node), edge_from_endpoint
        else:
            continuing_edge = node.continuing_edge(prev_edge)
            return continuing_edge.other_node(node), continuing_edge

    def simplify(self):
        for node1 in self.nodes:
            for adjacent_plus in node1.plus_edges:
                node2 = adjacent_plus.other_node(node1)
                if adjacent_plus in node2.plus_edges:


    def check_untangled(self):
        start_node = self.get_endpoint()



if __name__  == "__main__":
    # construct graph for simple trivial loop
    node_1 = CrossingNode()
    left_end = CrossingNode(True)
    right_end = CrossingNode(True)

    left_end_edge = Edge(node_1, True, left_end, True)
    right_end_edge = Edge(node_1, False, right_end, True)
    middle_edge = Edge(node_1, True, node_1, False)

    knot_graph = Graph([node_1, left_end, right_end], [left_end_edge, right_end_edge, middle_edge])