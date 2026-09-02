import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# 1. SIMULATED DATABASE (From IoT Sensors)
BIN_DATABASE = [
    {"id": "DEPOT", "coords": (0, 0), "fill_level": 0},      # Start/End point
    {"id": "BIN_001", "coords": (2, 4), "fill_level": 90},   # FULL
    {"id": "BIN_002", "coords": (5, 2), "fill_level": 20},   # Ignore (Empty)
    {"id": "BIN_003", "coords": (7, 6), "fill_level": 95},   # FULL
    {"id": "BIN_004", "coords": (3, 8), "fill_level": 15},   # Ignore (Empty)
    {"id": "BIN_005", "coords": (8, 9), "fill_level": 88},   # FULL
    {"id": "BIN_006", "coords": (1, 7), "fill_level": 85},   # FULL
]
FILL_THRESHOLD = 80 # Only collect bins >= 80% full

def get_locations_to_visit(database):
    locations = [database[0]["coords"]]
    node_names = [database[0]["id"]]
    for bin_data in database[1:]:
        if bin_data["fill_level"] >= FILL_THRESHOLD:
            locations.append(bin_data["coords"])
            node_names.append(bin_data["id"])
    return locations, node_names

def compute_euclidean_distance_matrix(locations):
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                dist = math.hypot(from_node[0] - to_node[0], from_node[1] - to_node[1])
                distances[from_counter][to_counter] = int(dist * 10)
    return distances

def main():
    locations, node_names = get_locations_to_visit(BIN_DATABASE)
    distance_matrix = compute_euclidean_distance_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print("\n🎯 OPTIMIZED ROUTE FOUND:")
        index = routing.Start(0)
        plan_output = 'Route:\n'
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            plan_output += f' {node_names[node_index]} ->'
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        node_index = manager.IndexToNode(index)
        plan_output += f' {node_names[node_index]}\n'
        print(plan_output)
        print(f"Total Route Distance: {route_distance / 10.0} units\n")
    else:
        print("No solution found!")

if __name__ == '__main__':
    main()