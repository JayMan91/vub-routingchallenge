from os import path
import sys, json, time
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import geopy.distance
from scipy.special import softmax, log_softmax
# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Read input data
print('Reading Input Data')
# Model Build output
model_path=path.join(BASE_DIR, 'data/model_build_outputs/zone_frequency_matrices.pickle')
with open(model_path, 'rb') as out_file:
    zone_frequency_matrices = pickle.load( out_file)
model_path=path.join(BASE_DIR, 'data/model_build_outputs/all_zones_ordered.pickle')
with open(model_path, 'rb') as out_file:
    all_zones_ordered = pickle.load( out_file)
model_path=path.join(BASE_DIR, 'data/model_build_outputs/all_zones_coodinates.pickle')
with open(model_path, 'rb') as out_file:
    all_zones_coodinates = pickle.load( out_file)   


# Prediction Routes (Model Apply input)
route_time_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json')
with open(route_time_path, newline='') as in_file:
    model_apply_time_data= json.load(in_file)

prediction_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
# with open(prediction_routes_path, newline='') as in_file:
#     prediction_routes = json.load(in_file)
model_apply_route_data = pd.read_json(prediction_routes_path)
model_apply_route_data_T = model_apply_route_data.T
print("Reading Input Data Complete")



# Algo
def create_data_model(prob_mat):
    """Stores the data for the problem."""
    # Note that distances SHOULD BE integers; multiply by 100 for probabilities
    data = {}
    data['distance_matrix'] = prob_mat
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data
def create_time_matrix(stopslist,jsondata):
    times = {}
    for i,source in enumerate(stopslist):
        times[i] = {}
        for j,dest in enumerate(stopslist):
            times[i][j] = jsondata[source][dest]
    return times

def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = np.zeros((len(locations),len(locations)  ))
    for from_counter, from_node in enumerate(locations):
        # distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                # distances[from_counter][to_counter] = (int(
                #     100*math.hypot((from_node[0] - to_node[0]), #multiplied by 100 because of int
                #                (from_node[1] - to_node[1]))))
                distances[from_counter][to_counter] = 100*geopy.distance.distance(from_node,to_node).km
    return distances
def get_routes(solution, routing, manager):
        """Get vehicle routes from a solution and store them in an array."""
        # Get vehicle routes and store them in a two dimensional array whose
        # i,j entry is the jth location visited by vehicle i along its route.
        routes = []
        for route_nbr in range(routing.vehicles()):
            index = routing.Start(route_nbr)
            route = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
            routes.append(route)
        return routes
def simple_algo(stops,proba_mat,inv_zones_selected):
    data = create_data_model(proba_mat)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                        data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback( from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        zone_order = get_routes(solution, routing, manager)
    zone_order_actual = list(map(inv_zones_selected.get,zone_order[0]))
#     print("zone_order_actual",zone_order_actual)
    locations = [] #initialize with loc of depot
    for i in stops:
        locations.append((stops[i]['lat'], stops[i]['lng']))
    dist_mat = compute_euclidean_distance_matrix(locations)
#     print(dist_mat)
    zone_list = []

    for i in stops:
        zone_list.append(stops[i]['zone_id'])

    zones_in_hist = [v for k,v in inv_zones_selected.items()]

    NoZones_idx = [i+1 for i, j in enumerate(zone_list[1:]) if j not in zones_in_hist]
    print("NoZones_idx", NoZones_idx)


    for NoZoneStop in NoZones_idx:
        dist_list = list(dist_mat[NoZoneStop].values())
        nearest_idx = dist_list.index(min([i for i in dist_list if i > 0]))
        zone_list[NoZoneStop] = zone_list[nearest_idx]   
    
    final_route = [0]
#     print("zone_list",zone_list)
    for z_id in zone_order_actual[1:-1]:
        stop_idx_in_zone = [i for i, j in enumerate(zone_list) if j == z_id]
        for num_of_stops_in_zone in range(len(stop_idx_in_zone)):
            dist_from_last_stop = []
            for idx in stop_idx_in_zone:
                if idx not in final_route:
                    dist_from_last_stop.append(dist_mat[final_route[-1]][idx])
                else:
                    dist_from_last_stop.append(1e+12)
            closest_idx = dist_from_last_stop.index(min(dist_from_last_stop))
            final_route.append(stop_idx_in_zone[closest_idx])

    
    return final_route

def simple_linearcombialgo(stops,proba_mat,time_mat,inv_zones_selected):

    locations = [] #initialize with loc of depot
    for i in stops:
        locations.append((stops[i]['lat'], stops[i]['lng']))
    dist_mat = compute_euclidean_distance_matrix(locations)
    
    zone_list = []
    for stop in stops:
        zone_list.append(stops[stop]["zone_id"])
    zones_in_hist = [v for k,v in inv_zones_selected.items()]

    NoZones_idx = [i+1 for i, j in enumerate(zone_list[1:]) if j not in zones_in_hist]
    for NoZoneStop in NoZones_idx:
        dist_list = list(dist_mat[NoZoneStop].values())
        nearest_idx = dist_list.index(min([i for i in dist_list if i > 0]))
        zone_list[NoZoneStop] = zone_list[nearest_idx]  
    zones_selected = {v: k for k, v in inv_zones_selected.items()} 
    zones_selected[None]=0
    zone_list_idx =  list(map(zones_selected.get,zone_list))
    combi_mat = np.zeros((len(stops),len(stops)))
    for i in range(len(stops)):
        for j in range(len(stops)):
            combi_mat[i][j] = dist_mat[i][j] + time_mat[i][j]+ proba_mat[zone_list_idx[i]][zone_list_idx[j]]
    data = create_data_model(combi_mat)

    # print("dist", dist_mat[1])
    # print("zone_mat", proba_mat[1])

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                        data['num_vehicles'], data['depot'])
    def distance_callback( from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        stop_order = get_routes(solution, routing, manager)[0][:-1]
        return stop_order
    else:
        raise Exception("Solution Not found")

def twostage_algo(stops,proba_mat,time_mat,zone_dist_mat,inv_zones_selected):
    locations = [] #initialize with loc of depot
    for i in stops:
        locations.append((stops[i]['lat'], stops[i]['lng']))
    dist_mat = compute_euclidean_distance_matrix(locations)
    zone_list = []
    for stop in stops:
        zone_list.append(stops[stop]["zone_id"])
    zones_in_hist = [v for k,v in inv_zones_selected.items()]

    NoZones_idx = [i+1 for i, j in enumerate(zone_list[1:]) if j not in zones_in_hist]
    for NoZoneStop in NoZones_idx:
        dist_list =   dist_mat[NoZoneStop].tolist() #list(dist_mat[NoZoneStop].values())
        nearest_idx = dist_list.index(min([i for i in dist_list if i > 0]))
        zone_list[NoZoneStop] = zone_list[nearest_idx]  
    zones_selected = {v: k for k, v in inv_zones_selected.items()} 
    zones_selected[None]=0
    zone_list_idx =  list(map(zones_selected.get,zone_list))
    
    # zone_dist_mat = softmax(zone_dist_mat, axis=1)
    # zone_dist_mat = zone_dist_mat/zone_dist_mat.sum(axis=-1,keepdims=True)
    # alpha = 0.001

    # comb_mat = alpha*zone_dist_mat - (1-alpha)*proba_mat
    comb_mat = 0.1*zone_dist_mat - proba_mat

    data = create_data_model(100*comb_mat)
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                        data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)    
    # Register callback with the solver.
    def distance_callback( from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 100

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    zone_order = get_routes(solution, routing, manager)
    zone_order_actual = list(map(inv_zones_selected.get,zone_order[0]))

    combi_mat = np.zeros((len(stops),len(stops)))
    for i in range(len(stops)):
        for j in range(len(stops)):
            # compute for zone penalty
            if zone_list_idx[i] == zone_list_idx[j]:
                zone_penalty = 0
            elif np.where(zone_dist_mat[zone_list_idx[j]]==np.partition(zone_dist_mat[ zone_list_idx[j]],1)[1])[0][0]  == zone_list_idx[i]:
                zone_penalty = 75
            elif zone_order[0].index(zone_list_idx[j])-1 == zone_order[0].index(zone_list_idx[i]): 
                zone_penalty = 80
            elif zone_order[0].index(zone_list_idx[j])-2 == zone_order[0].index(zone_list_idx[i]):
                zone_penalty = 85
            elif zone_order[0].index(zone_list_idx[j])-3 == zone_order[0].index(zone_list_idx[i]):
                zone_penalty = 100
            else:
                zone_penalty = 150
            combi_mat[i][j] = time_mat[i][j] + zone_penalty 
    data = create_data_model(combi_mat)
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                        data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Register callback with the solver.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 120

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        stop_order = get_routes(solution, routing, manager)[0][:-1]
        return stop_order


op_dict = {}
for index, model_apply_route_data_T0 in model_apply_route_data_T.iterrows():

    route_timedata = model_apply_time_data[index]
    station_id0= model_apply_route_data_T0['station_code']
    zones_ordered = all_zones_ordered[station_id0]
    zones_coodinates = all_zones_coodinates[station_id0]

    zone_freq0 = zone_frequency_matrices[station_id0]
    route_T0 = pd.DataFrame(model_apply_route_data_T0['stops']).T.sort_values(by='type',
                                        ascending=False)
    stops =  route_T0.to_dict('index')


    try:
        route_T0['zone_matrix_id'] = route_T0.zone_id.map(zones_ordered).fillna(0).astype(int)
        station_lat, station_lng = route_T0.lat[0],route_T0.lng[0]
        zoneidx_list = route_T0.zone_matrix_id.unique().tolist()
        zones_selected_coordinates = [  zones_coodinates[k] for k in zoneidx_list ]
        zone_locations =  [k['coordinate'] for k in zones_selected_coordinates]  
        zone_dist_mat = compute_euclidean_distance_matrix(zone_locations)
        

        time_mat = create_time_matrix([*stops],route_timedata)


        proba_mat = zone_freq0[np.ix_(zoneidx_list,zoneidx_list )]
        proba_mat = proba_mat/proba_mat.sum(axis=-1,keepdims=True)
        # proba_mat = - 100*proba_mat

        inv_zones_ordered = {v: k for k, v in zones_ordered.items()}
        inv_zones_selected = {k: inv_zones_ordered[ zoneidx_list[k] ] for k in  range(len(zoneidx_list)) }

        op_seq= twostage_algo(stops,proba_mat,time_mat,zone_dist_mat, inv_zones_selected)
        not_in_opseq = list(set([i for i in range(len(route_T0))]) - set(op_seq))
        # print("not_in_opseq", not_in_opseq)
        op_seq = op_seq + not_in_opseq
        op_list = route_T0.index[op_seq].tolist()
        op_dict [index] = { 'proposed':  {k: v for v, k in enumerate(op_list )} }
    except:
        print("Error!!")
        op_dict [index] = { 'proposed':  {k: v for v, k in enumerate( list(stops.keys()) )} }



# Write output data
output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
with open(output_path, 'w') as out_file:
    json.dump(op_dict, out_file)
    print("Success: The '{}' file has been saved".format(output_path))

print('Done!')

  



# def sort_by_key(stops, sort_by):
#     """
#     Takes in the `prediction_routes[route_id]['stops']` dictionary
#     Returns a dictionary of the stops with their sorted order always placing the depot first

#     EG:

#     Input:
#     ```
#     stops={
#       "Depot": {
#         "lat": 42.139891,
#         "lng": -71.494346,
#         "type": "depot",
#         "zone_id": null
#       },
#       "StopID_001": {
#         "lat": 43.139891,
#         "lng": -71.494346,
#         "type": "delivery",
#         "zone_id": "A-2.2A"
#       },
#       "StopID_002": {
#         "lat": 42.139891,
#         "lng": -71.494346,
#         "type": "delivery",
#         "zone_id": "P-13.1B"
#       }
#     }

#     print (sort_by_key(stops, 'lat'))
#     ```

#     Output:
#     ```
#     {
#         "Depot":1,
#         "StopID_001":3,
#         "StopID_002":2
#     }
#     ```

#     """
#     # Serialize keys as id into each dictionary value and make the dict a list
#     stops_list=[{**value, **{'id':key}} for key, value in stops.items()]

#     # Sort the stops list by the key specified when calling the sort_by_key func
#     ordered_stop_list=sorted(stops_list, key=lambda x: x[sort_by])

#     # Keep only sorted list of ids
#     ordered_stop_list_ids=[i['id'] for i in ordered_stop_list]

#     # Serialize back to dictionary format with output order as the values
#     return {i:ordered_stop_list_ids.index(i) for i in ordered_stop_list_ids}

# def propose_all_routes(prediction_routes, sort_by):
#     """
#     Applies `sort_by_key` to each route's set of stops and returns them in a dictionary under `output[route_id]['proposed']`

#     EG:

#     Input:
#     ```
#     prediction_routes = {
#       "RouteID_001": {
#         ...
#         "stops": {
#           "Depot": {
#             "lat": 42.139891,
#             "lng": -71.494346,
#             "type": "depot",
#             "zone_id": null
#           },
#           ...
#         }
#       },
#       ...
#     }

#     print(propose_all_routes(prediction_routes, 'lat'))
#     ```

#     Output:
#     ```
#     {
#       "RouteID_001": {
#         "proposed": {
#           "Depot": 0,
#           "StopID_001": 1,
#           "StopID_002": 2
#         }
#       },
#       ...
#     }
#     ```
#     """
#     return {key:{'proposed':sort_by_key(stops=value['stops'], sort_by=sort_by)} for key, value in prediction_routes.items()}

# # Apply faux algorithms to pass time
# time.sleep(1)
# print('Solving Dark Matter Waveforms')
# time.sleep(1)
# print('Quantum Computer is Overheating')
# time.sleep(1)
# print('Trying Alternate Measurement Cycles')
# time.sleep(1)
# print('Found a Great Solution!')
# time.sleep(1)
# print('Checking Validity')
# time.sleep(1)
# print('The Answer is 42!')
# time.sleep(1)


# print('\nApplying answer with real model...')
# sort_by=model_build_out.get("sort_by")
# print('Sorting data by the key: {}'.format(sort_by))
# output=propose_all_routes(prediction_routes=prediction_routes, sort_by=sort_by)
# print('Data sorted!')

# # Write output data
# output_path=path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')
# with open(output_path, 'w') as out_file:
#     json.dump(output, out_file)
#     print("Success: The '{}' file has been saved".format(output_path))

# print('Done!')
