from os import path
import sys, json, time
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Read input data
print('Reading Input Data')
training_routes_path=path.join(BASE_DIR, 'data/model_build_inputs/route_data.json')
actual_sequences_path=path.join(BASE_DIR, 'data/model_build_inputs/actual_sequences.json')
# with open(training_routes_path, newline='') as in_file:
#     actual_routes = json.load(in_file)

route_data_df = pd.read_json(training_routes_path, 
orient='index')# shape: (6, 6112)
actual_sequences_df = pd.read_json(actual_sequences_path,
 orient='index')# shape: (6, 6112)
route_actual_sequence_data_df = (pd.merge(route_data_df, actual_sequences_df,
 left_index=True, right_index=True, how = "inner"))


# apply transformations 
def sequence_of_zones(row):
    sequence = []
    actual = sorted(list(row["actual"]), key=lambda l: row["actual"][l])
    actual.append(actual[0])
    # finding the sequence of zones
    
    for i, stop in enumerate(actual):
        if len(sequence) == 0 or i == len(actual)-1:
            sequence.append("0")
        elif row["stops"][stop]["zone_id"] is None:
            continue
        elif row["stops"][stop]["zone_id"] != sequence[-1]:
            sequence.append(row["stops"][stop]["zone_id"])
        # drop the stop if it the stop has no zone id
   
    return sequence

route_actual_sequence_data_df["sequences_of_zones"] = route_actual_sequence_data_df.apply(lambda row: sequence_of_zones(row),
 axis = 1)
route_actual_sequence_data_df["numeric_route_score"] = route_actual_sequence_data_df.route_score.map({"High":5,"Medium":4,"Low":3})

# for every station all sequences of zones
station_sequences = defaultdict(list)
all_zones = defaultdict(set)

for id, row in route_actual_sequence_data_df.iterrows():
    station_sequences[row["station_code"]].append((row["sequences_of_zones"],row['numeric_route_score']))
    all_zones[row["station_code"]] |= set(row["sequences_of_zones"])
all_zones_ordered = {k: {zone:id for id, zone in enumerate(sorted(list(v)))} for k,v in all_zones.items()}

zone_frequency_matrices = {}
for k, v in all_zones_ordered.items():
    zone_frequency_matrices[k] = np.zeros((len(v), len(v)))

    
for station_code,list_of_sequences in station_sequences.items():
    for tuples in list_of_sequences:
        sequence = tuples[0]
        for i in range(len(sequence)-1):
            zone_pos_cur = all_zones_ordered[station_code][sequence[i]]
            zone_pos_next = all_zones_ordered[station_code][sequence[i+1]]
            #print(zone_pos_cur, "->", zone_pos_next)
            if zone_pos_cur == zone_pos_next:
                print(sequence)
                print("error!",i, zone_pos_cur, zone_pos_next)
            zone_frequency_matrices[station_code][zone_pos_cur][zone_pos_next] += tuples[1]


zone_coordinates = {}
for station_code in station_sequences.keys():
    zones_ordered = all_zones_ordered[station_code]
    zone_mapping = []
    stops = route_data_df[route_data_df['station_code']==station_code]['stops']
    for index, row in stops.items():
        for k,stop in row.items():
            zone_mapping.append({'zone_id':stop['zone_id'], 'coordinate':(stop['lat'],stop['lng'])})
    zone_mapping = pd.DataFrame(zone_mapping)
    zone_mapping = zone_mapping.groupby('zone_id').agg({'coordinate':lambda lst: list(set(lst.tolist()))})
    zone_mapping = zone_mapping.explode('coordinate').reset_index()
    zone_mapping.loc[:,'lat']= zone_mapping.coordinate.map(lambda x:x[0])
    zone_mapping.loc[:,'lng']= zone_mapping.coordinate.map(lambda x:x[1])
    df = zone_mapping.groupby('zone_id').agg({'lat':'mean','lng':'mean'}).reset_index()
    df['coordinate'] = list(zip(df.lat, df.lng))
    df['zone_order'] = df.zone_id.map(zones_ordered).fillna(0).astype(int)
    df = df.sort_values(by='zone_order')
    zone_coordinates[station_code] = df[['zone_id','coordinate']].to_dict(orient='records')




# Write output data
model_path=path.join(BASE_DIR, 'data/model_build_outputs/zone_frequency_matrices.pickle')
with open(model_path, 'wb') as out_file:
    pickle.dump(zone_frequency_matrices, out_file)
    print("Success: The '{}' file has been saved".format(model_path))

model_path=path.join(BASE_DIR, 'data/model_build_outputs/all_zones_ordered.pickle')
with open(model_path, 'wb') as out_file:
    pickle.dump(all_zones_ordered, out_file)
    print("Success: The '{}' file has been saved".format(model_path))

model_path=path.join(BASE_DIR, 'data/model_build_outputs/all_zones_coodinates.pickle')
with open(model_path, 'wb') as out_file:
    pickle.dump(zone_coordinates, out_file)
    print("Success: The '{}' file has been saved".format(model_path))






# route_data_df = pd.read_json("../data/model_build_inputs/route_data.json",
# orient= 'index')
# actual_sequence_df = pd.read_json("../data/model_build_inputs/actual_sequences.json")
# invalid_sequence_df = pd.read_json("../data/model_build_inputs/invalid_sequence_scores.json")
# travel_times_df = pd.read_json("../data/model_build_inputs/travel_times.json")
# package_data_df = pd.read_json("../data/model_build_inputs/package_data.json")



# # Solve for something hard
# print('Initializing Quark Reducer')
# # time.sleep(1)
# print('Placing Nano Tubes In Gravitational Wavepool')
# # time.sleep(1)
# print('Measuring Particle Deviations')
# # time.sleep(1)
# print('Programming Artificial Noggins')
# # time.sleep(1)
# print('Beaming in Complex Materials')
# # time.sleep(1)
# print('Solving Model')
# # time.sleep(1)
# print('Saving Solved Model State')
# output={
#     'Model':'Hello from the model_build.py script!',
#     'sort_by':'lat'
# }

# # Write output data
# model_path=path.join(BASE_DIR, 'data/model_build_outputs/model.json')
# with open(model_path, 'w') as out_file:
#     json.dump(output, out_file)
#     print("Success: The '{}' file has been saved".format(model_path))
