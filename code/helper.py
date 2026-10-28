import load_data as L

tr = L.load_trips()
st = L.build_stations(tr, L.load_station_list())
cl = L.neu_cluster(st)
print(len(st), "stations,", len(cl), "in cluster")
print(cl[["station_id", "name", "capacity", "match", "dist_to_neu_km"]].to_string(index=False))
print("cluster stations missing capacity:", cl.loc[cl["capacity"].isna(), "name"].tolist())
print("trips touching cluster:", len(L.filter_trips_to_cluster(tr, cl)))