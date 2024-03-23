import os, sys, copy, time, itertools, random
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

def main():
    def mahalanobis_distance(point, cluster):
        c = cluster["SUM"] / len(cluster["N"])
        sigma = cluster["SUMSQ"] / len(cluster["N"]) - (cluster["SUM"] / len(cluster["N"])) ** 2
        Sum = (point - c) / sigma
        return np.dot(Sum, Sum) ** (1 / 2)

    def mahalanobis_distance_cluster(cluster_1, cluster_2):
        center_1 = cluster_1["SUM"] / len(cluster_1["N"])
        center_2 = cluster_2["SUM"] / len(cluster_1["N"])
        sigma_1 = cluster_1["SUMSQ"] / len(cluster_1["N"]) - (cluster_1["SUM"] / len(cluster_1["N"])) ** 2
        sigma_2 = cluster_2["SUMSQ"] / len(cluster_2["N"]) - (cluster_2["SUM"] / len(cluster_2["N"])) ** 2
        sum_1 = (center_1 - center_2) / sigma_1
        sum_2 = (center_1 - center_2) / sigma_2
        return min(np.dot(sum_1, sum_1) ** (1 / 2), np.dot(sum_2, sum_2) ** (1 / 2))
    
    def fit_kmeans(data, n_clusters):
        k_means = KMeans(n_clusters=n_clusters).fit(data)
        return k_means

    start_time = time.time()
    
    random_seed = 553
    random.seed(random_seed)
    
    input_file = sys.argv[1]#'/content/hw6_clustering.txt'  # sys.argv[1]
    n_cluster = int(sys.argv[2])#10  # int(sys.argv[2])
    output_file = sys.argv[3]#'output.txt'  # sys.argv[3]

    DS = {}
    RS = []
    RS_idx = []
    result = ["The intermediate results:\n"]

    with open(input_file) as f:
        data = f.readlines()

    data = [line.strip("\n").split(',') for line in data]
    data = [(int(line[0]), tuple(map(float, line[2:]))) for line in data]
    data_dict = {k: v for k, v in data}
    data_dict_reversed = {v: k for k, v in data_dict.items()}
    data = [np.array(x) for x in data_dict.values()]
    random.shuffle(data)
    cluster_len = round(len(data) / 5)
    cluster_data = data[:cluster_len]

    n_cluster_multiplier=10
    k_means = fit_kmeans(cluster_data, n_cluster*n_cluster_multiplier)

    cluster_result_count = defaultdict(int)
    for label in k_means.labels_:
        cluster_result_count[label] += 1

    RS_idx = [idx for key in cluster_result_count.keys() if cluster_result_count[key] == 1
          for idx, label in enumerate(k_means.labels_) if key == label]

    for idx in sorted(RS_idx, reverse=True):
        RS.append(cluster_data.pop(idx))

    k_means = fit_kmeans(cluster_data, n_cluster)
    
    for label, point in zip(k_means.labels_, cluster_data):
        if label not in DS:
            DS[label] = {
                "N": [data_dict_reversed[tuple(point)]],
                "SUM": point,
                "SUMSQ": point ** 2
            }
        else:
            DS[label]["N"].append(data_dict_reversed[tuple(point)])
            DS[label]["SUM"] += point
            DS[label]["SUMSQ"] += point ** 2

    if len(RS) > 0:
        n_clusters_RS = 1 if len(RS) == 1 else len(RS) - 1
        k_means_RS = KMeans(n_clusters=n_clusters_RS).fit(RS)

        cluster_result_count = {}
        for label in k_means_RS.labels_:
            cluster_result_count[label] = cluster_result_count.get(label, 0) + 1

        RS_temp_idx = [k for k in cluster_result_count.keys() if cluster_result_count[k] == 1]

        if RS_temp_idx:
            RS_idx = [list(k_means_RS.labels_).index(k) for k in RS_temp_idx]

        cluster_pair_RS = tuple(zip(k_means_RS.labels_, RS))
        CS = {}

        for pair in cluster_pair_RS:
            if pair[0] not in RS_temp_idx:
                if pair[0] not in CS:
                    CS[pair[0]] = {
                        "N": [data_dict_reversed[tuple(pair[1])]],
                        "SUM": pair[1],
                        "SUMSQ": pair[1] ** 2
                    }
                    continue
                CS[pair[0]]["N"].append(data_dict_reversed[tuple(pair[1])])
                CS[pair[0]]["SUM"] += pair[1]
                CS[pair[0]]["SUMSQ"] += pair[1] ** 2

        RS_update = [RS[i] for i in reversed(sorted(RS_idx))]
        RS = copy.deepcopy(RS_update)

    result.append(f"Round 1: {sum([len(DS[cluster]['N']) for cluster in DS])},{len(CS)},{sum([len(CS[cluster]['N']) for cluster in CS])},{len(RS)}\n")


    for j in range(4):

        if j == 3:
            cluster_data = data[cluster_len * 4:]
        else:
            start_index = cluster_len * (j + 1)
            end_index = cluster_len * (j + 2)
            cluster_data = data[start_index:end_index]

        DS_idx = set()

        for i, point in enumerate(cluster_data):
            min_dist = sys.maxsize
            cst = 0

            for cluster in DS:
                dist = mahalanobis_distance(point, DS[cluster])

                if min_dist > dist:
                    min_dist = dist
                    cst = cluster

            if min_dist < 2 * (len(point) ** (1 / 2)):
                DS_idx.add(i)
                DS[cst]["N"].append(data_dict_reversed[tuple(point)])
                DS[cst]["SUM"] += point
                DS[cst]["SUMSQ"] += point ** 2


        if len(CS) > 0:
            CS_idx = set()

            for i, point in enumerate(cluster_data):
                if i in DS_idx:
                    continue
                
                min_dist = sys.maxsize
                cst = 0

                for cluster in CS:
                    dist = mahalanobis_distance(point, CS[cluster])

                    if min_dist > dist:
                        min_dist = dist
                        cst = cluster

                if min_dist < 2 * (len(point) ** (1 / 2)):
                    CS_idx.add(i)
                    CS[cst]["N"].append(data_dict_reversed[tuple(point)])
                    CS[cst]["SUM"] += point
                    CS[cst]["SUMSQ"] += point ** 2

        for i in range(len(cluster_data)):
            if i not in DS_idx and i not in CS_idx:
                RS.append(cluster_data[i])


        if len(RS) > 0:
            k_means = fit_kmeans(RS, 1 if len(RS) == 1 else len(RS) - 1)

        CS_set = set(CS.keys())
        RS_set = set(k_means.labels_)
        intersection = CS_set.intersection(RS_set)
        union = CS_set.union(RS_set)
        change = {}

        for intersect in intersection:
            random_int = random.randint(100, len(cluster_data))
            while random_int in union:
                random_int = random.randint(100, len(cluster_data))

            change[intersect] = random_int
            union.add(random_int)

        new_labels = [change[label] if label in change else label for label in k_means.labels_]

        cluster_result_count = defaultdict(int)
        for label in new_labels:
            cluster_result_count[label] += 1

        RS_temp_idx = [k for k in cluster_result_count if cluster_result_count[k] == 1]
        RS_idx = [new_labels.index(k) for k in RS_temp_idx]
        cluster_pair = tuple(zip(new_labels, RS))

        for pair in cluster_pair:
            if pair[0] in RS_temp_idx:
                continue
            if pair[0] not in CS:
                CS[pair[0]] = {
                    "N": [data_dict_reversed[tuple(pair[1])]],
                    "SUM": pair[1],
                    "SUMSQ": pair[1] ** 2
                }
                continue
            CS[pair[0]]["N"].append(data_dict_reversed[tuple(pair[1])])
            CS[pair[0]]["SUM"] += pair[1]
            CS[pair[0]]["SUMSQ"] += pair[1] ** 2

        RS_update = [RS[i] for i in reversed(sorted(RS_idx))]
        RS = copy.deepcopy(RS_update)
        
        while True:
            original_cluster = set(CS.keys())
            merge_occurred = False

            for compare in itertools.product(CS.keys(), repeat=2):
                if compare[0] != compare[1]:
                    dist = mahalanobis_distance_cluster(CS[compare[0]], CS[compare[1]])

                    if dist < 2 * (len(CS[compare[0]]["SUM"]) ** (1 / 2)):
                        CS[compare[0]]["N"] += CS[compare[1]]["N"]
                        CS[compare[0]]["SUM"] += CS[compare[1]]["SUM"]
                        CS[compare[0]]["SUMSQ"] += CS[compare[1]]["SUMSQ"]
                        CS.pop(compare[1])
                        merge_occurred = True
                        break

            new_cluster = set(CS.keys())

            if not merge_occurred or new_cluster == original_cluster:
                break


        CS_cluster = list(CS.keys())

        if j == 3 and len(CS) > 0:
            for cluster_CS in CS_cluster:
                min_dist, cst = sys.maxsize, 0

                for cluster in DS:
                    dist = mahalanobis_distance_cluster(CS[cluster_CS], DS[cluster])

                    if min_dist > dist:
                        min_dist, cst = dist, cluster

                if min_dist < 2 * len(CS[cluster_CS]["SUM"]) ** (1 / 2):
                    DS[cst]["N"] = DS[cst]["N"] + CS[cluster_CS]["N"]
                    DS[cst]["SUM"] += CS[cluster_CS]["SUM"]
                    DS[cst]["SUMSQ"] += CS[cluster_CS]["SUMSQ"]
                    CS.pop(cluster_CS)

        result.append(f"Round {j + 2}: {sum([len(DS[cluster]['N']) for cluster in DS])},{len(CS)},{sum([len(CS[cluster]['N']) for cluster in CS])},{len(RS)}\n")

    result.append("\nThe clustering results:\n")

    for cluster in DS:
        DS[cluster]["N"] = {p for p in DS[cluster]["N"]}

    for cluster in CS:
        CS[cluster]["N"] = {p for p in CS[cluster]["N"]}

    for p in range(len(data_dict)):
        if p in RS_set:
            result.append(f"{p},-1\n")
        else:
            assigned_cluster_DS = next((cluster for cluster in DS if p in DS[cluster]["N"]), None)
            assigned_cluster_CS = next((cluster for cluster in CS if p in CS[cluster]["N"]), None)

            if assigned_cluster_DS is not None:
                result.append(f"{p},{assigned_cluster_DS}\n")
            elif assigned_cluster_CS is not None:
                result.append(f"{p},-1\n")

    f = open(output_file, 'w')
    f.writelines(result)
    f.close()

    print(f"Duration: {time.time() - start_time}")

if __name__ == "__main__":
    main()