import gp3tools as gp3

x = gp3.simulate_gazepoint_cluster_timecourse_data(n_subjects=12, n_time=50)
result = gp3.run_gazepoint_cluster_permutation(x, n_permutations=99)
print(result["clusters"])
fig = gp3.plot_gazepoint_cluster_results(result)
fig.savefig("timecourse_clusters.png", dpi=120, bbox_inches="tight")
