import gp3tools as gp3

master = gp3.load_example_master()
print(gp3.check_sampling_rate(master, time_col="TIME", group_cols=["subject"]).head())
print(gp3.summarise_gazepoint_pupil(master, pupil_col="pupil", group_cols=["condition"]))
print(
    gp3.compute_gazepoint_aoi_entropy(master, aoi_col="aoi_current", group_cols=["subject"]).head()
)
fig = gp3.plot_gazepoint_heatmap(master)
fig.savefig("quickstart_heatmap.png", dpi=120, bbox_inches="tight")
