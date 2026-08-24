import gp3tools as gp3

master = gp3.load_example_master()
result = gp3.reconstruct_gazepoint_binocular_pupil(master, left_col="LPMM", right_col="RPMM")
print(gp3.diagnose_gazepoint_binocular_pupil(result, left_col="LPMM", right_col="RPMM"))
