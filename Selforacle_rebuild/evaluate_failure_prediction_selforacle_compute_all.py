import glob
import os
import pandas as pd
from utils_all import get_sorted_folders
# from utils_all import get_sorted_folders,calculate_threshold
from config import Config
from evaluate_failure_prediction_selforacle import evaluate_failure_prediction,evaluate_fp_and_tn,calculate_threshold,evaluate_fp_and_tn_new

if __name__ == '__main__':
    # cfg = Config()
    # cfg.from_pyfile("config_my.py")
    # # path = "/home/  root/ase22/selforacle_rebuild/test_data_results_tn_fp_fps5.csv"
    # #simulations includes all tracks
    # simulations = get_sorted_folders(cfg.NORMAL_SIMULATION_DIR)
    # for am in ['mean', 'max']:
    #     # threshold = calculate_threshold(path, am)
    #     # print(threshold)
    #     # for sim in simulations:
    #     #     sim = os.path.basename(sim)
    #     path = "/bdata/usrdata/  root/monitor_test/selforacle_rebuild/self_normal_log_bev.csv"
    #     evaluate_fp_and_tn_new(path,
    #                     # simulation_name=sim,
    #                     aggregation_method=am)
    #         # evaluate_failure_prediction(cfg,
    #         #                             simulation_name=sim,
    #         #                             aggregation_method=am,
    #         #                             threshold = threshold)
    cfg = Config()
    cfg.from_pyfile("config_my.py")
    # path = "/home/  root/ase22/selforacle_rebuild/test_data_results_tn_fp_fps5.csv"
    #simulations includes all tracks
    simulations = get_sorted_folders(cfg.ABNORMAL_SIMULATION_DIR)
    for am in ['max','mean']:
        if am == 'mean':
            threshold = 7.073331920882209
        else:
            threshold = 11.514808614343803
        # evaluate_fp_and_tn_new(cfg,
        #                 # simulation_name=sim,
        #                 aggregation_method=am)
        for sim in simulations:
            evaluate_failure_prediction(cfg,
                                        simulation_name=sim,
                                        aggregation_method=am,
                                        threshold = threshold)