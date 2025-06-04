import glob
import os
import pandas as pd
from utils_all import get_sorted_folders
  
from config import Config
from evaluate_failure_prediction_selforacle import evaluate_failure_prediction,evaluate_fp_and_tn,calculate_threshold,evaluate_fp_and_tn_new

if __name__ == '__main__':
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    cfg = Config()
    cfg.from_pyfile("config_my.py")
  
  
    simulations = get_sorted_folders(cfg.ABNORMAL_SIMULATION_DIR)
    for am in ['max','mean']:
        if am == 'mean':
            threshold = 7.073331920882209
        else:
            threshold = 11.514808614343803
  
  
  
        for sim in simulations:
            evaluate_failure_prediction(cfg,
                                        simulation_name=sim,
                                        aggregation_method=am,
                                        threshold = threshold)