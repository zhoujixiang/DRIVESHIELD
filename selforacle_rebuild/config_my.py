  
  
  
SAO_MODELS_DIR = "/bdata/usrdata/root/monitor_test/selforacle_rebuild/vae_ckpt"   
TEST_SIZE = 0.2   
FILEPATH = "/bdata/usrdata/root/monitor_test/selforacle_rebuild/train_output_fps10_bev.csv"
  
  
  
  
  
  
  
  

  
  
  
  
  
  
  
  
  
  

  
  
NORMAL_SIMULATION_DIR = "/bdata/usrdata/root/monitor_test/Bench2Drive/eval_bench2drive220_vad_traj/normal/evaluation"
ABNORMAL_SIMULATION_DIR = "/bdata/usrdata/root/monitor_test/Bench2Drive/eval_bench2drive220_vad_traj/collision_need_judge_2"
THRESHOLD_DIR = "/home/root/train_data_fps10/train"
  
  
  
  
  
  
  

  
NUM_EPOCHS_SAO_MODEL = 15   
SAO_LATENT_DIM = 2   
LOSS_SAO_MODEL = "MSE"   
  
SAO_BATCH_SIZE = 128
SAO_LEARNING_RATE = 0.0001
SAVE_BEST_ONLY = True

  
UNCERTAINTY_TOLERANCE_LEVEL = 0.00328
CTE_TOLERANCE_LEVEL = 2.5
IMPROVEMENT_RATIO = 1
