
SAO_MODELS_DIR = "/bdata/usrdata/  root/monitor_test/selforacle_rebuild/vae_ckpt"  # autoencoder-based self-assessment oracle models
TEST_SIZE = 0.2  # split of training data used for the validation set (keep it low)
FILEPATH = "/bdata/usrdata/  root/monitor_test/selforacle_rebuild/train_output_fps10_bev.csv"
NORMAL_SIMULATION_DIR = "/bdata/usrdata/  root/monitor_test/Bench2Drive/eval_bench2drive220_vad_traj/normal/evaluation"
ABNORMAL_SIMULATION_DIR = "/bdata/usrdata/  root/monitor_test/Bench2Drive/eval_bench2drive220_vad_traj/collision_need_judge_2"
THRESHOLD_DIR = "/home/  root/train_data_fps10/train"

NUM_EPOCHS_SAO_MODEL = 15  # training epochs for the autoencoder-based self-assessment oracle
SAO_LATENT_DIM = 2  # dimension of the latent space
LOSS_SAO_MODEL = "MSE"  # "VAE"|"MSE" objective function for the autoencoder-based self-assessment oracle
# DO NOT TOUCH THESE
SAO_BATCH_SIZE = 128
SAO_LEARNING_RATE = 0.0001
SAVE_BEST_ONLY = True

# adaptive anomaly detection settings
UNCERTAINTY_TOLERANCE_LEVEL = 0.00328
CTE_TOLERANCE_LEVEL = 2.5
IMPROVEMENT_RATIO = 1
