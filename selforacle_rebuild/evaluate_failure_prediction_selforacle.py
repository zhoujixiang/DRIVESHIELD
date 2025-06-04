import csv
import gc
import os

import numpy as np
import pandas as pd
from keras import backend as K
from tqdm import tqdm

import utils_all as utils
import utils_vae
from vae import normalize_and_reshape
from utils_all import load_all_images, append_results_to_csv
from scipy.stats import gamma

def load_or_compute_losses(anomaly_detector, dataset, cached_file_name, delete_cache):
    losses = []

    current_path = os.getcwd()
    cache_path = os.path.join(current_path, 'cache', cached_file_name + '.npy')

    if delete_cache:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("delete_cache = True. Removed losses cache file " + cached_file_name)
    try:
        losses = np.load(cache_path)
        losses = losses.tolist()
        print("Found losses for " + cached_file_name)
        return losses
    except FileNotFoundError:
        print("Losses for " + cached_file_name + " not found. Computing...")

        for x in tqdm(dataset):
            # x = utils.resize(x)
            x = normalize_and_reshape(x)

            loss = anomaly_detector.test_on_batch(x)[1]  # total loss
            losses.append(loss)

        np_losses = np.array(losses)
        np.save(cache_path, np_losses)
        print("Losses for " + cached_file_name + " saved.")

    return losses

def calculate_threshold(cfg, path, aggregation_method):

    dataset = load_all_images(path)
    data_df_nominal = pd.read_csv(path)
    name_of_autoencoder = "MSE-latent2"

    vae = utils_vae.load_vae_by_name(name_of_autoencoder)

    name_of_losses_file = "MSE-latent2-selforacle"

    original_losses = load_or_compute_losses(vae, dataset, name_of_losses_file, delete_cache=True)
    data_df_nominal['loss'] = original_losses
    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal,
                                                                                aggregation_method)
    return threshold

def evaluate_fp_and_tn_new(path, aggregation_method):
    dataset = load_all_images(path)
    data_df_nominal = pd.read_csv(path)
    name_of_autoencoder = "MSE-latent2"

    vae = utils_vae.load_vae_by_name(name_of_autoencoder)

    name_of_losses_file = "MSE-latent2-selforacle"

    original_losses = load_or_compute_losses(vae, dataset, name_of_losses_file, delete_cache=True)
    data_df_nominal['loss'] = original_losses
    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal,
                                                                                 aggregation_method)
    row_to_append = {
        "path": path,
        "false_positive_windows": false_positive_windows,
        "true_negative_windows": true_negative_windows,
        "threshold": threshold,
        "aggregation_method":aggregation_method
    }

    append_results_to_csv(row_to_append, output_csv = "vad_results_tn_fp.csv")

def evaluate_fp_and_tn(cfg, simulation_name, aggregation_method):
    path = os.path.join(cfg.NORMAL_SIMULATION_DIR,
                        simulation_name,
                        'driving_log_bev.csv')
    dataset = load_all_images(path)
    data_df_nominal = pd.read_csv(path)
    name_of_autoencoder = "MSE-latent2"

    vae = utils_vae.load_vae_by_name(name_of_autoencoder)

    name_of_losses_file = "MSE-latent2-selforacle" + '-' \
                          + simulation_name.replace("/", "-").replace("\\", "-")

    original_losses = load_or_compute_losses(vae, dataset, name_of_losses_file, delete_cache=True)
    print(original_losses)
    data_df_nominal['loss'] = original_losses
    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal,
                                                                                 aggregation_method)
    row_to_append = {
        "path": path,
        "false_positive_windows": false_positive_windows,
        "true_negative_windows": true_negative_windows,
        "threshold": threshold,
        "aggregation_method":aggregation_method
    }

    append_results_to_csv(row_to_append, output_csv = "test_data_results_tn_fp.csv")

def evaluate_failure_prediction(cfg, simulation_name, aggregation_method, threshold):
    name_of_autoencoder = "MSE-latent2"

    vae = utils_vae.load_vae_by_name(name_of_autoencoder)

    path = os.path.join(cfg.ABNORMAL_SIMULATION_DIR,
                        simulation_name,
                        'driving_log_bev.csv')
    
    dataset = load_all_images(path)

    data_df_anomalous = pd.read_csv(path)

    name_of_losses_file = "MSE-latent2-selforacle" + '-' + simulation_name.replace("/", "-").replace("\\",
                                                                                                                "-")
    new_losses = load_or_compute_losses(vae, dataset, name_of_losses_file, delete_cache=True)

    data_df_anomalous['loss'] = new_losses

    for seconds in range(1, 2):
        # true_positive_windows, false_negative_windows, undetectable_windows = compute_tp_and_fn(data_df_anomalous,
        #                                                                                         new_losses,
        #                                                                                         threshold,
        #                                                                                         seconds,
        #                                                                                         aggregation_method)
        true_positive_windows, false_negative_windows, undetectable_windows = compute_tp_and_fn_0_5s(data_df_anomalous,
                                                                                                new_losses,
                                                                                                threshold,
                                                                                                aggregation_method)
        row_to_append = {
            "path": path,
            "true_positive_windows": true_positive_windows,
            "false_negative_windows": false_negative_windows,
            "undetectable_windows":undetectable_windows,
            "seconds": seconds,
            "aggregation_method": aggregation_method
        }

        append_results_to_csv(row_to_append, output_csv = "vad_tp_fn.csv")
    del vae
    K.clear_session()
    gc.collect()


def compute_tp_and_fn(data_df_anomalous, losses_on_anomalous, threshold, seconds_to_anticipate,
                      aggregation_method='mean'):
    print("time to misbehaviour (s): %d" % seconds_to_anticipate)

    # only occurring when conditions == unexpected
    true_positive_windows = 0
    false_negative_windows = 0
    undetectable_windows = 0

    number_frames_anomalous = len(data_df_anomalous)
    fps_anomalous = 5  # only for icse20 configurations

    crashed_anomalous = data_df_anomalous['crashed']
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_crashed_sequences = []
    for idx, item in enumerate(crashed_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous:  # we have reached the end of the file
            continue

        if crashed_anomalous_in_anomalous_conditions[idx] == 0 and crashed_anomalous_in_anomalous_conditions[
            idx + 1] == 1:
            first_index_crash = idx + 1
            all_first_frame_position_crashed_sequences.append(first_index_crash)
            # print("first_index_crash: %d" % first_index_crash)

    print("identified %d crash(es)" % len(all_first_frame_position_crashed_sequences))
    print(all_first_frame_position_crashed_sequences)
    frames_to_reassign = fps_anomalous * seconds_to_anticipate  # start of the sequence

    # frames_to_reassign_2 = 1  # first frame before the failure
    frames_to_reassign_2 = fps_anomalous * (seconds_to_anticipate - 1)  # first frame n seconds before the failure

    reaction_window = pd.Series()

    for item in all_first_frame_position_crashed_sequences:
        print("analysing failure %d" % item)
        if item - frames_to_reassign < 0:
            undetectable_windows += 1
            continue

        # the detection window overlaps with a previous crash; skip it
        if crashed_anomalous_in_anomalous_conditions.loc[
           item - frames_to_reassign: item - frames_to_reassign_2].sum() > 2:
            print("failure %d cannot be detected at TTM=%d" % (item, seconds_to_anticipate))
            undetectable_windows += 1
        else:
            crashed_anomalous_in_anomalous_conditions.loc[item - frames_to_reassign: item - frames_to_reassign_2] = 1
            reaction_window = reaction_window.append(
                crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2])

            print("frames between %d and %d have been labelled as 1" % (
                item - frames_to_reassign, item - frames_to_reassign_2))
            print("reaction frames size is %d" % len(reaction_window))

            sma_anomalous = pd.Series(losses_on_anomalous)
            sma_anomalous = sma_anomalous.iloc[reaction_window.index.to_list()]
            assert len(reaction_window) == len(sma_anomalous)

            # print(sma_anomalous)

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = sma_anomalous.mean()
            elif aggregation_method == "max":
                aggregated_score = sma_anomalous.max()

            print("threshold %s\tmean: %s\tmax: %s" % (
                str(threshold), str(sma_anomalous.mean()), str(sma_anomalous.max())))

            if aggregated_score >= threshold:
                true_positive_windows += 1
            elif aggregated_score < threshold:
                false_negative_windows += 1

        print("failure: %d - true positives: %d - false negatives: %d - undetectable: %d" % (
            item, true_positive_windows, false_negative_windows, undetectable_windows))

    assert len(all_first_frame_position_crashed_sequences) == (
            true_positive_windows + false_negative_windows + undetectable_windows)
    return true_positive_windows, false_negative_windows, undetectable_windows

def compute_fp_and_tn(data_df_nominal, aggregation_method):
    # when conditions == nominal I count only FP and TN

    # if condition == "icse20":
    #     fps_nominal = 15  # only for icse20 configurations
    # else:
    #     number_frames_nominal = pd.Series.max(data_df_nominal['frameId'])
    #     simulation_time_nominal = pd.Series.max(data_df_nominal['time'])
    #     fps_nominal = number_frames_nominal // simulation_time_nominal
    # data_df_nominal['loss'] = data_df_nominal['loss'] - data_df_nominal['loss'].min() + 1e-6
    fps_nominal = 5  # only for icse20 configurations
    num_windows_nominal = len(data_df_nominal) // fps_nominal
    if len(data_df_nominal) % fps_nominal != 0:
        num_to_delete = len(data_df_nominal) - (num_windows_nominal * fps_nominal)
        data_df_nominal = data_df_nominal[:-num_to_delete]
    losses = pd.Series(data_df_nominal['loss'])
    sma_nominal = losses.rolling(fps_nominal, min_periods=1).mean()
    list_aggregated = []

    for idx, loss in enumerate(sma_nominal):

        if idx > 0 and idx % fps_nominal == 0:

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()

            elif aggregation_method == "max":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

            list_aggregated.append(aggregated_score)

        elif idx == len(sma_nominal) - 1:

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
            elif aggregation_method == "max":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

            list_aggregated.append(aggregated_score)
    assert len(list_aggregated) == num_windows_nominal

    import matplotlib.pyplot as plt
    import scipy.stats as stats

    losses = np.array(list_aggregated)
    losses = losses[losses > 0]  

    plt.figure(figsize=(8, 6))
    plt.hist(losses, bins=50, density=True, alpha=0.6, color='b', label="Empirical Distribution")

    shape, loc, scale = stats.gamma.fit(losses, floc=0)
    x = np.linspace(min(losses), max(losses), 100)
    pdf_fitted = stats.gamma.pdf(x, shape, loc=loc, scale=scale)

    plt.plot(x, pdf_fitted, 'r-', label="Fitted Gamma Distribution")
    plt.xlabel("Loss Value")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Histogram of Losses and Fitted Gamma Distribution")
    plt.savefig("Gamma Distribution plot.png", dpi=300, bbox_inches="tight")
    plt.show()
    threshold = get_threshold(list_aggregated, conf_level=0.74)

    false_positive_windows = len([i for i in list_aggregated if i > threshold])
    true_negative_windows = len([i for i in list_aggregated if i <= threshold])

    assert false_positive_windows + true_negative_windows == num_windows_nominal
    print("false positives: %d - true negatives: %d" % (false_positive_windows, true_negative_windows))

    return false_positive_windows, true_negative_windows, threshold

def get_threshold(losses, conf_level=0.95):
    print("Fitting reconstruction error distribution using Gamma distribution")

    # removing zeros
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)

    print("Creating threshold using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    print('threshold: ' + str(t))
    return t


def compute_tp_and_fn_0_5s(data_df_anomalous, losses_on_anomalous, threshold,
                      aggregation_method='mean'):
    # print("time to misbehaviour (s): %d" % seconds_to_anticipate)

    # only occurring when conditions == unexpected
    true_positive_windows = 0
    false_negative_windows = 0
    undetectable_windows = 0

    number_frames_anomalous = len(data_df_anomalous)
    fps_anomalous = 5  # only for icse20 configurations

    crashed_anomalous = data_df_anomalous['crashed']
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_crashed_sequences = []
    for idx, item in enumerate(crashed_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous:  # we have reached the end of the file
            continue

        if crashed_anomalous_in_anomalous_conditions[idx] == 0 and crashed_anomalous_in_anomalous_conditions[
            idx + 1] == 1:
            first_index_crash = idx + 1
            all_first_frame_position_crashed_sequences.append(first_index_crash)
            # print("first_index_crash: %d" % first_index_crash)

    print("identified %d crash(es)" % len(all_first_frame_position_crashed_sequences))
    print(all_first_frame_position_crashed_sequences)
    frames_to_reassign = fps_anomalous * 2  # start of the sequence

    # frames_to_reassign_2 = 1  # first frame before the failure
    frames_to_reassign_2 = 5  # first frame n seconds before the failure

    reaction_window = pd.Series()

    for item in all_first_frame_position_crashed_sequences:
        print("analysing failure %d" % item)
        if item - frames_to_reassign < 0:
            undetectable_windows += 1
            continue

        # the detection window overlaps with a previous crash; skip it
        if crashed_anomalous_in_anomalous_conditions.loc[
           item - frames_to_reassign: item - frames_to_reassign_2].sum() > 2:
            print("failure %d cannot be detected at TTM=%d" % (item, 0.5))
            undetectable_windows += 1
        else:
            crashed_anomalous_in_anomalous_conditions.loc[item - frames_to_reassign: item - frames_to_reassign_2] = 1
            reaction_window = reaction_window.append(
                crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2])

            print("frames between %d and %d have been labelled as 1" % (
                item - frames_to_reassign, item - frames_to_reassign_2))
            print("reaction frames size is %d" % len(reaction_window))

            sma_anomalous = pd.Series(losses_on_anomalous)
            sma_anomalous = sma_anomalous.iloc[reaction_window.index.to_list()]
            assert len(reaction_window) == len(sma_anomalous)

            # print(sma_anomalous)

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = sma_anomalous.mean()
            elif aggregation_method == "max":
                aggregated_score = sma_anomalous.max()

            print("threshold %s\tmean: %s\tmax: %s" % (
                str(threshold), str(sma_anomalous.mean()), str(sma_anomalous.max())))

            if aggregated_score >= threshold:
                true_positive_windows += 1
            elif aggregated_score < threshold:
                false_negative_windows += 1

        print("failure: %d - true positives: %d - false negatives: %d - undetectable: %d" % (
            item, true_positive_windows, false_negative_windows, undetectable_windows))

    assert len(all_first_frame_position_crashed_sequences) == (
            true_positive_windows + false_negative_windows + undetectable_windows)
    return true_positive_windows, false_negative_windows, undetectable_windows