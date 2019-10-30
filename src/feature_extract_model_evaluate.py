"""
Evaluate model on a specified dataset

(1) generate pairs (match/dismatch)
(2) read pairs
(3) evaluate

"""

import os
import cv2
import sys
import time 
import math
import random
import argparse
import numpy as np   
import tensorflow as tf
from itertools import combinations
from scipy import interpolate
from scipy.optimize import brentq
from sklearn.model_selection import KFold
from sklearn import metrics
from .get_distance import GetDistanceTensorflow, get_distance

def read_paris_file(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        # NOTE first line is properties of the pairs file
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def get_pairs(pairs):
    """
    pairs: numpy.array contain pairs lists
    return: path_list (list), is_same_list (list)
    """
    num_skipped_pairs = 0
    path_list = []
    is_same_list = []
    for pair in pairs:
        if len(pair) == 3:
            path_1 = pair[1]
            path_2 = pair[2]
            is_same = True
        elif len(pair) == 4:
            path_1 = pair[1]
            path_2 = pair[3]
            is_same = False
        else:
            print('Invalid length of pair list! Will ignore.')
            print('Length of pair list : {}'.format(len(pair)))
            continue
        if os.path.exists(path_1) and os.path.exists(path_2):
            path_list += (path_1, path_2)
            is_same_list.append(is_same)
        else:
            if not os.path.exists(path_1):
                print('Could not find {}.'.format(ptah_1))
            if not os.path.exists(path_2):
                print('Could not find {}.'.format(path_2))
            num_skipped_pairs += 1
    if num_skipped_pairs > 0:
        print('Skipped %d image pairs' % num_skipped_pairs)
    return path_list, is_same_list

def evaluate(embeddings, is_same_list, thresholds, far_target, batch_size, gdt, num_folds=10):
    """
    --------------------------------------------------------------------
    Argument :
        embeddings   : numpy.ndarray with shape=(num_pairs*2, emb_dim)
        is_same_list : list with length=num_pairs
        thresholds   : distance thresholds of two embeddings
        num_folds    : number of cross-validation folds
    --------------------------------------------------------------------
    Return :
        tpr      : true positive rate
        fpr      : false positive rate
        accuracy : accuracy
        val      : validation rate at given FAR
        val_std  : std of validation
        far      : false accept rate
    --------------------------------------------------------------------
    """
    print('    * len(is_same_list) : ', len(is_same_list))
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    
    #=======================#
    # get distance on batch #
    #=======================#
    num_embeddings = embeddings1.shape[0]
    num_batchs = num_embeddings // batch_size
    distance_array = np.zeros((num_embeddings))

    for batch_index in range(num_batchs):
        start_index = int(batch_index * batch_size)
        end_index = min((batch_index + 1) * batch_size, num_embeddings)
        distance_array[start_index:end_index] = gdt.infer(embeddings1[start_index:end_index, :], embeddings2[start_index:end_index, :])
       
    is_nan = np.isnan(distance_array).any()
    print('    * is_nan : ', is_nan)
    print('    * distance_array with shape={} : '.format(distance_array.shape))

    #============================#
    # calaulate eveluate metrics #
    #============================#
    tpr, fpr, accuracy = calculate_roc(thresholds, distance_array, is_same_list, num_folds)
    val, val_std, far, thresh_mean = calculate_val(thresholds, distance_array, is_same_list, far_target, num_folds)
    return tpr, fpr, accuracy, val, val_std, far, thresh_mean

def calculate_roc(thresholds, distance, is_same_actual, num_folds=10, metric_type='cos'):
    """
    thresholds  : list of thresholds (list)
    embeddings1 : embedding array in the 1-st column of the pairs (numpy.ndarray)
    embeddings2 : embedding array in the 2-nd column of the pairs (numpy.ndarray)
    is_same_actual : list contains true label (list)
    num_folds : number of folds for cross-validation (int)
    metric_type : metric type to calculate distance between two embeddings (str/int)
    """
    max_threshold = 0
    min_threshold = 100

    num_pairs =  min(len(is_same_actual), distance.shape[0])
    num_thresh = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)
    is_same_actual = np.asarray(is_same_actual)

    #========================================#
    # will run num_folds with each threshold #
    #========================================#
    tpr_array = np.zeros((num_folds, num_thresh))
    fpr_array = np.zeros((num_folds, num_thresh))
    acc_array = np.zeros((num_folds))

    indices = np.arange(num_pairs)

    #=========================#
    # enumerate k-folds of CV #
    #=========================#
    """
        TODO
        (1) find the best threshold for the fold (train)
        (2) calculate metrics for the fold (validation)
    """
    for i, (train_indices, valid_indices) in enumerate(k_fold.split(indices)):
        #==================================================#
        # (1) find the best threshold for the fold (train) #
        #==================================================#
        acc_train = np.zeros((num_thresh))
        for thresh_idx, thresh in enumerate(thresholds):
            _, _, acc_train[thresh_idx] = calculate_accuracy(thresh, distance[train_indices], is_same_actual[train_indices])
        best_threshold_idx = np.argmax(acc_train)

        #================================================#
        # (2)calculate metrics for the fold (validation) #
        #================================================#
        for thresh_idx, thresh in enumerate(thresholds):
            tpr_array[i, thresh_idx], fpr_array[i, thresh_idx], _ = calculate_accuracy(thresh, distance[valid_indices], is_same_actual[valid_indices])
        _, _, acc_array[i] = calculate_accuracy(thresholds[best_threshold_idx], distance[valid_indices], is_same_actual[valid_indices])

        #=======================#
        # update best threshold #
        #=======================#
        if max_threshold < thresholds[best_threshold_idx]:
            max_threshold = thresholds[best_threshold_idx]
        if min_threshold > thresholds[best_threshold_idx]:
            min_threshold = thresholds[best_threshold_idx]

    print('    * min thresh : {}, max thresh : {}'.format(round(min_threshold, 4), round(max_threshold, 4)))

    tpr = np.mean(tpr_array, axis=0)
    fpr = np.mean(fpr_array, axis=0)

    return tpr, fpr, acc_array

def calculate_val(thresholds, distance, is_same_actual, far_target, num_folds=10, metric_type='cos'):
    num_pairs =  min(len(is_same_actual), distance.shape[0])
    num_thresh = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)
    is_same_actual = np.asarray(is_same_actual)

    val_array = np.zeros(num_folds)
    far_array = np.zeros(num_folds)
    thresh_target_list = []

    indices = np.arange(num_pairs)

    #================================================#
    # Find the threshold that gives FAR = far_target #
    #================================================#
    for i, (train_indices, valid_indices) in enumerate(k_fold.split(indices)):
        far_train = np.zeros(num_thresh)
        for thresh_idx, thresh in enumerate(thresholds):
            _, far_train[thresh_idx] = calculate_val_far(thresh, distance[train_indices], is_same_actual[train_indices])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            thresh_target = f(far_target)
        else:
            thresh_target = 0.0
        thresh_target_list.append(thresh_target)

        #====================#
        # get validation val #
        #====================#
        val_array[i], far_array[i] = calculate_val_far(thresh_target, distance[valid_indices], is_same_actual[valid_indices])

    thresh_mean = np.mean(thresh_target_list, axis=0)
    val_mean = np.mean(val_array, axis=0)
    far_mean = np.mean(far_array, axis=0)
    val_std = np.std(val_array, axis=0)

    return val_mean, val_std, far_mean, thresh_mean

def calculate_accuracy(threshold, distances, is_same_actual):
    """
    Given a some distance metrics, a threshold and ground-true label. Return acc, tpr and fpr
    """    
    is_same_predict = np.less(distances, threshold)

    tp = np.sum(np.logical_and(is_same_predict, is_same_actual)) # NOTE true posotive : predict positive, actually true.
    fp = np.sum(np.logical_and(is_same_predict, np.logical_not(is_same_actual))) # NOTE false positive : predict positive, but not true.
    tn = np.sum(np.logical_and(np.logical_not(is_same_predict), np.logical_not(is_same_actual))) # NOTE true negative : predict negitive, actually true.
    fn = np.sum(np.logical_and(np.logical_not(is_same_predict), is_same_actual)) # NOTE false negitive : predict negative, but not true.

    tpr = 0 if tp + fn == 0 else float(tp) / float(tp + fn)
    fpr = 0 if fp + tn == 0 else float(fp) / float(fp + tn)
    acc = float(tp + tn) / float(tp + tn + fp + fn)

    return tpr, fpr, acc

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = 0 if n_same == 0 else float(true_accept) / float(n_same)
    far = 0 if n_diff == 0 else float(false_accept) / float(n_diff)
    return val, far

def img_pipeline(path_list, img_size):
    num_imgs = len(path_list)
    output_img_array = np.zeros((num_imgs, img_size, img_size, 3)) 
    for i, path in enumerate(path_list):
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        img_rgb = img[:, :, ::-1]
        output_img_array[i, :, :, :] = img_rgb
    return output_img_array

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs_filename', type=str, help='pairs file to evaluate.')
    parser.add_argument('--use_model', type=str, help='feature extraction model to evaluate.', default='facenet')
    parser.add_argument('--model_path', type=str, help='model path with related use model.', default='/home/yiliu/face3/alphaiss_data_server/face/facenet/vggface20180402-114759')
    parser.add_argument('--far_target', type=float, help='target FAR(False Accept Rate) to control', default=1e-2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--image_size', type=int, default=80)
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.6)
    return parser.parse_args(argv)
    
def main(args):

    thresholds = np.arange(0, 4, 0.001)
    global gdt
    gdt = GetDistanceTensorflow(metric_type='cos', embedding_dim=args.embedding_dim, epslon=1e-8)

    print('** start evaluation pipeline.')
    print('(1) load model.\n')
    fe = FeatureExtractorCtrl(args.use_model, args.model_path, args.gpu_memory_fraction)

    print('\n(2) get pairs info.\n')
    pairs = read_paris_file(pairs_filename=args.pairs_filename)
    path_list, is_same_list = get_pairs(pairs)
    # path_list, is_same_list = get_pairs('/home/yiliu/lfw-160', pairs)
    print('    * number of pairs : {}'.format(len(is_same_list)))

    print('\n(3) feature extraction.\n')
    num_embeddings = len(is_same_list) * 2 # NOTE num_pairs * 2 (2 images per pairs)
    # num_batchs = num_embeddings // args.batch_size
    num_batchs = int(math.ceil(1.0 * num_embeddings / args.batch_size))

    print('    * start feature extraction.')
    start_time = time.time()
    embeddings_array = np.zeros((num_embeddings, args.embedding_dim))
    for batch_index in range(num_batchs):
        start_index = int(batch_index * args.batch_size)
        end_index = min((batch_index + 1) * args.batch_size, num_embeddings)
        batch_img_list = path_list[start_index:end_index]
        embeddings_array[start_index:end_index, :] = fe.infer(img_pipeline(batch_img_list, args.image_size))
    use_time = time.time() - start_time
    print('    * use time : {}s for {} images, {}s per 1 image'.format(round(use_time, 4), num_embeddings, round(use_time / num_embeddings, 3)))

    print('\n(4) evaluate\n')
    tpr, fpr, accuracy, val, val_std, far, thresh_mean = evaluate(embeddings_array, is_same_list, thresholds, args.far_target, args.batch_size, gdt, 10)
                                                                    
    print('\n*** result ***\n')
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    print('Threshold=%2.5f @ FAR=%2.5f' % (thresh_mean, far))
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))




