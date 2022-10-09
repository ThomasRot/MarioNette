import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from argparse import Namespace
import os
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import numpy as np
import joblib
from PIL import Image
import random

from torchvision.transforms.functional import to_tensor, crop

N_NEIGHBORS = 24
DISPLAY_CENTROIDS = True

label_list_space_invaders = ["no_label"] + [f"{side}_score" for side in ['left', 'right']] + [f"enemy_{idx}"
                                                                                              for idx in
                                                                                              range(6)] \
                            + ["space_ship", "player", "block", "bullet"]

label_idxs_space_invaders = [i for i, l in enumerate(label_list_space_invaders)]


def evaluate_z_what(model, fname):
    data = "/".join(fname.split("/")[:-1]).replace("train", "validation")
    files = [os.path.join(data, f) for f in os.listdir(data) if
             f.endswith('png') or f.endswith('jpg') or f.endswith('jpeg')]
    files = sorted(files,
                   key=lambda f: int(''.join(x for x in os.path.basename(f) if x.isdigit())))

    z_whats = []
    labels = []
    model.model.eval()
    with torch.no_grad():
        for fnames, imgs in get_batches(files):
            fwd_data = model.forward(imgs.to(model.device))

            what = fwd_data["im_codes"]
            pres = fwd_data["probs"]
            shift = fwd_data["shifts"]
            B, NL = pres.shape[:2]
            LZ, dim_z = what.shape[2:]

            what = what.reshape(B, NL, LZ, LZ, dim_z)
            pres = pres.reshape(B, NL, LZ, LZ)
            shift = shift.reshape(B, NL, LZ, LZ, 2)

            z_whats.append(what[pres > 0.5])
            for idx, f in enumerate(fnames):
                labels.extend(compute_labels(data, f, shift[idx][pres[idx] > 0.5]))

    model.model.train()
    z_what = torch.cat(z_whats, 0).cpu()
    labels = torch.tensor(labels)
    c = Counter(labels.tolist() if labels is not None else [])
    relevant_labels = list(c.keys())
    print(c)
    if len(c) < 2:
        return Counter()
    train_portion = 0.9
    nb_sample = int(train_portion * len(labels))
    test_x = z_what[nb_sample:]
    test_y = labels[nb_sample:]
    train_x = z_what[:nb_sample]
    train_y = labels[:nb_sample]

    results = {}
    z_what_by_game = {rl: train_x[train_y == rl] for rl in relevant_labels}
    labels_by_game = {rl: train_y[train_y == rl] for rl in relevant_labels}
    print(z_what_by_game)
    for training_objects_per_class in [1, 4, 16, 64]:
        current_train_sample = torch.cat([z_what_by_game[rl][:training_objects_per_class] for rl in relevant_labels])
        current_train_labels = torch.cat([labels_by_game[rl][:training_objects_per_class] for rl in relevant_labels])
        clf = RidgeClassifier()
        clf.fit(current_train_sample, current_train_labels)
        acc = clf.score(test_x, test_y)
        joblib.dump(clf, f"marionette_classifier_{training_objects_per_class}")
        results[f'few_shot_accuracy_with_{training_objects_per_class}'] = acc

    clf = KMeans(n_clusters=len(relevant_labels))
    y = clf.fit_predict(z_what)
    results['adjusted_mutual_info_score'] = metrics.adjusted_mutual_info_score(labels, y)
    results['adjusted_rand_score'] = metrics.adjusted_rand_score(labels, y)
    print(results)
    return results


canvas_size = 128


def get_batches(files):
    fnames = []
    imgs = []
    for f in files[:128]:
        im = Image.open(f).convert('RGB')
        w, h = im.size
        a = min(w, h) / canvas_size
        im = im.resize((int(w / a), int(h / a)), Image.LANCZOS)
        w, h = im.size
        im = im.crop(((w - canvas_size) // 2, (h - canvas_size) // 2,
                      (w - canvas_size) // 2 + canvas_size,
                      (h - canvas_size) // 2 + canvas_size))
        im = to_tensor(im)
        imgs.append(im)
        fnames.append(f)
        if len(imgs) == 8:
            yield fnames, torch.stack(imgs)
            imgs = []
            fnames = []


def compute_labels(data_path, f, z_where):
    # print("--------------")
    # print(z_where)
    bb_path = f.replace("space_like_128", "bb").replace(".png", ".csv")
    print(bb_path)
    gt_bbs = pd.read_csv(f'{bb_path}', header=None)
    labels = match_bbs(gt_bbs, z_where, label_list_space_invaders)
    # labels = [random.choice(label_idxs_space_invaders) for x, y in z_where]
    print(labels)
    # print("--------------")
    return labels


def get_labels(gt_bbs, game, boxes_batch):
    """
    Compare ground truth to boxes computed by SPACE
    """
    return match_bbs(gt_bbs, boxes_batch, label_list_for(game))


def get_labels_moving(gt_bbs, game, boxes_batch):
    """
    Compare ground truth to boxes computed by SPACE
    """
    return match_bbs(gt_bbs[gt_bbs[4] == "M"], boxes_batch, label_list_for(game))


def match_bbs(gt_bbs, boxes_batch, label_list):
    labels = []
    for bb in boxes_batch:
        label, min_dist = min(((gt_bb[5], dist(bb, gt_bb)) for gt_bb in gt_bbs.itertuples(index=False, name=None)),
                              key=lambda tup: tup[1])
        if min_dist > 0.3:
            label = "no_label"
        labels.append(label)
    return torch.LongTensor([label_list.index(label) for label in labels])


def dist(pos, gt_pos):
    pixel_pos = (pos + 1)
    gt_pos = (gt_pos[0] + gt_pos[1], gt_pos[2] + gt_pos[3])
    return (pixel_pos[0] - gt_pos[0]) ** 2 + (pixel_pos[1] - gt_pos[1]) ** 2
