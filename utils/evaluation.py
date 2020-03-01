# Copyright 2019 Damian Schori. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
from imgaug import augmenters as iaa
import numpy as np
from skimage.measure import label
from scipy import ndimage as ndi
from skimage.morphology import dilation, erosion, watershed, remove_small_objects, remove_small_holes, binary_dilation, \
    binary_erosion, disk, star, square
from skimage.feature import peak_local_max
import rasterio as rio
from rasterio.crs import CRS
from skimage.transform import rescale, resize
from prettytable import PrettyTable
import geopandas as gpd
import matplotlib.pyplot as plt
import cv2
from rasterio.features import shapes
import pandas as pd


class Segmentation_Evaluation:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def preprocess_img(self, img):
        # Convert to uint (for imgaug):
        return img

    def preprocess_mask(self, msk):
        msk = [(msk == channel).astype(float) for channel in range(1, 3)]

        msk = np.stack(msk, axis=-1)

        background = 1 - msk.sum(axis=-1, keepdims=True)
        msk = np.concatenate((msk, background), axis=-1)
        return msk

    def apply_watershed(self, pred):
        pred = pred > 0.5
        D = ndi.distance_transform_edt(pred)
        localMax = peak_local_max(D, indices=False, min_distance=5,
                                  labels=pred)

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(D * -1, markers, mask=pred)
        return labels

    def post_process(self, pred):
        prediction = pred.squeeze() > 0.5
        shape = pred.shape[0:2]

        # put all hemp to one class to get all hemp instances
        all_hemp = np.zeros(shape)
        for i in range(prediction.shape[-1] - 1):
            all_hemp += prediction[:, :, i]

        all_hemp = all_hemp > 0

        if False:
            hemp_instances = self.apply_watershed(all_hemp)
        else:
            all_hemp = remove_small_objects(all_hemp, 30)
            hemp_instances = label(all_hemp)
        new_prediction = np.zeros((*shape, 2))

        # loop over instances and assign class acording to bigger sum:
        for i in range(1, hemp_instances.max() + 1):
            instance = hemp_instances == i
            # instance = erosion(instance, disk(9))
            instance_class1 = prediction[:, :, 0].copy()
            instance_class1[instance == 0] = 0
            instance_class2 = prediction[:, :, 1].copy()
            instance_class2[instance == 0] = 0

            if instance_class1.sum() > instance_class2.sum():
                new_prediction[:, :, 0] += instance_class1
            else:
                new_prediction[:, :, 1] += instance_class2
        for i in range(2):
            new_prediction[:, :, i] = binary_dilation(new_prediction[:, :, i], disk(5))
            new_prediction[:, :, i] = binary_erosion(new_prediction[:, :, i], disk(5))
            new_prediction[:, :, i] = remove_small_holes(new_prediction[:, :, i] > 0.5, 30)

        background = 1 - new_prediction.sum(axis=-1, keepdims=True)
        new_prediction = np.concatenate((new_prediction, background), axis=-1)

        return new_prediction.astype(float)

    def majority_vote2(self, predictions_on_all_dates, pred):
        ''' 
        predictions_on_all_dates: predictions on 3 dates of same window in shape: (dates, heigth, width, classes)
        pred: prediction on which to apply the voting in shape: (heigth, width, classes)
        '''

        predicted_hemp_instances = []
        for i in range(2):
            instances = label(pred[:, :, i])
            for v in range(1, instances.max() + 1):
                # skip small objects:
                if np.sum(instances == v) < 50:
                    continue
                predicted_hemp_instances.append((instances == v).astype('bool'))

        if len(predicted_hemp_instances) == 0:
            no_instances = np.zeros((*pred.shape[0:2], 3))
            no_instances[:, :, 2] = 1.0
            return no_instances

        predicted_hemp_instances = np.stack(predicted_hemp_instances, axis=0)

        # loop over all dates:
        for i, prediction_at_date in enumerate(predictions_on_all_dates):
            max_values = [0 for _ in range(2)]

        new_prediction = np.zeros((*pred.shape[0:2], 2))

        for i, instance in enumerate(predicted_hemp_instances):
            max_values = [0 for _ in range(2)]
            # print(max_values)
            classes = ['class0', 'class1']
            for j, c in enumerate(classes):
                tmp = np.sum(predictions_on_all_dates[:, :, :, j], axis=0)
                # tmp = binary_dilation(tmp, disk(9))
                tmp[instance == False] = 0
                max_values[j] = tmp.max()
            # print(max_values)
            # print(np.argmax(max_values))
            new_prediction[:, :, np.argmax(max_values)] += instance
            # print(votes)

        for i in range(2):
            new_prediction[:, :, i] = binary_dilation(new_prediction[:, :, i], disk(5))
            new_prediction[:, :, i] = binary_erosion(new_prediction[:, :, i], disk(5))
            # new_prediction[:, :, i] = remove_small_holes(new_prediction[:, :, i]>0.5, 20)

        background = 1 - new_prediction.sum(axis=-1, keepdims=True)
        new_prediction = np.concatenate((new_prediction, background), axis=-1)
        # print(new_prediction.max())
        return np.clip(new_prediction, a_min=0.0, a_max=1.0)

    def majority_vote(self, predictions_on_all_dates, pred):
        ''' 
        predictions_on_all_dates: predictions on 3 dates of same window in shape: (dates, heigth, width, classes)
        pred: prediction on which to apply the voting in shape: (heigth, width, classes)
        '''
        # TODO generalize and expand to n classes and dates

        predicted_hemp_instances = []
        for i in range(2):
            instances = label(pred[:, :, i])
            for v in range(1, instances.max() + 1):
                # skip small objects:
                if np.sum(instances == v) < 50:
                    continue
                predicted_hemp_instances.append((instances == v).astype('bool'))

        if len(predicted_hemp_instances) == 0:
            no_instances = np.zeros((*pred.shape[0:2], 3))
            no_instances[:, :, 2] = 1.0
            return no_instances

        predicted_hemp_instances = np.stack(predicted_hemp_instances, axis=0)

        new_prediction = np.zeros((*pred.shape[0:2], 2))

        for i, instance in enumerate(predicted_hemp_instances):
            max_values = [0 for _ in range(2)]
            # print(max_values)
            classes = ['class0', 'class1']
            for j, c in enumerate(classes):
                tmp = np.sum(predictions_on_all_dates[:, :, :, j], axis=0)
                # tmp = binary_dilation(tmp, disk(9))
                tmp[instance == False] = 0
                max_values[j] = tmp.max()
            # print(max_values)
            # print(np.argmax(max_values))
            new_prediction[:, :, np.argmax(max_values)] += instance
            # print(votes)

        for i in range(2):
            new_prediction[:, :, i] = binary_dilation(new_prediction[:, :, i], disk(5))
            new_prediction[:, :, i] = binary_erosion(new_prediction[:, :, i], disk(5))
            # new_prediction[:, :, i] = remove_small_holes(new_prediction[:, :, i]>0.5, 20)

        background = 1 - new_prediction.sum(axis=-1, keepdims=True)
        new_prediction = np.concatenate((new_prediction, background), axis=-1)
        # print(new_prediction.max())
        return np.clip(new_prediction, a_min=0.0, a_max=1.0)

    def dice_score(self, msk, pred):
        intersection = np.logical_and(msk, pred)
        denominator = np.sum(msk) + np.sum(pred)
        dice_score = 2. * np.sum(intersection) / denominator
        return dice_score

    def iou_score(self, msk, pred):
        intersection = np.logical_and(msk, pred)
        union = np.logical_or(msk, pred)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def evaluate_on_map(self, prediction_rasters, ground_truth_shapes):
        def get_stiched_raster_pair(prediction_raster, gt_shapes):
            species_encoding = {1001: 1, 1005: 2}

            shapes = [feature.geometry for i, feature in gt_shapes.iterrows()]

            pred_image, pred_transform = rio.mask.mask(prediction_raster, shapes, crop=True)
            pred_image = np.moveaxis(pred_image, 0, -1).astype(float)

            shapes = ((row.geometry, species_encoding[row.Species]) for _, row in gt_shapes.iterrows())
            rastered_shape = rio.features.rasterize(shapes=shapes,
                                                    out_shape=pred_image.shape[0:2],
                                                    transform=pred_transform)

            gt_image = [(rastered_shape == value).astype(float) for value in [1, 2, 0]]
            gt_image = np.stack(gt_image, axis=-1)

            return pred_image, gt_image

        values = []

        for prediction, shapes in zip(prediction_rasters, ground_truth_shapes):
            pred_image, gt_image = get_stiched_raster_pair(prediction, shapes)
            values.append(self.dice_score(gt_image[:, :, 0:2] > 0.5, pred_image[:, :, 0:2] > 0.5))

        return values

    def evaluate_on_set(self, data_set, idx, apply_majority_vote=False, center_crop=False, skip_background=True):
        dates = ['20190703', '20190719', '20190822']
        scores = {}
        imgs_all = {}
        msks_all = {}
        preds_all = {}

        for date in dates:
            imgs_all[date] = {}
            msks_all[date] = {}
            preds_all[date] = {}

        # allocate images, masks and predictions
        for batch in data_set:
            images, masks, names = batch[0].numpy(), batch[1].numpy(), batch[2].numpy()
            predictions = (self.model.predict(images) > self.threshold).astype('float32')
            for img, msk, pred, name in zip(images, masks, predictions, names):
                d = name[-24:-16].decode("utf-8")
                n = name[-15:-4].decode("utf-8")
                imgs_all[d][n] = img.astype('float32')
                msks_all[d][n] = msk.astype('float32')
                preds_all[d][n] = pred.astype('float32')

        for i, date in enumerate(dates):
            iou_scores = []
            dice_scores = []
            names = []
            for key in idx:
                img, msk, pred = imgs_all[date][key], msks_all[date][key], preds_all[date][key]

                if apply_majority_vote:
                    preds = np.stack([preds_all[dates[0]][key], preds_all[dates[1]][key], preds_all[dates[2]][key]])
                    pred = self.majority_vote(preds, preds[i])

                if center_crop:
                    msk = msk[32:-32, 32:-32]
                    pred = pred[32:-32, 32:-32]

                # skip background in calculation
                if skip_background:
                    msk, pred = msk[..., 0:2], pred[..., 0:2]

                iou = self.iou_score(msk, pred)
                dice = self.dice_score(msk, pred)
                if iou == 0.0:
                    iou = np.nan
                if dice == 0.0:
                    dice = np.nan
                iou_scores.append(iou)
                dice_scores.append(dice)
                names.append(key)

            scores[date] = {'names': names,
                            'iou_scores': iou_scores,
                            'dice_scores': dice_scores}
        return scores

    def create_prediction_map(self, data_interface, dataset_index, get_arr=False, apply_majority_vote=False, overlap=0):

        srcs = [rio.open(dataset.rgb_path) for dataset in data_interface.datasets]

        outer_shapefile = data_interface.datasets[dataset_index].outer_shapefile
        outer_shapefile.geometry = outer_shapefile.buffer(5)

        shapes = [feature.geometry for i, feature in outer_shapefile.iterrows()]

        out_images = []
        out_transforms = []

        for src in srcs:
            out_image, out_transform = rio.mask.mask(src, shapes, crop=True)
            out_image = np.moveaxis(out_image, 0, -1)[:, :, 0:3]
            out_images.append(out_image)
            out_transforms.append(out_transform)

        prediction = np.zeros_like(out_images[0])

        xmax_p = ((out_image.shape[1] // 384) + 0) * 384
        ymax_p = ((out_image.shape[0] // 384) + 0) * 384

        xs = list(np.arange(0, xmax_p - 0, 384 - overlap).astype(int))
        ys = list(np.arange(0, ymax_p - 0, 384 - overlap).astype(int))

        for ix, x in enumerate(xs):
            if ix % 5 == 0 and x != 0:
                pass
            for y in ys:
                if apply_majority_vote:
                    try:
                        preds = []
                        for out_image in out_images:
                            img = out_image[y:y + 384, x:x + 384].copy() / 255.0
                            pred = self.model.predict(img.reshape(1, *img.shape)).squeeze() > self.threshold
                            preds.append(pred)
                        preds = np.stack(preds, axis=0)
                        pred = self.majority_vote(preds, preds[dataset_index])
                        prediction[y + overlap // 2:y + 384 - overlap // 2,
                        x + overlap // 2:x + 384 - overlap // 2] = pred[overlap // 2:384 - overlap // 2,
                                                                   overlap // 2:384 - overlap // 2] * 255
                    except:
                        pass
                else:
                    out_image = out_images[dataset_index]
                    img = out_image[y:y + 384, x:x + 384].copy() / 255.0
                    pred = self.model.predict(img.reshape(1, *img.shape)).squeeze() > self.threshold
                    prediction[y + overlap // 2:y + 384 - overlap // 2, x + overlap // 2:x + 384 - overlap // 2] = pred[
                                                                                                                   overlap // 2:384 - overlap // 2,
                                                                                                                   overlap // 2:384 - overlap // 2] * 255

        dataset = rio.open(
            '../data/exports/prediction_{}_mv_{}_overlap_{}.tif'.format(data_interface.datasets[dataset_index].name,
                                                                        str(apply_majority_vote), str(overlap)), 'w',
            driver='GTiff',
            height=prediction.shape[0], width=prediction.shape[1],
            count=3, dtype=str(prediction.dtype),
            crs=CRS.from_epsg(32632),
            transform=out_transforms[0])

        dataset.write(np.moveaxis(prediction, -1, 0))
        dataset.close()

        if get_arr:
            return prediction


def print_results(results_new, results_base):
    t = PrettyTable(['Date', 'Dice Validation', 'Diff to Val Base', 'Dice Test', 'Diff to Test Base'])
    t.float_format = '0.2'

    new, base = results_new.copy(), results_base.copy()

    for key, value in new.items():
        new[key] *= 100
    for key, value in base.items():
        base[key] *= 100

    t.add_row(['03.07.2019', new['0703_C'],
               new['0703_C'] - base['0703_C'],
               new['0703_A'],
               new['0703_A'] - base['0703_A']])
    t.add_row(['19.07.2019', new['0719_C'],
               new['0719_C'] - base['0719_C'],
               new['0719_A'],
               new['0719_A'] - base['0719_A']])
    t.add_row(['22.08.2019', new['0822_C'],
               new['0822_C'] - base['0822_C'],
               new['0822_A'],
               new['0822_A'] - base['0822_A']])
    print(t)


def print_results_old(results_base_val, results_base_test, results_val, results_test):
    t = PrettyTable(['Date', 'Dice Validation', 'Diff to Val Base', 'Dice Test', 'Diff to Test Base'])
    t.float_format = '0.2'
    values = {'dice_base_val_20190703': np.nanmean(results_base_val['20190703']['dice_scores']) * 100,
              'dice_base_test_20190703': np.nanmean(results_base_test['20190703']['dice_scores']) * 100,
              'dice_base_val_20190719': np.nanmean(results_base_val['20190719']['dice_scores']) * 100,
              'dice_base_test_20190719': np.nanmean(results_base_test['20190719']['dice_scores']) * 100,
              'dice_base_val_20190822': np.nanmean(results_base_val['20190822']['dice_scores']) * 100,
              'dice_base_test_20190822': np.nanmean(results_base_test['20190822']['dice_scores']) * 100,
              'dice_val_20190703': np.nanmean(results_val['20190703']['dice_scores']) * 100,
              'dice_test_20190703': np.nanmean(results_test['20190703']['dice_scores']) * 100,
              'dice_val_20190719': np.nanmean(results_val['20190719']['dice_scores']) * 100,
              'dice_test_20190719': np.nanmean(results_test['20190719']['dice_scores']) * 100,
              'dice_val_20190822': np.nanmean(results_val['20190822']['dice_scores']) * 100,
              'dice_test_20190822': np.nanmean(results_test['20190822']['dice_scores']) * 100
              }

    t.add_row(['20190703', values['dice_val_20190703'],
               values['dice_val_20190703'] - values['dice_base_val_20190703'],
               values['dice_test_20190703'],
               values['dice_test_20190703'] - values['dice_base_test_20190703']])
    t.add_row(['20190719', values['dice_val_20190719'],
               values['dice_val_20190719'] - values['dice_base_val_20190719'],
               values['dice_test_20190719'],
               values['dice_test_20190719'] - values['dice_base_test_20190719']])
    t.add_row(['20190822', values['dice_val_20190822'],
               values['dice_val_20190822'] - values['dice_base_val_20190822'],
               values['dice_test_20190822'],
               values['dice_test_20190822'] - values['dice_base_test_20190822']])
    print(t)


def overlay_mask(img, mask, alpha=0.6):
    mask[:, :, 2] = 0
    out_img = cv2.add(img.astype(float), np.multiply(mask, alpha).astype(float))
    return np.clip(out_img, 0, 1)


def raster_gt(box, gt):
    transform = rio.transform.from_bounds(*box.geometry.values[0].bounds, 768, 768)
    species_encoding = {1001: 1, 1005: 2}
    inter = gpd.overlay(gt, box, how='intersection')
    shapes = ((row.geometry, species_encoding[row.Species]) for _, row in inter.iterrows())
    rastered_shape = rio.features.rasterize(shapes=shapes,
                                            out_shape=(768, 768),
                                            transform=transform)
    rgb_mask = np.zeros((768, 768, 3))
    rgb_mask[:, :, 0] = rastered_shape == 1
    rgb_mask[:, :, 1] = rastered_shape == 2
    return rgb_mask


def display_results(rgb_map, prediction_map, grid, gt_shapefile):
    fig, ax = plt.subplots(2, 6, figsize=(16, 6))
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    i = 0
    for y in range(2):
        for x in range(0, 6, 2):
            box = [feature.geometry for _, feature in grid.iloc[[i]].iterrows()]

            rgb, _ = rio.mask.mask(rgb_map, box, crop=True)
            rgb = np.moveaxis(rgb, 0, -1)[:, :, 0:3]

            rgb = resize(rgb, (768, 768))

            pred, _ = rio.mask.mask(prediction_map, box, crop=True)
            pred = np.moveaxis(pred, 0, -1)

            pred = resize(pred, (768, 768))

            pred = overlay_mask(rgb, pred)

            rastered_gt = raster_gt(grid.iloc[[i]], gt_shapefile)
            rastered_gt = overlay_mask(rgb, rastered_gt)

            ax[y, x].imshow(pred)
            ax[y, x + 1].imshow(rastered_gt)
            ax[y, x].axis('off')
            ax[y, x + 1].axis('off')
            ax[y, x].text(20, 20, "(" + grid.iloc[[i]].label.values[0] + ")", fontsize=20, verticalalignment='top',
                          bbox=props)
            i += 1
    ax[0, 0].set_title("Prediction", fontsize=20)
    ax[0, 1].set_title("Ground Truth", fontsize=20)
    ax[0, 2].set_title("Prediction", fontsize=20)
    ax[0, 3].set_title("Ground Truth", fontsize=20)
    ax[0, 4].set_title("Prediction", fontsize=20)
    ax[0, 5].set_title("Ground Truth", fontsize=20)

    plt.tight_layout()


def vectorize_prediction_map(prediction_map):
    image_1001 = prediction_map.read(1)
    image_1005 = prediction_map.read(2)
    results_1001 = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(shapes(image_1001, mask=None, transform=prediction_map.transform)))
    results_1005 = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(shapes(image_1005, mask=None, transform=prediction_map.transform)))

    geoms_1001 = list(results_1001)
    geoms_1005 = list(results_1005)

    gpd_polygonized_raster_1001 = gpd.GeoDataFrame.from_features(geoms_1001)
    gpd_polygonized_raster_1005 = gpd.GeoDataFrame.from_features(geoms_1005)
    gpd_polygonized_raster_1001['Species'] = 1001
    gpd_polygonized_raster_1005['Species'] = 1005
    df_prediction = pd.concat([gpd_polygonized_raster_1001, gpd_polygonized_raster_1005]).drop(columns=['raster_val'])
    return df_prediction


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0., xB - xA) * max(0., yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calculate_iou(df_prediction, df_ground_truth, iou_threshold=0.5):
    m = np.zeros((len(df_ground_truth), len(df_prediction)))

    for i, p_gt in enumerate(df_ground_truth['geometry'].bounds.values):
        for j, p_pred in enumerate(df_prediction['geometry'].bounds.values):
            p_pred = np.asarray(p_pred, dtype=float)
            p_gt = np.asarray(p_gt, dtype=float)
            iou = bb_intersection_over_union(p_pred, p_gt)
            m[i, j] = iou

    TP = np.clip(np.sum(m > iou_threshold, axis=1), 0, 1).sum()

    FN = m.shape[0] - TP
    FP = m.shape[1] - TP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    t = PrettyTable(
        ['All Plants GT', 'All Plants Pred', 'True Positives', 'False Positves', 'False Negatives',
         'Precision', 'Recall'])
    t.float_format = '0.2'
    t.add_row([len(df_ground_truth),
               len(df_prediction),
               TP,
               FP,
               FN,
               precision * 100,
               recall * 100])
    print(t)

def calc_volume(prediction_tif, dsm_tif, field_shape, seed_shape, species):
    intersection = gpd.overlay(field_shape, seed_shape, how='intersection')
    shapes = [feature.geometry for i, feature in intersection.iterrows()]

    out_pred, transform_pred = rio.mask.mask(prediction_tif, shapes, crop=True)
    out_dsm, transform_dsm = rio.mask.mask(dsm_tif, shapes, crop=True)
    out_pred = np.moveaxis(out_pred, 0, -1)
    out_dsm = out_dsm.squeeze()

    out_dsm = resize(out_dsm, out_pred.shape[0:2])
    assert out_dsm.shape == out_pred.shape[0:2], "shapes dont match"

    masked_dsm = out_dsm.copy()
    for i in range(out_pred.shape[-1]):
        if i == species:
            continue
        masked_dsm[out_pred[..., i] > 0] = 0
    masked_dsm = np.clip(masked_dsm, a_min=0, a_max=None)

    hemp_area = masked_dsm.sum() * transform_pred[0] * transform_pred[4] * -1

    return 100 * hemp_area / intersection.area.sum(), intersection.area.sum()
