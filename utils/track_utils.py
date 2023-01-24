import numpy as np
import cv2
import torch


def rect1_2_cxy_wh(rect):
    # rect: xmin, ymin, w, h
    return np.array([rect[0] + rect[2] / 2 - 1, rect[1] + rect[3] / 2 - 1]), np.array([rect[2], rect[3]])  # 0-index


def rect0_2_cxy_wh(rect):
    # rect: xmin, ymin, w, h
    return np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]), np.array([max(1., rect[2]), max(1., rect[3])])  # 0-index


# for vot, we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
        return cx, cy, w, h
    else:
        return region


def _poly(region, center):
    if region.shape[0] == 4:
        return _rect(region, center)
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx, cy, w, h
    else:
        return cx - w / 2, cy - h / 2, w, h


def pad_frame(frame, frame_sz, pos_x, pos_y, original_patch_sz, avg_color):
    # frame in (H, W, C) cv2 format
    # find center of original patch, which should be the center of the target (if no padding is needed)
    c = original_patch_sz / 2
    # how much padding is needed according to frame size & absolute target position
    left_pad = np.max([0, -np.round(pos_x - c)])  # if pos_x > patch_center, => need to pad left side by c - pos_x
    right_pad = np.max(
        [0, np.round(pos_x + c) - frame_sz[1]])  # if pos_x + c > width => pad right side by pos_x + c - width
    top_pad = np.max([0, -np.round(pos_y - c)])  # if pos_y > patch_center => pad top by c - pos_y
    bottom_pad = np.max(
        [0, np.round(pos_y + c) - frame_sz[0]])  # is pos_y + c > height => pad bottom by pos_y + c -height

    # pad frame if needed using opencv
    npad = (int(left_pad), int(right_pad), int(top_pad), int(bottom_pad))
    if max(npad) > 0:
        padded_frame = cv2.copyMakeBorder(frame, int(top_pad), int(bottom_pad), int(left_pad), int(right_pad),
                                          cv2.BORDER_CONSTANT, value=avg_color)
    else:
        padded_frame = frame
    return padded_frame, npad


def extract_crops_z(frame, npad, pos_x, pos_y, original_patch_sz, destination_sz):
    # frame in (H, W, C) cv2 format
    # find center of original patch
    c = original_patch_sz / 2
    crop_width = max(1, np.round(pos_x + c) - np.round(pos_x - c))  # width, height should be about 2*c
    crop_height = max(1, np.round(pos_y + c) - np.round(pos_y - c))
    top_left_x = max(1, min(npad[0] + np.round(pos_x - c), frame.shape[1] - crop_width))
    top_left_y = max(1, min(npad[2] + np.round(pos_y - c), frame.shape[0] - crop_height))
    # print 'z :', top_left_x, top_left_y, crop_width, crop_height

    crop = frame[int(top_left_y):int(top_left_y + crop_height + 1), int(top_left_x):int(top_left_x + crop_width + 1), :]

    crop_rs = cv2.resize(crop, (int(destination_sz), int(destination_sz)), interpolation=cv2.INTER_CUBIC)
    return crop_rs


def image_to_tensor(image):
    return torch.FloatTensor(np.transpose(image, (2, 0, 1)))


def tensor_to_image(tensor):
    return tensor.cpu().permute(2, 0, 1).numpy()


def extract_crops_x(frame, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
    # frame in (H, W, C) cv2 format, sz_src is tuple of sizes (for pyramid)
    c = sz_src2 / 2
    width = max(1.0, np.round(pos_x + c) - np.round(pos_x - c))
    height = max(1.0, np.round(pos_y + c) - np.round(pos_y - c))
    top_left_x = max(1, min(npad[0] + np.round(pos_x - c), frame.shape[1] - width))
    top_left_y = max(1, min(npad[2] + np.round(pos_y - c), frame.shape[0] - height))

    search_area = frame[int(top_left_y):int(top_left_y + height + 1),
                  int(top_left_x):int(top_left_x + width + 1), :]

    offset_s0 = max(1, (sz_src2 - sz_src0) / 2)
    offset_s1 = max(1, (sz_src2 - sz_src1) / 2)

    if offset_s0 + 1 >= search_area.shape[0] or offset_s0 + 1 >= search_area.shape[1]:
        crop_s0 = search_area
    else:
        crop_s0 = search_area[int(offset_s0):int(offset_s0 + sz_src0) + 1,
                              int(offset_s0):int(offset_s0 + sz_src0) + 1, :]
    crop_s0 = cv2.resize(crop_s0, (int(sz_dst), int(sz_dst)), interpolation=cv2.INTER_CUBIC)
    if offset_s1 + 1 >= search_area.shape[0] or offset_s0 + 1 >= search_area.shape[1]:
        crop_s1 = search_area
    else:
        crop_s1 = search_area[int(offset_s1):int(offset_s1 + sz_src1) + 1,
                              int(offset_s1):int(offset_s1 + sz_src1) + 1, :]

    crop_s1 = cv2.resize(crop_s1, (int(sz_dst), int(sz_dst)), interpolation=cv2.INTER_CUBIC)
    crop_s2 = cv2.resize(search_area, (int(sz_dst), int(sz_dst)), interpolation=cv2.INTER_CUBIC)

    del search_area
    crops = np.vstack([crop_s0[np.newaxis, :], crop_s1[np.newaxis, :], crop_s2[np.newaxis, :]])
    return crops


def resize_scoremap(scoremap, final_score_sz=None):
    return cv2.resize(scoremap.squeeze(), (final_score_sz, final_score_sz), cv2.INTER_CUBIC)


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='cv'):

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
        # print top_pad, bottom_pad, left_pad, right_pad
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original

    return image_to_tensor(im_patch) if out_mode in 'torch' else im_patch


def center_error(rects1, rects2):
    r"""Center error.
    """
    centers1 = rects1[:, :2] + rects1[:, 2:] / 2.0
    centers2 = rects2[:, :2] + rects2[:, 2:] / 2.0
    ces = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=1))

    return ces


def iou(rects1, rects2):
    r"""Intersection over union.
    """
    rects_inter = _intersection(rects1, rects2)

    if rects1.ndim == 1:
        areas1 = np.prod(rects1[2:])
        areas2 = np.prod(rects2[2:])
        area_inter = np.prod(rects_inter[2:])
    elif rects1.ndim == 2:
        areas1 = np.prod(rects1[:, 2:], axis=1)
        areas2 = np.prod(rects2[:, 2:], axis=1)
        area_inter = np.prod(rects_inter[:, 2:], axis=1)
    else:
        raise Exception('Wrong dimension of rects!')

    area_union = areas1 + areas2 - area_inter
    ious = area_inter / (area_union + 1e-12)

    return ious


def _intersection(rects1, rects2):
    r"""Rectangle intersection.
    """
    assert rects1.shape == rects2.shape

    if rects1.ndim == 1:
        x1 = max(rects1[0], rects2[0])
        y1 = max(rects1[1], rects2[1])
        x2 = min(rects1[0] + rects1[2], rects2[0] + rects2[2])
        y2 = min(rects1[1] + rects1[3], rects2[1] + rects2[3])

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        return np.array([x1, y1, w, h])
    elif rects1.ndim == 2:
        x1 = np.maximum(rects1[:, 0], rects2[:, 0])
        y1 = np.maximum(rects1[:, 1], rects2[:, 1])
        x2 = np.minimum(rects1[:, 0] + rects1[:, 2],
                        rects2[:, 0] + rects2[:, 2])
        y2 = np.minimum(rects1[:, 1] + rects1[:, 3],
                        rects2[:, 1] + rects2[:, 3])

        w = np.maximum(x2 - x1, 0)
        h = np.maximum(y2 - y1, 0)

        return np.stack((x1, y1, w, h), axis=1)


def compute_success_overlap(gt_bb, result_bb, n_thresholds=21):
    thresholds_overlap = np.linspace(0, 1, num=n_thresholds)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou_ = iou(gt_bb, result_bb)
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou_ > thresholds_overlap[i]) / float(n_frame)
    return success
