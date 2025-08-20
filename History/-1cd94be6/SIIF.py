# Copyright (c) OpenMMLab. All Rights Reserved.

import os
import json
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

from mmengine.logging import print_log
from mmdet.apis import DetInferencer

from lvis import LVIS, LVISEval

# ---------------------------------------------------------------------
LVIS_VAL_ANNO = "/home/jhp0720/.vscode-server/data/User/LVIS_with_MLLM/datasets/lvis_v1_minival.json"
LVIS_IMG_ROOT = "/home/jhp0720/.vscode-server/data/User/LVIS_with_MLLM/datasets/val2017"
# ---------------------------------------------------------------------

def eval_on_lvis(pred_path, gt_path, img_ids=None):
    lvis_gt = LVIS(gt_path)
    lvis_dt = lvis_gt.load_res(pred_path)
    lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type='bbox')
    if img_ids is not None:
        lvis_eval.params.imgIds = img_ids
    lvis_eval.run()
    lvis_eval.print_results()

def get_coco_filename(img_info):
    if 'file_name' in img_info and img_info['file_name']:
        return img_info['file_name']
    if 'coco_url' in img_info and img_info['coco_url']:
        return os.path.basename(img_info['coco_url'])
    return os.path.join('val2017', f"{int(img_info['id']):012d}.jpg")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py')
    parser.add_argument('--weights', type=str,
                        default='grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', type=str, default='outputs_lvis_baseline')
    # None이면 모델(config) 기본값을 그대로 사용
    parser.add_argument('--pred-score-thr', type=float, default=None)
    parser.add_argument('--max-per-img', type=int, default=None)
    parser.add_argument('--custom-entities', action='store_true', default=True)
    parser.add_argument('--palette', default='random',
                        choices=['coco', 'voc', 'citys', 'random', 'none'])
    parser.add_argument('--split', type=int, default=0)
    args = parser.parse_args()

    init_args = dict(model=args.model, weights=args.weights,
                     device=args.device, palette=args.palette)
    call_args = dict(out_dir=args.out_dir,
                     pred_score_thr=args.pred_score_thr,
                     max_per_img=args.max_per_img,
                     custom_entities=args.custom_entities,
                     split=args.split)
    return init_args, call_args

def xyxy_to_xywh(xyxy):
    if len(xyxy) != 4:
        raise ValueError("xyxy must be [x1,y1,x2,y2]")
    x1, y1, x2, y2 = map(float, xyxy)
    return [x1, y1, x2 - x1, y2 - y1]

def main():
    init_args, call_args = parse_args()
    inferencer = DetInferencer(**init_args)

    # max_per_img 덮어쓰기(옵션)
    if call_args.get('max_per_img') is not None:
        inferencer.model.test_cfg.max_per_img = int(call_args['max_per_img'])
    print('max_per_img =', inferencer.model.test_cfg.max_per_img)

    # LVIS 로드
    lvis = LVIS(LVIS_VAL_ANNO)
    all_img_ids = lvis.get_img_ids()
    cats = lvis.load_cats(lvis.get_cat_ids())
    lvis_category_names = [c['name'] for c in cats]
    name2id = {c['name']: c['id'] for c in cats}

    # 스플릿(500장 단위)
    data_split = int(call_args['split'])
    split_unit = 500
    if data_split == 21:
        split_img_ids = all_img_ids[split_unit * data_split:]
    else:
        split_img_ids = all_img_ids[split_unit * data_split: split_unit * (data_split + 1)]
    print(f'#images in this split: {len(split_img_ids)}')

    user_thr = call_args['pred_score_thr']  # float or None

    # 사용자가 스코어 임계값을 지정한 경우: 내부 필터 OFF(0.0)로 모두 받아온 뒤 수동 필터
    infer_kwargs = {}
    if user_thr is not None:
        infer_kwargs['pred_score_thr'] = 0.0

    all_predictions = []
    os.makedirs(call_args['out_dir'], exist_ok=True)

    print(f'[split {data_split}] #images: {len(split_img_ids)}')

    pbar = tqdm(
        split_img_ids,
        total=len(split_img_ids),
        dynamic_ncols=True,
        desc=f"split {data_split}",
        leave=True
    )

    for i, img_id in enumerate(pbar, 1):
        img_info = lvis.load_imgs([img_id])[0]
        file_name = get_coco_filename(img_info)
        img_path = os.path.join(LVIS_IMG_ROOT, file_name)

        results = inferencer(
            inputs=img_path,
            texts=[tuple(lvis_category_names)],
            custom_entities=call_args['custom_entities'],
            **infer_kwargs
        )

        labels = np.asarray(results['predictions'][0]['labels'], dtype=int)
        scores = np.asarray(results['predictions'][0]['scores'], dtype=float)
        bboxes = np.asarray(results['predictions'][0]['bboxes'], dtype=float).reshape(-1, 4)

        # 수동 필터는 사용자가 임계값을 준 경우에만
        if user_thr is not None:
            keep = scores >= float(user_thr)
            labels, scores, bboxes = labels[keep], scores[keep], bboxes[keep]
            if labels.size == 0:
                continue

        for label_idx, score, bbox in zip(labels, scores, bboxes):
            cat_name = lvis_category_names[label_idx]
            cat_id = name2id[cat_name]
            all_predictions.append({
                "image_id": int(img_id),
                "bbox": xyxy_to_xywh(bbox),
                "category_id": int(cat_id),
                "score": float(score),
            })

    # 저장 & (스플릿 단위) 평가
    thr_tag = "auto" if user_thr is None else f"{float(user_thr):.3f}"
    results_path = os.path.join(
        call_args['out_dir'],
        f"lvis_gdino_baseline_split{data_split}_mp{inferencer.model.test_cfg.max_per_img}_thr{thr_tag}.json"
    )
    print('Saving to', results_path)
    with open(results_path, 'w') as f:
        json.dump(all_predictions, f)

    print("Evaluating on LVIS (this split only)...")
    eval_on_lvis(results_path, LVIS_VAL_ANNO, img_ids=split_img_ids)

if __name__ == "__main__":
    main()
