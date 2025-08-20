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
    # 1) file_name이 있으면 그대로
    if 'file_name' in img_info and img_info['file_name']:
        return img_info['file_name']
    # 2) coco_url의 basename
    if 'coco_url' in img_info and img_info['coco_url']:
        return os.path.basename(img_info['coco_url'])
    # 3) 마지막 수단: COCO 규칙 + 현재 디렉토리 구조 반영
    #    LVIS_IMG_ROOT가 .../val2017 이므로 "val2017/xxxxxxxxxxxx.jpg"
    return os.path.join('val2017', f"{int(img_info['id']):012d}.jpg")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py')
    parser.add_argument('--weights', type=str,
                        default='grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', type=str, default='outputs_lvis_baseline')
    parser.add_argument('--pred-score-thr', type=float, default=0.1)
    parser.add_argument('--max-per-img', type=int, default=None)
    parser.add_argument('--custom-entities', action='store_true', default=True)
    parser.add_argument('--palette', default='random',
                        choices=['coco', 'voc', 'citys', 'random', 'none'])
    parser.add_argument('--split', type=int, default=0)
    args = parser.parse_args()

    # DetInferencer 초기화에 필요한 키만 추림
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

    # max_per_img 덮어쓰기(지정한 경우)
    if call_args.get('max_per_img') is not None:
        inferencer.model.test_cfg.max_per_img = int(call_args['max_per_img'])
    print('max_per_img =', inferencer.model.test_cfg.max_per_img)

    # LVIS 로드
    lvis = LVIS(LVIS_VAL_ANNO)
    all_img_ids = lvis.get_img_ids()
    cats = lvis.load_cats(lvis.get_cat_ids())
    lvis_category_names = [c['name'] for c in cats]
    name2id = {c['name']: c['id'] for c in cats}

    # 스플릿 쪼개기 (500장 단위)
    data_split = int(call_args['split'])
    split_unit = 500
    if data_split == 21:
        split_img_ids = all_img_ids[split_unit * data_split:]
    else:
        split_img_ids = all_img_ids[split_unit * data_split: split_unit * (data_split + 1)]

    print(f'#images in this split: {len(split_img_ids)}')

    score_thr = float(call_args['pred_score_thr'])
    all_predictions = []

    os.makedirs(call_args['out_dir'], exist_ok=True)

    for img_id in tqdm(split_img_ids):
        img_info = lvis.load_imgs([img_id])[0]
        file_name = get_coco_filename(img_info)
        img_path = os.path.join(LVIS_IMG_ROOT, file_name)

        # GDINO 호출: LVIS 전체 클래스명을 프롬프트로
        results = inferencer(
            inputs=img_path,
            texts=[tuple(lvis_category_names)],
            custom_entities=call_args['custom_entities'],
            pred_score_thr=score_thr,
        )

        # 결과 파싱 + 점수 필터(안전)
        labels = np.asarray(results['predictions'][0]['labels'], dtype=int)
        scores = np.asarray(results['predictions'][0]['scores'], dtype=float)
        bboxes = np.asarray(results['predictions'][0]['bboxes'], dtype=float).reshape(-1, 4)

        keep = scores >= score_thr
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

    # 저장 & 평가
    results_path = os.path.join(
        call_args['out_dir'],
        f"lvis_gdino_baseline_split{data_split}_mp{inferencer.model.test_cfg.max_per_img}_thr{score_thr}.json"
    )
    print('Saving to', results_path)
    with open(results_path, 'w') as f:
        json.dump(all_predictions, f)

    print("Evaluating on LVIS (this split only)...")
    eval_on_lvis(results_path, LVIS_VAL_ANNO, img_ids=split_img_ids)

if __name__ == "__main__":
    main()
