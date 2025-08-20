# detector_only_lvis_eval.py
# Copyright (c) OpenMMLab. All Rights Reserved.
import os, json, warnings
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from mmengine.logging import print_log
from mmdet.apis import DetInferencer

from lvis import LVIS, LVISEval

# -----------------------------------------------------------------------------
LVIS_VAL_ANNO = "/home/jhp0720/.vscode-server/data/User/LVIS_with_MLLM/datasets/lvis_v1_minival.json"
LVIS_IMG_ROOT = "/home/jhp0720/.vscode-server/data/User/LVIS_with_MLLM/datasets/val2017"
# -----------------------------------------------------------------------------

def eval_on_lvis(pred_path, gt_path, img_ids=None):
    lvis_gt = LVIS(gt_path)
    lvis_dt = lvis_gt.load_res(pred_path)
    lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type='bbox')
    if img_ids is not None:
        lvis_eval.params.imgIds = img_ids  # 스플릿 단위 평가
    lvis_eval.run()
    lvis_eval.print_results()

def get_coco_filename(img_info):
    # 1) file_name이 있으면 그대로
    if 'file_name' in img_info and img_info['file_name']:
        return img_info['file_name']
    # 2) coco_url basename
    if 'coco_url' in img_info and img_info['coco_url']:
        return os.path.basename(img_info['coco_url'])
    # 3) 당신의 디렉토리 구조(…/val2017/val2017/…)를 그대로 따름
    return os.path.join('val2017', f"{int(img_info['id']):012d}.jpg")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='configs/mm_grounding_dino/dod/grounding_dino_swin-t_pretrain_zeroshot_parallel_dod.py',
        help='Config or checkpoint .pth file or the model name/alias.')
    parser.add_argument('--weights', default='grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth',
                        help='Checkpoint file')
    parser.add_argument('--out-dir', type=str, default='outputs_lvis', help='Output directory.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--pred-score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument('--show', action='store_true', help='Display the image in a popup window.')
    parser.add_argument('--no-save-vis', action='store_true', default=True, help='Do not save detection vis results')
    parser.add_argument('--no-save-pred', action='store_true', default=False, help='Do not save detection json results')
    parser.add_argument('--print-result', action='store_true', help='Whether to print the results.')
    parser.add_argument('--palette', default='random', choices=['coco', 'voc', 'citys', 'random', 'none'],
                        help='Color palette used for visualization')
    parser.add_argument('--custom-entities', '-c', default=True, help='Use custom entity names (texts).')
    parser.add_argument('--split', default=0, type=int, help='500장 단위 스플릿 인덱스')

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically assign it to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {k: call_args.pop(k) for k in init_kws}
    return init_args, call_args

def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return [x1, y1, x2 - x1, y2 - y1]

def main():
    warnings.filterwarnings("ignore")

    # Detector 초기화
    init_args, call_args = parse_args()
    init_args['show_progress'] = False
    inferencer = DetInferencer(**init_args)
    # 감지 최대 수 상향(원 코드 유지)
    if hasattr(inferencer.model, 'test_cfg') and hasattr(inferencer.model.test_cfg, 'max_per_img'):
        inferencer.model.test_cfg.max_per_img = 100

    # LVIS 메타 로드
    lvis = LVIS(LVIS_VAL_ANNO)
    all_img_ids = lvis.get_img_ids()
    cats = lvis.load_cats(lvis.get_cat_ids())
    lvis_category_names = [c['name'] for c in cats]
    name2id = {c['name']: c['id'] for c in cats}

    # 스플릿 정의(500장 단위)
    data_split = call_args.pop('split')
    split_unit = 500
    if data_split == 21:
        split_img_ids = all_img_ids[split_unit * data_split:]
    else:
        split_img_ids = all_img_ids[split_unit * data_split:split_unit * (data_split + 1)]

    # GroundingDINO에 LVIS 카테고리 전체를 텍스트로 제공(라벨 인덱스 매핑용)
    # DetInferencer의 labels는 아래 texts 순서의 인덱스가 됩니다.
    texts_for_detector = [tuple(lvis_category_names)]

    all_predictions = []
    score_thr = float(call_args.get('pred_score_thr', 0.1))

    print(f'[split {data_split}] #images: {len(split_img_ids)}')
    for img_id in tqdm(split_img_ids, total=len(split_img_ids), dynamic_ncols=True, desc=f"split {data_split}", leave=True):
        img_info = lvis.load_imgs([img_id])[0]
        file_name = get_coco_filename(img_info)
        img_path = os.path.join(LVIS_IMG_ROOT, file_name)

        # 디텍터 추론(텍스트는 LVIS 전체 카테고리)
        det_kwargs = dict(call_args)  # 원본 call_args 보존
        det_kwargs['inputs'] = img_path
        det_kwargs['texts'] = texts_for_detector
        results = inferencer(**det_kwargs)

        # 출력 파싱
        labels = np.asarray(results['predictions'][0]['labels'], dtype=int)
        scores = np.asarray(results['predictions'][0]['scores'], dtype=float)
        bboxes = np.asarray(results['predictions'][0]['bboxes'], dtype=float).reshape(-1, 4)

        # 스코어 필터
        keep = scores >= score_thr
        if not np.any(keep):
            continue
        labels, scores, bboxes = labels[keep], scores[keep], bboxes[keep]

        # LVIS category_id로 매핑하여 누적
        for label_idx, score, bbox in zip(labels, scores, bboxes):
            # label_idx는 texts_for_detector[0]의 인덱스
            lvis_category_name = lvis_category_names[int(label_idx)]
            cat_id = int(name2id[lvis_category_name])
            all_predictions.append({
                "image_id": int(img_id),
                "bbox": xyxy_to_xywh(bbox),
                "category_id": cat_id,
                "score": float(score),
            })

    # 결과 저장 경로(저장하지 않더라도 평가를 위해 경로는 정해둠)
    os.makedirs(call_args['out_dir'], exist_ok=True) if call_args['out_dir'] else None
    results_path = os.path.join(
        call_args['out_dir'] if call_args['out_dir'] else '.',
        f"lvis_detector_only_results_split{data_split}.json"
    )

    if not call_args['no_save_pred']:
        print('Saving to', results_path)
        with open(results_path, 'w') as f:
            json.dump(all_predictions, f)

    # 스플릿 단위로 평가(현재 스플릿의 imgIds만)
    print("Evaluating on LVIS (this split only)...")
    eval_on_lvis(results_path, LVIS_VAL_ANNO, img_ids=split_img_ids)

if __name__ == '__main__':
    main()
