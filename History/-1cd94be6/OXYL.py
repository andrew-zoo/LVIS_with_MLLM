# detector_only_lvis_eval_mpi.py
# Copyright (c) OpenMMLab. All Rights Reserved.
import os, json, warnings
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from lvis import LVIS, LVISEval

# ----------------------------------------------------------------------
LVIS_VAL_ANNO = "/home/jhp0720/.vscode-server/data/User/LVIS_with_MLLM/datasets/lvis_v1_minival.json"
LVIS_IMG_ROOT = "/home/jhp0720/.vscode-server/data/User/LVIS_with_MLLM/datasets/val2017"
# ----------------------------------------------------------------------

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

def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return [x1, y1, x2 - x1, y2 - y1]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Config or checkpoint .pth file or model alias.')
    parser.add_argument('--weights', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--pred-score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument('--show', action='store_true', help='Display the image.')
    parser.add_argument('--no-save-vis', action='store_true', default=True, help='Do not save vis')
    parser.add_argument('--no-save-pred', action='store_true', default=False, help='Do not save json')
    parser.add_argument('--print-result', action='store_true', help='Print results.')
    parser.add_argument('--palette', default='random',
                        choices=['coco', 'voc', 'citys', 'random', 'none'])
    parser.add_argument('--custom-entities', '-c', default=True)
    parser.add_argument('--split', type=int, default=0, help='500-image split index')
    parser.add_argument('--max-per-img', type=int, default=300, help='MPI: max dets per image')
    args = parser.parse_args()
    return args

def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Detector 초기화
    init_args = {
        'model': args.model, 'weights': args.weights,
        'device': args.device, 'palette': args.palette,
    }
    inferencer = DetInferencer(**init_args)
    # max_per_img 적용
    if hasattr(inferencer.model, 'test_cfg') and hasattr(inferencer.model.test_cfg, 'max_per_img'):
        inferencer.model.test_cfg.max_per_img = int(args.max_per_img)
    mpi = int(getattr(inferencer.model.test_cfg, 'max_per_img', -1))
    print("max_per_img =", mpi)

    # LVIS 메타
    lvis = LVIS(LVIS_VAL_ANNO)
    all_img_ids = lvis.get_img_ids()
    cats = lvis.load_cats(lvis.get_cat_ids())
    lvis_category_names = [c['name'] for c in cats]
    name2id = {c['name']: c['id'] for c in cats}

    # 스플릿 (500장 단위)
    s = int(args.split)
    unit = 500
    split_img_ids = all_img_ids[unit * s:] if s == 21 else all_img_ids[unit * s: unit * (s + 1)]

    texts_for_detector = [tuple(lvis_category_names)]
    score_thr = float(args.pred_score_thr)
    all_predictions = []

    print(f'[split {s}] #images: {len(split_img_ids)}')
    pbar = tqdm(split_img_ids, total=len(split_img_ids), dynamic_ncols=True, desc=f"split {s}",
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]")
    for img_id in pbar:
        img_info = lvis.load_imgs([img_id])[0]
        img_path = os.path.join(LVIS_IMG_ROOT, get_coco_filename(img_info))

        # Inferencer 호출 (pred_score_thr는 여기서 넘기지 않고, 사후 필터로만 사용해도 OK)
        det_kwargs = {'inputs': img_path, 'texts': texts_for_detector}
        results = inferencer(**det_kwargs)

        labels = np.asarray(results['predictions'][0]['labels'], dtype=int)
        scores = np.asarray(results['predictions'][0]['scores'], dtype=float)
        bboxes = np.asarray(results['predictions'][0]['bboxes'], dtype=float).reshape(-1, 4)

        keep = scores >= score_thr
        if not np.any(keep):
            continue
        labels, scores, bboxes = labels[keep], scores[keep], bboxes[keep]

        for li, sc, bb in zip(labels, scores, bboxes):
            cname = lvis_category_names[int(li)]
            cid = int(name2id[cname])
            all_predictions.append({
                "image_id": int(img_id),
                "bbox": xyxy_to_xywh(bb),
                "category_id": cid,
                "score": float(sc),
            })

    # 저장 (파일명에 mpi 포함)
    pred_path = os.path.join(args.out_dir, f"lvis_results_mpi{mpi}_split{s}.json")
    if not args.no_save_pred:
        with open(pred_path, 'w') as f:
            json.dump(all_predictions, f)
        print("Saved:", pred_path, "| #dets:", len(all_predictions))
    else:
        # 평가 위해 임시 경로라도 필요
        with open(pred_path, 'w') as f:
            json.dump(all_predictions, f)
        print("Saved (eval-only):", pred_path, "| #dets:", len(all_predictions))

    # 실행 메타 저장(재현성)
    meta = {
        "model": args.model,
        "weights": args.weights,
        "pred_score_thr": args.pred_score_thr,
        "max_per_img": mpi,
        "split": s
    }
    with open(os.path.join(args.out_dir, f"runmeta_mpi{mpi}_split{s}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 스플릿 단위 평가
    print("Evaluating on LVIS (this split only)...")
    eval_on_lvis(pred_path, LVIS_VAL_ANNO, img_ids=split_img_ids)

if __name__ == '__main__':
    main()
