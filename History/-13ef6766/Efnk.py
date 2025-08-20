# Copyright (c) OpenMMLab. All Rights Reserved.
import ast
from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
from PIL import Image
import copy
import torch
import warnings
import numpy as np

# MMDetection imports
from mmengine.logging import print_log
from mmdet.apis import DetInferencer

# LLava-OneVision imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

# COCO tools for LVIS evaluation
from pycocotools.coco import COCO
from lvis import LVIS, LVISEval

# Visualizer for creating images with bounding boxes (if needed)
from visualizer import Visualizer

# -----------------------------------------------------------------------------
LVIS_VAL_ANNO = "/home/jhp0720/.vscode-server/data/User/LVIS_with_MLLM/datasets/lvis_v1_minival.json"
LVIS_IMG_ROOT = "/home/jhp0720/.vscode-server/data/User/LVIS_with_MLLM/datasets/val2017"
# -----------------------------------------------------------------------------

def eval_on_lvis(pred_path, gt_path, img_ids=None):
    lvis_gt = LVIS(gt_path)
    lvis_dt = lvis_gt.load_res(pred_path)
    lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type='bbox')
    if img_ids is not None:
        lvis_eval.params.imgIds = img_ids  # 분할 평가 시
    lvis_eval.run()
    lvis_eval.print_results()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='configs/mm_grounding_dino/dod/grounding_dino_swin-t_pretrain_zeroshot_parallel_dod.py',
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile.')
    parser.add_argument('--weights', default='grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth', help='Checkpoint file')
    parser.add_argument('--out-dir', type=str, default='outputs_lvis', help='Output directory.')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--pred-score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument('--show', action='store_true', help='Display the image in a popup window.')
    parser.add_argument('--no-save-vis', action='store_true', default=True, help='Do not save detection vis results')
    parser.add_argument('--no-save-pred', action='store_true', default=False, help='Do not save detection json results')
    parser.add_argument('--print-result', action='store_true', help='Whether to print the results.')
    parser.add_argument('--palette', default='random', choices=['coco', 'voc', 'citys', 'random', 'none'], help='Color palette used for visualization')
    parser.add_argument('--custom-entities', '-c', default=True, help='Whether to customize entity names?')
    parser.add_argument("--split", default=0, type=int)

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None
    
    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

def xyxy_to_xywh(xyxy):
    """
    XYXY 형식을 XYWH 형식으로 변환합니다.
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return [xyxy[0], xyxy[1], w_temp, h_temp]

def main():
    # LLava-OneVision 모델 로딩 (기존 코드 유지)
    warnings.filterwarnings("ignore")
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
    model.eval()

    # Conversation template (기존 코드 유지)
    conv_template = "qwen_1_5"
    question = DEFAULT_IMAGE_TOKEN + "\nHere’s an image with objects labeled in numbers. Is the object labeled as 1 in the image representing {lvis_category}? Please respond with either 'yes' or 'no'."

    # Visualization 옵션
    mark_label = 1
    label_mode = '1'
    alpha = 0.1
    anno_mode = ['Mark', 'Box']
    red_color = [1.0, 0.0, 0.0]

    # Grounding DINO 모델 설정 (기존 코드 유지)
    init_args, call_args = parse_args()
    init_args['show_progress'] = False
    inferencer = DetInferencer(**init_args)
    inferencer.model.test_cfg.max_per_img = 10

    # LVIS 데이터셋 설정
    lvis = COCO(LVIS_VAL_ANNO)
    all_img_ids = lvis.getImgIds()
    
    # LVIS의 모든 카테고리 이름 목록 가져오기
    lvis_categories = {cat['id']: cat['name'] for cat in lvis.loadCats(lvis.getCatIds())}
    lvis_category_names = list(lvis_categories.values())
    name2id = {cat['name']: cat['id'] for cat in lvis.loadCats(lvis.getCatIds())}

    # 병렬 처리를 위한 데이터 분할
    data_split = call_args.pop('split')
    split_unit = 500
    if data_split == 21:
        split_img_ids = all_img_ids[split_unit * data_split:]
    else:
        split_img_ids = all_img_ids[split_unit * data_split:split_unit * (data_split + 1)]

    all_predictions = []

    for img_id in tqdm(split_img_ids):
        img_info = lvis.loadImgs(img_id)[0]
        # 'val2017' 폴더 내의 이미지 경로를 올바르게 구성
        img_path = os.path.join(LVIS_IMG_ROOT, img_info["file_name"])

        # Grounding DINO의 텍스트 프롬프트로 모든 LVIS 카테고리 이름을 사용
        call_args['inputs'] = img_path
        call_args['texts'] = [tuple(lvis_category_names)]

        results = inferencer(**call_args)
        
        # 시각화 및 LLava를 위한 이미지 로드
        im_load = Image.open(img_path)
        img_ori = np.asarray(im_load)
        h, w = img_ori.shape[:2]

        for label_idx, score, bbox in zip(results['predictions'][0]['labels'], 
                                           results['predictions'][0]['scores'], 
                                           results['predictions'][0]['bboxes']):
            
            # 프롬프트에 사용할 LVIS 카테고리 이름 가져오기
            lvis_category_name = lvis_category_names[label_idx]

            # 시각화 로직 (기존 코드 유지)
            visual = Visualizer(img_ori)
            bbox_int = np.around(bbox)
            bmask = np.zeros((h, w), dtype=np.uint8)
            bmask[int(bbox_int[1]):int(bbox_int[3]), int(bbox_int[0]):int(bbox_int[2])] = 1
            demo = visual.draw_box_with_number(bmask, bbox_int, color=red_color, text=str(mark_label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
            im = demo.get_image()
            pil_im = Image.fromarray(im)

            # LLava-OneVision 추론 (기존 코드 유지)
            image_tensor = process_images([pil_im], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question.format(lvis_category=lvis_category_name))
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [pil_im.size]

            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            
            ans_lower = text_outputs[0].lower()
            
            # 'yes'가 포함된 경우 예측 결과 저장
            if 'yes' in ans_lower:
                lvis_category_name = lvis_category_names[label_idx]
                cat_id = name2id[lvis_category_name]          # ← 변환 추가
                all_predictions.append({
                    "image_id": img_id,
                    "bbox": xyxy_to_xywh(bbox),
                    "category_id": cat_id, 
                })
    
    # 결과 저장
    if not call_args['no_save_pred']:
        os.makedirs(call_args['out_dir'], exist_ok=True)
        results_path = os.path.join(call_args['out_dir'], f"lvis_llava_onevision_7B_results{data_split}.json")
        print('Saving to', results_path)
        json.dump(all_predictions, open(results_path, 'w'))

    # 모든 분할 결과를 합친 후 평가 수행
    print("Evaluating on LVIS...")
    # eval_on_lvis(results_path, LVIS_VAL_ANNO)

if __name__ == '__main__':
    main()