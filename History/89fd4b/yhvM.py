# Copyright (c) OpenMMLab. All rights reserved.


import ast
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes

import os
from tqdm import tqdm
import json

# import logging
from PIL import Image

# visualiser
from visualizer import Visualizer
import numpy as np

# llava-onevision related
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import matplotlib.pyplot as plt

import copy
import torch

import sys
import warnings

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
JSON_ANNO_PATH = "/home/kyp/repo/prj-xdecoder/DesCo/DATASET/DOD/d3_json"

def eval_on_d3(pred_path, mode="pn"):
    assert mode in ("pn", "p", "n")
    if mode == "pn":
        gt_path = os.path.join(JSON_ANNO_PATH, "d3_full_annotations.json")
    elif mode == "p":
        gt_path = os.path.join(JSON_ANNO_PATH, "d3_pres_annotations.json")
    else:
        gt_path = os.path.join(JSON_ANNO_PATH, "d3_abs_annotations.json")
    coco = COCO(gt_path)
    d3_res = coco.loadRes(pred_path)
    cocoEval = COCOeval(coco, d3_res, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

# CUDA_VISIBLE_DEVICES=5 python demo/simple_subject_llava_dod_eval.py configs/mm_grounding_dino/dod/grounding_dino_swin-t_pretrain_zeroshot_parallel_dod.py --weights grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth --palette random --split 4

# CUDA_VISIBLE_DEVICES=0 python demo/simple_subject_llava_dod_eval.py --split 0
# CUDA_VISIBLE_DEVICES=1 python demo/simple_subject_llava_dod_eval.py --split 1
# CUDA_VISIBLE_DEVICES=2 python demo/simple_subject_llava_dod_eval.py --split 2
# CUDA_VISIBLE_DEVICES=3 python demo/simple_subject_llava_dod_eval.py --split 3
# CUDA_VISIBLE_DEVICES=4 python demo/simple_subject_llava_dod_eval.py --split 4
# CUDA_VISIBLE_DEVICES=5 python demo/simple_subject_llava_dod_eval.py --split 5
# CUDA_VISIBLE_DEVICES=6 python demo/simple_subject_llava_dod_eval.py --split 6
# CUDA_VISIBLE_DEVICES=7 python demo/simple_subject_llava_dod_eval.py --split 7

# CUDA_VISIBLE_DEVICES=0 python demo/simple_subject_llava_dod_eval.py --split 8
# CUDA_VISIBLE_DEVICES=1 python demo/simple_subject_llava_dod_eval.py --split 9
# CUDA_VISIBLE_DEVICES=2 python demo/simple_subject_llava_dod_eval.py --split 10
# CUDA_VISIBLE_DEVICES=3 python demo/simple_subject_llava_dod_eval.py --split 11
# CUDA_VISIBLE_DEVICES=4 python demo/simple_subject_llava_dod_eval.py --split 12
# CUDA_VISIBLE_DEVICES=5 python demo/simple_subject_llava_dod_eval.py --split 13
# CUDA_VISIBLE_DEVICES=6 python demo/simple_subject_llava_dod_eval.py --split 14
# CUDA_VISIBLE_DEVICES=7 python demo/simple_subject_llava_dod_eval.py --split 15

# CUDA_VISIBLE_DEVICES=0 python demo/simple_subject_llava_dod_eval.py --split 16
# CUDA_VISIBLE_DEVICES=1 python demo/simple_subject_llava_dod_eval.py --split 17
# CUDA_VISIBLE_DEVICES=2 python demo/simple_subject_llava_dod_eval.py --split 18
# CUDA_VISIBLE_DEVICES=3 python demo/simple_subject_llava_dod_eval.py --split 19
# CUDA_VISIBLE_DEVICES=4 python demo/simple_subject_llava_dod_eval.py --split 20
# CUDA_VISIBLE_DEVICES=5 python demo/simple_subject_llava_dod_eval.py --split 21

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument(
    #     'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--model',
        type=str,
        default='configs/mm_grounding_dino/dod/grounding_dino_swin-t_pretrain_zeroshot_parallel_dod.py',
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default='grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth', help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    # Once you input a format similar to $: xxx, it indicates that
    # the prompt is based on the dataset class name.
    # support $: coco, $: voc, $: cityscapes, $: lvis, $: imagenet_det.
    # detail to `mmdet/evaluation/functional/class_names.py`
    # parser.add_argument(
    #     '--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.1,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        # action='store_true',
        default=True,
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        # action='store_true',
        default=True,
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='random',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP and Grounding DINO
    # parser.add_argument(
    #     '--custom-entities',
    #     '-c',
    #     action='store_true',
    #     help='Whether to customize entity names? '
    #     'If so, the input text should be '
    #     '"cls_name1 . cls_name2 . cls_name3 ." format')
    parser.add_argument(
        '--custom-entities',
        '-c',
        default=True,
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    # parser.add_argument(
    #     '--chunked-size',
    #     '-s',
    #     type=int,
    #     default=-1,
    #     help='If the number of categories is very large, '
    #     'you can specify this parameter to truncate multiple predictions.')
    # only for Grounding DINO
    # parser.add_argument(
    #     '--tokens-positive',
    #     '-p',
    #     type=str,
    #     default='-1',
    #     help='Used to specify which locations in the input text are of '
    #     'interest to the user. -1 indicates that no area is of interest, '
    #     'None indicates ignoring this parameter. '
    #     'The two-dimensional array represents the start and end positions.')
    parser.add_argument("--split", default=0, type=int)

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    # if call_args['texts'] is not None:
    #     if call_args['texts'].startswith('$:'):
    #         dataset_name = call_args['texts'][3:].strip()
    #         class_names = get_classes(dataset_name)
    #         call_args['texts'] = [tuple(class_names)]

    # if call_args['tokens_positive'] is not None:
    #     call_args['tokens_positive'] = ast.literal_eval(
    #         call_args['tokens_positive'])

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


import numpy as np

def xyxy_to_xywh(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return [xyxy[0], xyxy[1], w_temp, h_temp]

def main():
    ####### model load #######
    warnings.filterwarnings("ignore")
    # pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
    # pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    # pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov"

    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

    model.eval()

    ####### conv template load #######
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    # question = DEFAULT_IMAGE_TOKEN + "\nHere’s an image with objects labeled in numbers. Please provide unique detailed descriptions of the objects that are marked as 1. Treat each mark as a unique and separate entity, requiring a full description exclusive to that mark only."
    # question = DEFAULT_IMAGE_TOKEN + "\nIs the object labeled as 1 in the image representing {dod_des}?"
    # question = DEFAULT_IMAGE_TOKEN + "\nHere’s an image with objects labeled in numbers. Is the object labeled as 1 in the image representing {dod_des}?"
    # question = DEFAULT_IMAGE_TOKEN + "\nHere’s an image with objects labeled in numbers. Is the object labeled as 1 in the image representing {dod_des}? Please start by responding with 'yes' or 'no,' followed by a detailed explanation."
    question = DEFAULT_IMAGE_TOKEN + "\nHere’s an image with objects labeled in numbers. Is the object labeled as 1 in the image representing {dod_des}? Please respond with either 'yes' or 'no'."

    ## visualization option
    ## option (hard-coding)
    mark_label = 1
    label_mode = '1'
    alpha = 0.1
    # anno_mode = ['Box']
    anno_mode = ['Mark', 'Box']
    red_color = [1.0, 0.0, 0.0]

    ## detection model setup
    init_args, call_args = parse_args()
    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    init_args['show_progress'] = False
    inferencer = DetInferencer(**init_args)

    ##
    inferencer.model.test_cfg.max_per_img = 10
    # inferencer.model.test_cfg.max_per_img = 5
    # chunked_size = call_args.pop('chunked_size')


    ## dod dataset setup
    IMG_ROOT = "/home/kyp/repo/prj-mllm/mmdetection/data/d3/d3_images"
    PKL_ANNO_PATH = "/home/kyp/repo/prj-mllm/mmdetection/data/d3/d3_pkl"
    # import the dataset class
    from d_cube import D3
    # init a dataset instance
    d3 = D3(IMG_ROOT, PKL_ANNO_PATH)
    all_img_ids = d3.get_img_ids()  # get the image ids in the dataset
    
    # init subject json info
    sent_subject = json.load(open("dod_sentences_subject_llama3.json"))
    
    # import pdb; pdb.set_trace()
    data_split = call_args.pop('split')
    split_unit = 500
    if data_split == 21:
        split_img_ids = all_img_ids[split_unit*data_split:]
    else:
        split_img_ids = all_img_ids[split_unit*data_split:split_unit*(data_split+1)]

    all_predictions = []
    # for img_id in tqdm(all_img_ids):
    for img_id in tqdm(split_img_ids):
        img_info = d3.load_imgs(img_id)  # load images by passing a list containing some image ids

        img_path = img_info[0]["file_name"]  # obtain one image path so you can load it and inference
        # then you can load the image as input for your model

        group_ids = d3.get_group_ids(img_ids=[img_id])  # get the group ids by passing anno ids, image ids, etc.
        sent_ids = d3.get_sent_ids(group_ids=group_ids)  # get the sentence ids by passing image ids, group ids, etc.
        sent_list = d3.load_sents(sent_ids=sent_ids)
        ref_list = [sent['raw_sent'] for sent in sent_list]  # list[str]
        # import pdb; pdb.set_trace()
        # use these language references in `ref_list` as the references to your REC/OVD/DOD model
        call_args['inputs'] = os.path.join(IMG_ROOT, img_path)
        # call_args['texts'] = [". ".join(ref_list)]
        # call_args['texts'] = ". ".join(ref_list)
        call_args['texts'] = [tuple(ref_list)]
        # call_args['texts'] = ref_list
        # call_args['sent_ids'] = sent_ids
        subject_list = [sent_subject[str(sid)]['subject_llama3'] for sid in sent_ids]
        
        # print([tuple(ref_list)])
        # print(subject_list)
        # print("=========")
        # print(img_id)
        # import pdb; pdb.set_trace()
        # continue

        ## for visualize and llava-onevision
        im_load = Image.open(os.path.join(IMG_ROOT, img_path))

        ### Detection with subject
        for sub_id, subject in enumerate(subject_list):
            # call_args['texts'] = [tuple([subject])]
            call_args['texts'] = [tuple([ref_list[sub_id]])]

            # import pdb; pdb.set_trace()

            results = inferencer(**call_args)
            # import pdb; pdb.set_trace()
            if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                            and call_args['no_save_pred']):
                print_log(f'results have been saved at {call_args["out_dir"]}')
            # import pdb; pdb.set_trace()

            for label, score, bbox in zip(results['predictions'][0]['labels'], 
                                        results['predictions'][0]['scores'], 
                                        results['predictions'][0]['bboxes']):
                # import pdb; pdb.set_trace()
                ### Classify with llava-onevision
                # visualizer setup
                img_ori = np.asarray(im_load)
                visual = Visualizer(img_ori)
                h,w = img_ori.shape[:2]
                # assert(img_ori.shape[:2] == (h,w))
                ## bbox to binary mask
                bbox_int = np.around(bbox)
                bmask = np.zeros((h,w),dtype=np.uint8)
                bmask[int(bbox_int[1]):int(bbox_int[3]), int(bbox_int[0]):int(bbox_int[2])] = 1 

                # remove criteri on bbox size
                # ratio = anno['area'] / (h*w)
                # ratio_objs.append(ratio)

                demo = visual.draw_box_with_number(bmask, bbox_int, color=red_color, text=str(mark_label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
                # demo = visual.draw_box_with_number(bmask, bbox, text=str(label)+'+'+str(round(ratio * 100, 1)), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
                # label += 1
                # tot_box += 1 ## just for analysis

                im = demo.get_image()
                # plt.imshow(im)
                # plt.show()

                ### llava-one vision inference ###
                # import pdb; pdb.set_trace()
                ## input setting for llava-one vision
                pil_im = Image.fromarray(im)
                image_tensor = process_images([pil_im], image_processor, model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question.format(dod_des=ref_list[sub_id]))
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                image_sizes = [pil_im.size]

                # import pdb; pdb.set_trace()
                cont = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=4096,
                )
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

                ## print results
                # print(question.format(dod_des=ref_list[sub_id]))
                # print(text_outputs)
                # import pdb; pdb.set_trace()
                # break
                # import pdb; pdb.set_trace()
                # continue
                
                # if 
                ans_lower = text_outputs[0].lower()
                all_in = 'yes' in ans_lower[:10] and 'no' in ans_lower[:10]
                none_in = 'yes' not in ans_lower[:10] and 'no' not in ans_lower[:10]

                if all_in or none_in:
                    print(text_outputs, img_id, sub_id)
                    # import pdb; pdb.set_trace()

                if 'yes' in ans_lower:
                    all_predictions.append({
                            "image_id": img_id,
                            "bbox": xyxy_to_xywh(bbox),
                            "category_id": sent_ids[sub_id],
                            "score": score,
                    })
        # import pdb; pdb.set_trace()

        # save the result to a JSON file
    # import pdb; pdb.set_trace()
    # result_save_json = "%s_results.json"%(dataset_name)
    # result_save_json = "./outputs/d3_try_results.json"
    # results_path = os.path.join(output_folder, result_save_json)
    # results_path = "./outputs/d3_simple_subject_llava_onevision_8B_results{split}.json".format(split=data_split)
    # results_path = "./outputs/d3_simple_subject_llava_onevision_72B_results{split}.json".format(split=data_split)
    # results_path = "./outputs/d3_simple_subject_llava_onevision_05B_results{split}.json".format(split=data_split)
    results_path = "./outputs/d3_simple_subject_llava_onevision_7B_results{split}.json".format(split=data_split)
    print('Saving to', results_path)
    json.dump(all_predictions, open(results_path, 'w'))

    # eval_on_d3(results_path, mode='pn')
    # eval_on_d3(results_path, mode='p')
    # eval_on_d3(results_path, mode='n')

if __name__ == '__main__':
    main()
