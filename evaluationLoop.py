"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from CRAFT.DTRutils import AttnLabelConverter
from CRAFT.DTRmodel import Model

import CRAFT.craft_utils as craft_utils

import datasets
from torch.utils.data import DataLoader

import glob
import testFunctions

from CRAFT.craft import CRAFT

import funcUtils
from collections import OrderedDict

from interactiveWindows import Application
import tkinter as tk
import numpy as np


currentPath = os.path.dirname(os.path.abspath(__file__))

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='CRAFT/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='CRAFT/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
parser.add_argument('--cityName', default='Barrow_in_Furness', type=str, help='Name of the city of interest')
parser.add_argument('--cityKey', default='0', type=str, help='Identifying key of the city of interest', required=True)

#parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model', default='CRAFT/weights/TPS-ResNet-BiLSTM-Attn.pth',  help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', default='TPS', type=str,  help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', default='ResNet', type=str,  help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', default='BiLSTM', type=str,  help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', default='Attn', type=str, help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
parser.add_argument('--datasetPath', type=str, default = 'datasets/', required=False)
parser.add_argument('--savedFileExtension', type=str, default='.npz', required=False)
parser.add_argument('--saveFolderPath', type=str, default = './MAPHIS-Classification/', required=False)
args = parser.parse_args()

def testNet(network:CRAFT, x:torch.Tensor, textThreshold:float, linkThreshold:float,  low_text, poly:bool, refine_net):
    t0 = time.time()

    # forward pass
    with torch.no_grad():
        y, feature = network(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, textThreshold, linkThreshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, 1, 1)
    polys = craft_utils.adjustResultCoordinates(polys, 1, 1)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys

def main():
    # load detection net
    detectionNet = CRAFT()     # initialize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:        
        detectionNet.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        detectionNet.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            detectionNet = nn.DataParallel(detectionNet)

        detectionNet.to(device)
        cudnn.benchmark = False

    detectionNet.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from CRAFT.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            if torch.cuda.device_count() > 1:
                refine_net = nn.DataParallel(refine_net)
            refine_net = refine_net.to(device)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    # Load classification net

    converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3
    classificationModel = Model(args)
    print('model input parameters', args.imgH, args.imgW, args.num_fiducial, args.input_channel, args.output_channel,
          args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
          args.SequenceModeling, args.Prediction)

    print('loading pretrained classification model from %s' % args.saved_model)
    if args.cuda:
            classificationModel.load_state_dict(copyStateDict(torch.load(args.saved_model, map_location=device)))
            if torch.cuda.device_count() > 1:
                classificationModel = nn.DataParallel(classificationModel)
            classificationModel = classificationModel.to(device)
    else:
        classificationModel.load_state_dict(copyStateDict(torch.load(args.saved_model, map_location='cpu')))

    classificationModel.eval()

    t = time.time()

    # load data

    cityName = funcUtils.matchKeyToName('resources/cityZone.txt', args.cityKey)

    datasetPath = glob.glob(args.datasetPath+cityName+'/*/*/')[0]

    transform = datasets.unfold()
    mapDataset = datasets.Maps(datasetPath, transform=transform)
    mapDataloader = DataLoader(mapDataset, args.batchSize, shuffle=True, num_workers=2)

    dictOfResults = {}
    with torch.no_grad():
        for data in mapDataloader:
            tiledMap = data['map']
            mapName = data['mapName'][0]
            mD = data['metaData']
            w_b = mD['west_bound']
            n_b = mD['north_bound']
            x_diff = mD['x_diff']
            y_diff = mD['y_diff']
            indexDetectedFeature = 0
            for rowIndex in range(transform.hRatio):
                for colIndex in range(transform.wRatio):
                    tileIndex = rowIndex*transform.wRatio+colIndex
                    tileHindexMap = rowIndex*transform.stride[0]
                    tileWindexMap = colIndex*transform.stride[1]
                    vignette = torch.cat(3*[tiledMap[:,tileIndex]]).unsqueeze(0).cuda(device)
                    
                    bBoxes, _ = testNet(detectionNet, vignette, args.text_threshold, args.link_threshold, args.low_text, args.poly, refine_net)
                    if len(bBoxes)>0:
                        print("Feature Detected on tile "+str(tileIndex))
                        for bBox in bBoxes:
                            minH = int((bBox[0][1]+bBox[1][1])/2)
                            maxH = int((bBox[2][1]+bBox[3][1])/2)
                            H = maxH - minH
                            minW = int((bBox[0][0]+bBox[3][0])/2)
                            maxW = int((bBox[1][0]+bBox[2][0])/2)
                            W = maxW - minW
                            x = w_b+(minW + tileWindexMap)*x_diff
                            y = n_b+(minH + tileHindexMap)*y_diff
                            image = vignette[:,0,minH:maxH, minW:maxW].unsqueeze(1)
                            if 10<image.size()[2]<500 and 10<image.size()[3]<500 :
                                
                                # For max length prediction
                                length_for_pred = torch.IntTensor([args.batch_max_length] * args.batchSize).to(device)
                                text_for_pred = torch.LongTensor(args.batchSize, args.batch_max_length + 1).fill_(0).to(device)

                                preds = classificationModel(image, text_for_pred, is_train=False)

                                # select max probabilty (greedy decoding) then decode index to character
                                _, preds_index = preds.max(2)
                                preds_str = converter.decode(preds_index, length_for_pred)

                                preds_prob = F.softmax(preds, dim=2)
                                preds_max_prob, _ = preds_prob.max(dim=2)
                                for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                                    pred_EOS = pred.find('[s]')
                                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                                    pred_max_prob = pred_max_prob[:pred_EOS]

                                    # calculate confidence score (= multiply of pred_max_prob)
                                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                                    if confidence_score > 0.1:
                                        properties = [x.item(), y.item(),'\''+cityName+'\'',mapName,H,W, confidence_score.item(), minW + tileWindexMap, minH + tileHindexMap]
                                        np.savez(f'{args.saveFolderPath}{cityName}/tile_{mapName}_{indexDetectedFeature}.npz',image[0,0].detach().cpu())

                                        if pred not in dictOfResults.keys():
                                            dictOfResults[pred] = []                                            
                                        dictOfResults[pred] = testFunctions.checkIfAlreadyDetected(dictOfResults[pred], x.item(), y.item(), cityName, mapName, H, W, confidence_score.item(), minW + tileWindexMap, minH + tileHindexMap, pred)
                                    
                            else:
                                print(H, W) 
        print(dictOfResults)
        testFunctions.writeDetectionResultsDict(dictOfResults, datasetPath, cityName)

    print("elapsed time : {}s".format(time.time() - t))

if __name__ == '__main__':
    

    main()
