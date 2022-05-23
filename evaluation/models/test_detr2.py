import torch
import time
# Some basic setup:
# Setup detectron2 logger
import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random
from PIL import Image
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# Install detectron2 that matches the above pytorch version
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html
# python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# If there is not yet a detectron2 release that matches the given torch + CUDA version, you need to install a different pytorch.

# exit(0)  # After installation, you may need to "restart runtime" in Colab. This line can also restart runtime

def getModel(model_info,type='OD'):
    model_name,hub_root,model_type = model_info
    
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(hub_root+'/'+ model_name+'.yaml'))
    print(model_zoo.get_config_file(hub_root+'/'+ model_name+'.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(hub_root+'/'+ model_name+'.yaml')

    predictor = DefaultPredictor(cfg)
    return predictor

def plot_results(img,classes, boxes,tasks,latency,percentage,CLASSES):
    plt.figure(figsize=(16,10))
    plt.imshow(img)
    ax = plt.gca()
    # colors = COLORS * 100
    for cl,(xmin, ymin, xmax, ymax) in zip(classes,boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='r', linewidth=3))
        # cl = p.argmax()
        text = f'{CLASSES[cl]}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(str(np.random.randint(0,30))+'.png')
    plt.show()

def return_results(classes,tasks,latency,percentage,CLASSES,inference=False):
    labels = {}
    for cl in classes:
        # cl = p.argmax()
        n = CLASSES[cl]
        # if n not in tasks:
        #     continue
        # if np.random.rand() > percentage:
        #     continue
        # print(p[cl].item())
        labels[n] = 1 # round(prob[cl],2)
    # print(labels)
    return(labels)

def inference(img_path, model,tasks=[],latency=0, percentage=1,categories=[],inference=False):
    # img_path, model,tasks,latency, percentage,categories

    # Image
    # im = Image.open(img_path)
    im = cv2.imread(img_path)

    # Model
    # predictor = config()
    try:
        outputs = model(im)['instances'].to("cpu")
        classes = list(outputs.pred_classes.numpy())
        # print(classes)
        # prob = outputs.objectness_logits
    except:
        outputs = model(im)#['proposals'].to("cpu")
        print(outputs)
        print('##############')
        print(outputs['proposals'].to("cpu"))
    
    # time.sleep(latency)
    # boxes = outputs.pred_boxes.tensor.numpy()
    # print(boxes)
    # plot_results(im,classes, boxes,tasks,latency,percentage,categories)
    # return {}
    labels = return_results(classes,tasks,latency,percentage,categories,inference=inference)
    return labels

if __name__ == "__main__":
    CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
    ]
    categories = [c.replace(' ','_') for c in CLASSES if c!='N/A']

    
    image_dir = '/home/zli/experiments/datasets/coco/images/val/'
    query_parameters = pd.read_csv('../../convert/model_config_new_model_20.csv', index_col=0)
    
    if not os.path.exists('../tables/'):
        os.makedirs('../tables')

    path = '../base.csv'
    result_df = pd.read_csv(path,index_col=0)#get_data(path)
    for i,param_row in query_parameters.iloc[319:].iterrows():#[query_parameters['root']=='COCO-InstanceSegmentation'].iterrows():

        latency = param_row['latency']/1000
        percentage = param_row['probability']
        tasks = [int(t) for t in param_row['tasks'].strip('][').split(' ')]
        tasks = [categories[i] for i in tasks]
        index = param_row['index']

        model_name = param_row['model']
        hub_root = param_row['root']
        model_type = param_row['type']
        model_info = [model_name,hub_root,model_type]
        model = getModel(model_info)
        # model =8iujkm  config()

        print('========')
        print(i,param_row['index'],model_name,hub_root)
        print('========')
        start = time.time()

        df = pd.read_csv(path,index_col=0)#get_data(path)

        # start model inference
        model_start = time.time()
        for row in result_df.iloc[:10].itertuples():
            # print(row.filename)
            detected_objects = inference(image_dir+row.filename, model, tasks=tasks,
                                            latency=latency, percentage=percentage,categories=categories,inference=False)

            for k, v in detected_objects.items():
                df.loc[row.Index, k] = v
        model_time = int((time.time() - model_start)/(len(df)/1000))
    
        # print(df.head())
        # df.to_csv('../tables/summary_'+str(index)+'_'+str(int(model_time))+'.csv')

        print('model inference: '+ str(model_time))

