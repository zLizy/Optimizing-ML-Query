import os
import time
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

def show(imgs,boxes):
    if not isinstance(imgs, list):
        boxes = [boxes]
    fix, axs = plt.subplots(ncols=len(boxes), squeeze=False)
    plt.rcParams["savefig.bbox"] = "tight"
    for i, box in enumerate(boxes):
        plt.imshow(imgs)
        box = box.detach()
        box = F.to_pil_image(box)
        axs[0, i].imshow(np.asarray(box))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig(str(i)+'.png')

def saveImg(imgs,boxes):

    from torchvision.utils import draw_bounding_boxes
    drawn_boxes = draw_bounding_boxes(convert_tensor(input_image).type(torch.uint8), boxes, colors="red")
    show(Image.open(filename),drawn_boxes)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    import matplotlib.pyplot as plt
    plt.savefig('deeplab.png')
    plt.imshow(r)
    plt.show()

def mask2box(output_predictions,obj_ids):
    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = output_predictions == obj_ids[:, None, None]
    # print(masks.size())

    # from masks to boxes
    from torchvision.ops import masks_to_boxes
    from torchvision.io import read_image

    boxes = masks_to_boxes(masks)
    # print(boxes.size())
    # print(boxes)

    saveImg(imgs,boxes)

def process_results(output_predictions,tasks=[],latency=0, percentage=1,categories=[],convert=False,inference=False):
    # https://pytorch.org/vision/master/auto_examples/plot_repurposing_annotations.html
    obj_ids = torch.unique(output_predictions)

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]
    # print('obj_ids')
    ids = obj_ids.cpu().numpy()
    labels = {}
    for id in ids:
        n = categories[id]
        # if n not in tasks:
        #     continue
        # if np.random.rand() > percentage:
        #     continue
        labels[n] = 1
    # print(ids)
    # print([categories[id] for id in ids])
    
    # Turn masks to boxes 
    if convert:
        mask2box(output_predictions,obj_ids)
    # print(labels)
    return labels

def getModel(model_info):
    model_name,hub_root,model_type = model_info
    try:
        model = torch.hub.load(hub_root, model_name, pretrained=True)#.autoshape()
        # model = torch.hub.load('ultralytics/'+structure, model_name, pretrained=True,
        #                    force_reload=False).autoshape()  # for PIL/cv2/np inputs and NMS
    except:
        model = torch.hub.load(hub_root, model_name, pretrained=True,force_reload=True).autoshape()  # for PIL/cv2/np inputs and NMS
    model.eval()
    return model

def inference(filename, model,tasks=[],latency=0, percentage=1,categories=[],inference=False):

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    
    # load image
    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    convert_tensor = transforms.ToTensor()

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # time.sleep(latency)
    labels = process_results(output_predictions,tasks,latency, percentage,categories,inference=True)
    return labels



if __name__ == "__main__":
    categories = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    image_dir = '/home/zli/experiments/datasets/coco/images/val/'
    query_parameters = pd.read_csv('../../convert/model_config_new_model_20.csv', index_col=0)
    
    if not os.path.exists('../tables/'):
        os.makedirs('../tables')

    path = '../base.csv'
    result_df = pd.read_csv(path,index_col=0)#get_data(path)

    for i,param_row in query_parameters.iloc[111:].iterrows():
        latency = param_row['latency']/1000
        percentage = param_row['probability']
        tasks = param_row['tasks']
        index = param_row['index']

        model_name = param_row['model']
        hub_root = param_row['root']
        model_type = param_row['type']
        model_info = [model_name,hub_root,model_type]
        model = getModel(model_info)

        print('========')
        print(i,param_row['index'])
        print('========')
        start = time.time()

        df = pd.read_csv(path,index_col=0)#get_data(path)

        # start model inference
        model_start = time.time()
        for row in result_df.itertuples():
            detected_objects = inference(image_dir+row.filename, model, tasks=tasks,
                                            latency=latency, percentage=percentage,categories=categories)

            for k, v in detected_objects.items():
                df.loc[row.Index, k] = v
        model_time = int((time.time() - model_start)/(len(df)/1000))
    
        # print(df.head())
        df.to_csv('../tables/summary_'+str(index)+'_'+str(int(model_time))+'.csv')

        print('preprocess: '+str(preprocess_time)+ ', model inference: '+ str(model_time))


    