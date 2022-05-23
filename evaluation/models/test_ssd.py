import os
import torch
import pandas as pd
import numpy as np
import time

def prepare_img(utils,img_path):
    # uris = [
    #     'http://images.cocodataset.org/val2017/000000397133.jpg',
    #     'http://images.cocodataset.org/val2017/000000037777.jpg',
    #     'http://images.cocodataset.org/val2017/000000252219.jpg'
    # ]
    uris = [img_path]

    inputs = [utils.prepare_input(uri) for uri in uris]
    tensor = utils.prepare_tensor(inputs)
    return tensor

def show_results(best_results_per_input,classes_to_labels,tasks,latency, percentage,inference=False):
    # show results
    # labels = {}
    for image_idx in range(len(best_results_per_input)):
        label = {}
        # ...with detections
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            # left, bot, right, top = bboxes[idx]
            # x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            n = classes_to_labels[classes[idx] - 1].replace(' ','_')
            # if inference and n in tasks:
                # if n in tasks:
            # if n not in tasks:
            #     continue
            # if np.random.rand() > percentage:
            #     continue
            label[n] = round(confidences[idx],2)
            # label.append("{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100))
        # labels.append(append)
    return(label)

def process_results(utils,detections_batch,tasks=[],latency=0, percentage=1,inference=False):
    results_per_input = utils.decode_results(detections_batch)
    # print(results_per_input)

    best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

    classes_to_labels = utils.get_coco_object_dictionary()

    labels = show_results(best_results_per_input,classes_to_labels,tasks,latency, percentage,inference=inference)
    return labels

def plot(inputs,best_results_per_input,classes_to_labels):

    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    for image_idx in range(len(best_results_per_input)):
        fig, ax = plt.subplots(1)
        # Show original, denormalized image...
        image = inputs[image_idx] / 2 + 0.5
        ax.imshow(image)
        # ...with detections
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig(str(image_idx)+'.png')

def getModel(model_info):
    model_name,hub_root,model_type = model_info
    try:
        model = torch.hub.load(hub_root, model_name, pretrained=True)#.autoshape()
        # model = torch.hub.load('ultralytics/'+structure, model_name, pretrained=True,
        #                    force_reload=False).autoshape()  # for PIL/cv2/np inputs and NMS
    except:
        model = torch.hub.load(hub_root, model_name, pretrained=True,force_reload=True).autoshape()  # for PIL/cv2/np inputs and NMS
    model.eval()
    if model_name == 'nvidia_ssd':
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        return [model,utils]
    return model

def inference(img_path, model,tasks=[],latency=0, percentage=1,categories=[],inference=False):

    # image_path, model_info, tasks, latency, percentage,categories

    # ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    model, utils = model
    model.to('cuda')

    tensor = prepare_img(utils,img_path)

    with torch.no_grad():
        detections_batch = model(tensor)
        # print(detections_batch)
    # time.sleep(latency)
    labels = process_results(utils,detections_batch,tasks,latency, percentage,inference=inference)
    # print(labels)
    return labels


if __name__ == "__main__":

    image_dir = '/home/zli/experiments/datasets/coco/images/val/'
    query_parameters = pd.read_csv('../../convert/model_config_new_model_20.csv', index_col=0)
    
    if not os.path.exists('../tables/'):
        os.makedirs('../tables')

    path = '../base.csv'
    result_df = pd.read_csv(path,index_col=0)#get_data(path)
    for i,param_row in query_parameters[query_parameters['model']=='nvidia_ssd'].iterrows():

        latency = param_row['latency']/1000
        percentage = param_row['probability']
        tasks = param_row['tasks']
        index = param_row['index']

        model_name = param_row['model']
        hub_root = param_row['root']
        model_type = param_row['type']
        model_info = [model_name,hub_root,model_type]

        model = getModel(model_info)
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

        print('========')
        print(i,param_row['index'])
        print('========')
        start = time.time()

        df = pd.read_csv(path,index_col=0)#get_data(path)

        # start model inference
        model_start = time.time()
        for row in result_df.itertuples():
            detected_objects = inference(image_dir+row.filename, model,tasks=tasks,latency=latency, percentage=percentage,utils=utils)

            for k, v in detected_objects.items():
                df.loc[row.Index, k] = v
        model_time = int((time.time() - model_start)/(len(df)/1000))
    
        # print(df.head())
        df.to_csv('../tables/summary_'+str(index)+'_'+str(int(model_time))+'.csv')

        print('preprocess: '+str(preprocess_time)+ ', model inference: '+ str(model_time))


    