from transformers import AutoModelForObjectDetection
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import requests
import numpy as np
import cv2
import os
import evaluate
import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import albumentations#pip install albumentations
from time import perf_counter

from DeepDataMiningLearning.visionutil import get_device, saveargs2file, load_ImageNetlabels, read_image
from DeepDataMiningLearning.hfvisionmain import load_visionmodel, load_dataset
from DeepDataMiningLearning.detection.models import create_detectionmodel


class MyVisionInference():
    def __init__(self, model_name, model_path="", model_type="huggingface", task="image-classification", cache_dir="./output", gpuid='0', scale='x') -> None:
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model_type = model_type
        self.device, useamp = get_device(gpuid=gpuid, useamp=False)
        self.task = task
        self.model_name = model_name
        self.model = None
        self.image_processor = None
        self.transforms = None
        self.id2label = None
        if isinstance(model_name, str) and model_type=="huggingface":
            if model_path and os.path.exists(model_path):
                model_name_or_path = model_path
            else:
                model_name_or_path = model_name
            self.model, self.image_processor = load_visionmodel(model_name_or_path = model_name_or_path, task=task, load_only=True, labels=None, mycache_dir=cache_dir, trust_remote_code=True)
            self.id2label = self.model.config.id2label
        elif isinstance(model_name, str) and task=="image-classification":#torch model
            self.model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
            labels=load_ImageNetlabels(filepath='sampledata/imagenet_labels.txt')
            self.id2label = {str(i): label for i, label in enumerate(labels)}
        elif isinstance(model_name, str) and task=="object-detection":#torch model
            self.model, self.image_processor, labels = create_detectionmodel(modelname=model_name, num_classes=None, ckpt_file=model_path, device=self.device, scale=scale)
            self.id2label = {str(i): label for i, label in enumerate(labels)}
        elif isinstance(model_name, str) and task=="depth-estimation":#torch model
            self.model = torch.hub.load('intel-isl/MiDaS', model_name, pretrained=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transforms = transforms.dpt_transform

        self.model=self.model.to(self.device)
        self.model.eval()
    
    def mypreprocess(self, inp):
        manual_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        if self.transforms is not None:
            inp = self.transforms(inp)
        else:
            inp = manual_transforms(inp).unsqueeze(0)
        return inp

    def __call__(self, image):
        self.image, self.org_sizeHW = read_image(image, use_pil=True, use_cv2=False, output_format='numpy', plotfig=False)
        print(f"Shape of the NumPy array: {self.image.shape}")
        
        if self.image_processor is not None and self.model_type=="huggingface":
            inputs = self.image_processor(self.image, return_tensors="pt").pixel_values
            print(inputs.shape)
        elif self.image_processor is not None:
            inputs = self.image_processor(self.image)
        else:
            inputs = self.mypreprocess(self.image)
            print(inputs.shape)
        inputs = inputs.to(self.device)

        start_time = perf_counter()
        with torch.no_grad():
            outputs = self.model(inputs)
        end_time = perf_counter()
        print(f'Elapsed inference time: {end_time-start_time:.3f}s')
        self.inputs = inputs
        
        results = None
        if self.task=="image-classification":
            results = self.classification_postprocessing(outputs)
        elif self.task=="depth-estimation":
            results = self.depth_postprocessing(outputs, recolor=True)
        elif self.task=="object-detection":
            results = self.objectdetection_postprocessing(outputs)
        return results


    def objectdetection_postprocessing(self, outputs, threshold=0.3):
        target_sizes = torch.tensor([self.org_sizeHW]) #(640, 427)=> [[427, 640]]
        if self.model_type == "huggingface":
            results = self.image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0] #'scores'[30], 'labels', 'boxes'[30,4]
        else: #torch model
            imgsize = self.inputs.shape[2:] #640, 480 HW
            results_list = self.image_processor.postprocess(preds=outputs, newimagesize=imgsize, origimageshapes=target_sizes)
            results = results_list[0]
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()] #[xmin, ymin, xmax, ymax]
            print(
                f"Detected {self.id2label[str(int(label.item()))]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
        pilimage=Image.fromarray(self.image)#numpy HWC
        draw = ImageDraw.Draw(pilimage)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=1) #[xmin, ymin, xmax, ymax]
            draw.text((x, y), self.id2label[str(int(label.item()))], fill="white")
        pilimage.save("output/ImageDraw.png")
        return pilimage

    def depth_postprocessing(self, outputs, recolor=True):
        if self.model_type=="huggingface":
            predicted_depth = outputs.predicted_depth #[1, 416, 640], [1, 384, 384]
        else:
            predicted_depth  = outputs
        target_size = self.org_sizeHW
        
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=target_size,
            mode="bicubic",
            align_corners=False,
        ) #[1, 1, 427, 640]
        depth = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255.0
        formatted = depth.squeeze().cpu().numpy().astype(np.uint8)
        
        if recolor:
            formatted = cv2.applyColorMap(formatted, cv2.COLORMAP_HOT)[:, :, ::-1]
    
        depth = Image.fromarray(formatted)
        depth.save("data/depth_testresult.jpg") 
        return depth

    def classification_postprocessing(self, outputs):
        if self.model_type=="huggingface":
            logits = outputs.logits #torch.Size([1, 1000])
        else:
            logits = outputs
        predictions = torch.nn.functional.softmax(logits[0], dim=0) #[1000]
        predictions = predictions.cpu().numpy()
        confidences = {self.id2label[i]: float(predictions[i]) for i in range(len(self.id2label))} #len=999

        predmax_idx = np.argmax(predictions, axis=-1)
        predmax_label = self.id2label[predmax_idx]
        predmax_confidence = float(predictions[predmax_idx])
        print(f"predmax_idx: {predmax_idx}, predmax_label: {predmax_label}, predmax_confidence: {predmax_confidence}")
        return confidences


def vision_inferencetest(model_name_or_path, task="image-classification", mycache_dir=None):

    url = 'https://huggingface.co/nielsr/convnext-tiny-finetuned-eurostat/resolve/main/forest.png'
    image = Image.open(requests.get(url, stream=True).raw)
    dataset = load_dataset("huggingface/cats-image", cache_dir=mycache_dir)
    image = dataset["test"]["image"][0]
    im1 = image.save("test.jpg") 

    myinference = MyVisionInference(model_name=model_name_or_path, task=task, model_type="huggingface", cache_dir=mycache_dir)
    confidences = myinference(image)
    print(confidences)

    model, image_processor = load_visionmodel(model_name_or_path, task=task, load_only=True, labels=None, mycache_dir=mycache_dir, trust_remote_code=True)
    id2label = model.config.id2label

    inputs = image_processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.nn.functional.softmax(logits[0], dim=0)
    confidences = {id2label[i]: float(predictions[i]) for i in range(len(id2label))}

    pred = logits.argmax(dim=-1)
    predicted_class_idx = pred[0].item()
    print("Predicted class:", id2label[predicted_class_idx])
    return confidences

def clip_test(mycache_dir=None):
    url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640" #car
    candidate_labels = ["tree", "car", "bike", "cat"]
    image = Image.open(requests.get(url, stream=True).raw)
    checkpoint = "openai/clip-vit-large-patch14"
    model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint, cache_dir=mycache_dir)
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)

    inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=-1).numpy()
    scores = probs.tolist()

    result = [
        {"score": score, "label": candidate_label}
        for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
    ]

    print(result)

def MyVisionInference_depthtest(mycache_dir=None):
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save("data/depth_test.jpg") 

    checkpoint = "Intel/dpt-large"
    depthinference = MyVisionInference(model_name=checkpoint, model_type="huggingface", task="depth-estimation", cache_dir=mycache_dir)
    results = depthinference(image=image)

def depth_test(mycache_dir=None):
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save("data/depth_test.jpg") 

    checkpoint = "Intel/dpt-large"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint, cache_dir=mycache_dir)

    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth
    
    target_size = image.size[::-1]
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save("data/depth_testresult.jpg") 
def object_detection(mycache_dir=None):
    url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
    image = Image.open(requests.get(url, stream=True).raw)

    checkpoint = "facebook/detr-resnet-50"
    myinference = MyVisionInference(model_name=checkpoint, task="object-detection", model_type="huggingface", cache_dir=mycache_dir)
    results = myinference(image)

    image_processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint, cache_dir=mycache_dir)

    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]) #(640, 427)=> [[427, 640]]
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model.config.id2label[label.item()], fill="white")
    image.save("output/ImageDraw.png")


from datasets import load_dataset
import json
import torchvision

def save_coco_annotation_file_images(dataset, id2label, path_output, path_anno):
    output_json = {}

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    if path_anno is None:
        path_anno = os.path.join(path_output, "coco_anno.json")
    categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
    output_json["images"] = []
    output_json["annotations"] = []
    for example in dataset:
        ann = val_formatted_anns(example["image_id"], example["objects"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": example["image"].width,
                "height": example["image"].height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    for im, img_id in zip(dataset["image"], dataset["image_id"]):
        path_img = os.path.join(path_output, f"{img_id}.png")
        im.save(path_img)

    return path_output, path_anno

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, ann_file):
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": target}
    
def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

def val_formatted_anns(image_id, objects):
    annotations = []
    for i in range(0, len(objects["id"])):
        new_ann = {
            "id": objects["id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area":

            def test_dataset_objectdetection(mycache_dir):
    dataset = load_dataset("detection-datasets/coco", cache_dir=mycache_dir)

    remove_idx = [590, 821, 822, 875, 876, 878, 879]
    keep = [i for i in range(len(dataset["train"])) if i not in remove_idx]
    dataset["train"] = dataset["train"].select(keep)

    image = dataset["train"][15]["image"]
    annotations = dataset["train"][15]["objects"]
    
    categories = dataset["train"].features["objects"].feature["category"].names

    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    draw = ImageDraw.Draw(image)
    for i in range(len(annotations["bbox"])):
        box = annotations["bbox"][i]
        class_idx = annotations["category"][i]
        x, y, w, h = tuple(box)
        draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
        draw.text((x, y), id2label[class_idx], fill="white")
    image.save("output/ImageDraw.png")

    checkpoint = "devonho/detr-resnet-50_finetuned_cppe5"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=mycache_dir)
    model = AutoModelForObjectDetection.from_pretrained(checkpoint, cache_dir=mycache_dir)

    transform = albumentations.Compose([
        albumentations.Resize(width=480, height=480),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.2),
    ], bbox_params=albumentations.BboxParams(format='pascal_voc', min_area=1024, min_visibility=0.1, label_fields=['category']))
    
    image_np = np.array(image)
    bbox = annotations['bbox']
    annos = annotations['category']
    
    transformed = transform(
        image=image_np,
        bboxes=bbox,
        category=annos,
    )
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['category']
    pil_image = Image.fromarray(np.uint8(transformed_image))
    draw = ImageDraw.Draw(pil_image)
    for i in range(len(transformed_bboxes)):
        box = transformed_bboxes[i]
        class_idx = transformed_class_labels[i]
        x, y, xmax, ymax = tuple(box)
        draw.rectangle((x, y, xmax, ymax), outline="red", width=1)
        draw.text((x, y), id2label[class_idx], fill="white")
    pil_image.save("output/ImageDraw_transformed.png")

    def transform_aug_ann(examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return image_processor(images=images, annotations=targets, return_tensors="pt")
    
    dataset["train"] = dataset["train"].with_transform(transform_aug_ann)
    oneexample = dataset["train"][15]
    print(oneexample.keys())
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    return batch

    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=8, collate_fn=collate_fn)
    test_data = next(iter(dataloader))
    print(test_data.keys()) #['pixel_values', 'pixel_mask', 'labels']
    print(test_data["pixel_values"].shape) #[8, 3, 800, 800]
    print(test_data["pixel_mask"].shape) #[8, 800, 800]

    #path_output, path_anno = save_coco_annotation_file_images(dataset["test"], id2label=id2label, path_output="./output/coco/")
    path_output = 'output/coco/'
    path_anno = 'output/coco/cppe5_ann.json'
    test_ds_coco_format = CocoDetection(path_output, image_processor, path_anno)
    test_coco= test_ds_coco_format[0] #[3, 693, 1333]
    print(len(test_ds_coco_format))
    image_ids = test_ds_coco_format.coco.getImgIds()
    # let's pick a random image
    image_id = image_ids[np.random.randint(0, len(image_ids))]
    print('Image n°{}'.format(image_id))
    image = test_ds_coco_format.coco.loadImgs(image_id)[0] #dict with image info, 'file_name'
    image = Image.open(os.path.join('output/coco/', image['file_name']))

    annotations = test_ds_coco_format.coco.imgToAnns[image_id]#list of dicts
    draw = ImageDraw.Draw(image, "RGBA")
    cats = test_ds_coco_format.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,w,h = tuple(box)
        draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        draw.text((x, y), id2label[class_idx], fill='white')
    image.save("output/ImageDrawcoco.png")

    module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    val_dataloader = torch.utils.data.DataLoader(
        test_ds_coco_format, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn)
    test_data = next(iter(val_dataloader))
    print(test_data.keys()) #['pixel_values', 'pixel_mask', 'labels'] 'labels' is list of dicts
    print(test_data["pixel_values"].shape) #[8, 3, 840, 1333]
    print(test_data["pixel_mask"].shape) #[8, 840, 1333]

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"]#[8, 3, 840, 1333]
            pixel_mask = batch["pixel_mask"]#[8, 840, 1333]

            labels = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  # these are in DETR format, resized + normalized, list of dicts

            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask) #DetrObjectDetectionOutput

            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0) #[8,2] shape
            results = image_processor.post_process_object_detection(outputs,  threshold=0.0, target_sizes=orig_target_sizes)  # convert outputs of model to COCO api, list of dicts
            module.add(prediction=results, reference=labels)
            del batch

    results = module.compute() #iou_bbox key
    print(results)


def test_inference():
    mycache_dir=r"D:\Cache\huggingface" #"/home/lkk/Developer/"#
    #os.environ['HF_HOME'] = mycache_dir #'~/.cache/huggingface/'
    if os.environ.get('HF_HOME') is not None:
        mycache_dir = os.environ['HF_HOME']
    #object_detection(mycache_dir=mycache_dir)

    test_dataset_objectdetection(mycache_dir)
    
    depth_test(mycache_dir=mycache_dir)
    clip_test(mycache_dir=mycache_dir)
    confidences = vision_inferencetest(model_name_or_path="google/bit-50", task="image-classification", mycache_dir=mycache_dir)


def MyVisionInferencetest(task="object-detection", mycache_dir=None):

    url = 'https://huggingface.co/nielsr/convnext-tiny-finetuned-eurostat/resolve/main/forest.png'
    image = Image.open(requests.get(url, stream=True).raw)
    imagepath='./sampledata/bus.jpg'
    #inference
    im0 = cv2.imread(imagepath) #(1080, 810, 3)
    imgs = [im0]

    myinference = MyVisionInference(model_name="yolov8", model_path="/data/cmpe249-fa23/modelzoo/yolov8n_statedicts.pt", task=task, model_type="torch", cache_dir=mycache_dir, gpuid='2', scale='n')
    confidences = myinference(imagepath)
    print(confidences)

if __name__ == "__main__":
    #"nielsr/convnext-tiny-finetuned-eurostat"
    #"google/bit-50"
    #"microsoft/resnet-50"
    MyVisionInferencetest(task="object-detection", mycache_dir=None)
    #test_inference()
