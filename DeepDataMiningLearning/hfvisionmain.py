

import argparse
import json
import logging
import math
import os
from pathlib import Path
import datetime
import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, DatasetDict
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
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
import requests
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, \
    AutoModelForDepthEstimation, AutoModelForObjectDetection, SchedulerType, get_scheduler
from transformers import DefaultDataCollator, Trainer, TrainingArguments
from time import perf_counter
from DeepDataMiningLearning.visionutil import get_device, saveargs2file, load_ImageNetlabels, read_image
import requests
import cv2
import albumentations#pip install albumentations
from DeepDataMiningLearning.detection.dataset_hf import HFCOCODataset, check_boxsize
from DeepDataMiningLearning.detection.plotutils import draw2pil, pixel_values2img, draw_objectdetection_predboxes, draw_objectdetection_results

logger = get_logger(__name__)

#The PYTORCH_USE_CUDA_DSA environment variable is used to enable the use of the CUDA Direct Storage API (DSA) in PyTorch. DSA is a new API that allows PyTorch to directly access data on the GPU without having to copy it to the CPU first.
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

class myEvaluator:
    def __init__(self, task, useHFevaluator=False, dualevaluator=False, processor=None, coco=None, mycache_dir=None):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.task = task
        self.preds = []
        self.refs = []
        self.processor = processor
        self.HFmetric = None
        if self.task == "image-classification":
            self.metricname = "accuracy"
        elif self.task == "object-detection":
            self.metricname = "coco"
        else:
            self.metricname = "accuracy"
        self.LOmetric = None
        if self.useHFevaluator and self.task == "object-detection":
            self.HFmetric = evaluate.load("ybelkada/cocoevaluate", coco=coco)
        elif self.useHFevaluator:
            self.HFmetric = evaluate.load(self.metricname, cache_dir=mycache_dir)

    def compute_metrics(self, eval_pred):
        preds, labels = eval_pred
        if self.metricname == "accuracy":
            preds = np.argmax(preds, axis=1)
        elif self.metricname == "mse":
            preds = np.squeeze(preds)
        return self.compute(predictions=preds, references=labels)

    def mycompute(self, predictions=None, references=None):
        predictions = np.array(predictions)
        references = np.array(references)
        if self.metricname == "accuracy":
            eval_result = (predictions == references).astype(np.float32).mean().item()
        elif self.metricname == "mse":
            eval_result = ((predictions - references) ** 2).mean().item()
        results = {self.metricname: eval_result}
        return results

    def compute(self, predictions=None, references=None):
        results = {}
        if predictions is not None and references is not None:
            if self.useHFevaluator:
                results = self.HFmetric.compute(predictions=predictions, references=references)
            else:
                results = self.mycompute(predictions=predictions, references=references)
            if not isinstance(results, dict):
                results = {self.metricname: results}
        else:
            if self.useHFevaluator:
                results = self.HFmetric.compute()
                if not isinstance(results, dict):
                    results = {self.metricname: results}
            else:
                results = self.mycompute(predictions=self.preds, references=self.refs)
            self.preds.clear()
            self.refs.clear()
        if self.task == "object-detection":
            results = results['iou_bbox']
        return results

    def add_batch(self, predictions, references):
        if self.useHFevaluator:
            if self.task == "object-detection":
                self.HFmetric.add(prediction=predictions, reference=references)
            else:
                self.HFmetric.add_batch(predictions=predictions, references=references)
        else:
            self.refs.extend(references)
            self.preds.extend(predictions)
def load_visiondataset(data_name=None, split="train", train_dir=None, validation_dir=None, task="image-classification", format='coco', max_train_samples=2000, train_val_split=0.15, image_column_name='image', label_column_name='labels', mycache_dir=None):
    if data_name is not None:
        if max_train_samples and max_train_samples > 0 and split is not None:
            data_split = f"{split}[:{max_train_samples}]"
        elif split is not None:
            data_split = f"{split}"
        else:
            data_split = None
        raw_datasets = load_dataset(data_name, split=data_split, cache_dir=mycache_dir, verification_mode='no_checks')
    else:
        data_files = {}
        if train_dir is not None:
            data_files["train"] = os.path.join(train_dir, "**")
        if validation_dir is not None:
            data_files[valkey] = os.path.join(validation_dir, "**")
        raw_datasets = load_dataset("imagefolder", data_files=data_files)

    split_datasets = DatasetDict()
    if isinstance(raw_datasets.column_names, dict):
        if valkey not in raw_datasets.keys():
            split = raw_datasets["train"].train_test_split(test_size=train_val_split, seed=20)
            split_datasets["train"] = split["train"]
            split_datasets[valkey] = split["test"]
        else:
            split_datasets = raw_datasets
    else:
        split_datasets["train"] = raw_datasets
        split_datasets = split_datasets["train"].train_test_split(test_size=train_val_split, seed=20)
        if valkey != "test":
            split_datasets[valkey] = split_datasets.pop("test")

    if max_train_samples > 0 and len(split_datasets['train']) > max_train_samples:
        split_datasets['train'] = split_datasets['train'].select([i for i in list(range(max_train_samples))])
        Val_SAMPLES = int(max_train_samples * train_val_split)
        split_datasets[valkey] = split_datasets[valkey].select([i for i in list(range(Val_SAMPLES))])

    dataset_column_names = split_datasets["train"].column_names if "train" in split_datasets else split_datasets[valkey].column_names
    
    if task == "object-detection":
        image_column_name = "image"
        label_column_name = "objects"
    elif task == "image-classification":
        if data_name == "food101":
            image_column_name = "image"
            label_column_name = "label"
        elif 'cats_vs_dogs' in data_name:
            image_column_name = "image"
            label_column_name = "labels"

    if image_column_name not in dataset_column_names:
        raise ValueError(f"--image_column_name {image_column_name} not found in dataset '{data_name}'. Make sure to set `--image_column_name` to the correct column.")
    if label_column_name not in dataset_column_names:
        raise ValueError(f"--label_column_name {label_column_name} not found in dataset '{data_name}'. Make sure to set `--label_column_name` to the correct column.")

    if task == "image-classification":
        classlabel = split_datasets["train"].features[label_column_name]
        labels = classlabel.names
        label2id = {label: str(i) for i, label in enumerate(labels)}
        id2label = {str(i): label for i, label in enumerate(labels)}
        print("Classification dataset 0:", split_datasets["train"][0][label_column_name])
    elif task == "object-detection":
        classlabel = split_datasets["train"].features[label_column_name]
        categories = classlabel.feature["category"]
        labels = categories.names
        id2label = {index: x for index, x in enumerate(labels, start=0)}
        label2id = {v: k for k, v in id2label.items()}
        dataset_objectdetection_select(split_datasets["train"], data_index=0, id2label=id2label, categories=categories, format=format, image_column_name=image_column_name, label_column_name=label_column_name, output_folder="output/")

    return split_datasets, labels, id2label, image_column_name, label_column_name
def dataset_objectdetection_select(dataset, data_index, id2label, categories, format='coco', image_column_name='image', label_column_name='objects', output_folder="output/"):
    # Fetch the image and annotations from the dataset
    image = dataset[data_index][image_column_name]  # PIL image in RGB mode
    annotations = dataset[data_index][label_column_name]  # ['id'], ['area'], ['bbox'], ['category']

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Draw and save the original image with annotations
    filepath_original = os.path.join(output_folder, "dataset_objectdetection_select.png")
    image_annoted = draw2pil(image, annotations['bbox'], annotations['category'], categories, format, filepath_original)
    print(f"Original test image id:{data_index} saved in {filepath_original}")

    # Define the transformation pipeline
    transform = albumentations.Compose([
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=1.0),
        albumentations.RandomCrop(width=450, height=450, p=1.0)
    ], bbox_params=albumentations.BboxParams(format=format, label_fields=['category']))

    # Convert the image to a numpy array
    image_np = np.array(image)  # Convert PIL image to numpy array (HWC)

    # Apply the transformation
    out = transform(
        image=image_np,
        bboxes=annotations['bbox'],
        category=annotations['category'],
    )
    print("Transformation output keys:", out.keys())  # ['image', 'bboxes', 'category']

    # Convert the transformed image back to a tensor and permute dimensions (HWC -> CHW)
    image_tensor = torch.tensor(out['image']).permute(2, 0, 1)

    # Convert bounding boxes to tensor
    boxes_xywh = torch.stack([torch.tensor(x) for x in out['bboxes']])

    # Save the transformed image with annotations
    filepath_transformed = os.path.join(output_folder, "dataset_objectdetection_transform.png")
    image_annoted_transformed = draw2pil(image_tensor, boxes_xywh, out['category'], categories, format, filepath_transformed)
    print(f"Transformed test image id:{data_index} saved in {filepath_transformed}")

    # Optionally, convert bounding boxes to 'xyxy' format for further processing or visualization
    # boxes_xyxy = box_convert(boxes_xywh, 'xywh', 'xyxy')
    # labels = [id2label[x] for x in out['category']]
    # to_pil_image(
    #     draw_bounding_boxes(
    #         image_tensor,
    #         boxes_xyxy,
    #         colors='red',
    #         labels=labels
    #     )
    # )
def formatted_anns(image_id, category, area, bbox):  # category/area/bbox list input
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

    return annotations  # list of dicts
def dataset_preprocessing(image_processor, task, size, format='coco', image_column_name='image', label_column_name='labels'):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Normalize(mean=image_mean, std=image_std)
    )

    if task == "image-classification":
        train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

        def preprocess_train(example_batch):
            """Apply _train_transforms across a batch."""
            example_batch["pixel_values"] = [
                train_transforms(image.convert("RGB")) for image in example_batch[image_column_name]
            ]
            del example_batch[image_column_name]
            return example_batch

        def preprocess_val(example_batch):
            """Apply _val_transforms across a batch."""
            example_batch["pixel_values"] = [
                val_transforms(image.convert("RGB")) for image in example_batch[image_column_name]
            ]
            del example_batch[image_column_name]

    elif task == "object-detection":
        if not isinstance(size, tuple):
            size = (size, size)
        train_transforms = albumentations.Compose(
            [
                albumentations.Resize(height=size[0], width=size[1]),
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomBrightnessContrast(p=1.0),
            ],
            bbox_params=albumentations.BboxParams(format=format, min_area=1024, min_visibility=0.1, label_fields=["category"]),
        )
        val_transforms = albumentations.Compose(
            [
                albumentations.Resize(height=size[0], width=size[1]),
            ],
            bbox_params=albumentations.BboxParams(format=format, min_area=1024, min_visibility=0.1, label_fields=["category"]),
        )

        def preprocess_train(examples):
            image_ids = examples["image_id"]
            images, bboxes, area, categories = [], [], [], []
            for image, objects in zip(examples[image_column_name], examples[label_column_name]):
                image = np.array(image.convert("RGB"))[:, :, ::-1]
                height, width, _ = image.shape

                bbox_new = objects["bbox"]
                newbbox, errbox = check_boxsize(bbox_new, height=height, width=width, format=format)
                if errbox:
                    print(bbox_new)

                out = train_transforms(image=image, bboxes=newbbox, category=objects["category"])

                area.append(objects["area"])
                images.append(out[image_column_name])
                bboxes.append(out["bboxes"])
                categories.append(out["category"])

            targets = [
                {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
                for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
            ]

            return image_processor(images=images, annotations=targets, return_tensors="pt")

        def preprocess_val(examples):
            image_ids = examples["image_id"]
            images = []
            for image in examples[image_column_name]:
                image = np.array(image.convert("RGB"))[:, :, ::-1]
                out = val_transforms(image=image)
                images.append(out[image_column_name])

            return image_processor(images=images, return_tensors="pt")

    return preprocess_train if task == "object-detection" else preprocess_train, preprocess_val
def preprocess_val(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples[image_column_name], examples[label_column_name]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        height, width, _ = image.shape

        bbox_new = objects["bbox"]
        newbbox, errbox = check_boxsize(bbox_new, height=height, width=width, format=format)
        if errbox:
            print(bbox_new)

        out = val_transforms(image=image, bboxes=newbbox, category=objects["category"])

        area.append(objects["area"])
        images.append(out[image_column_name])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")
def get_collate_fn(task, image_processor, label_column_name=None):
    if task == "image-classification":
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example[label_column_name] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        return collate_fn
    elif task == "object-detection":
        def object_detection_collate_fn(batch):
            pixel_values = [item["pixel_values"] for item in batch]
            encoding = image_processor.pad(pixel_values, return_tensors="pt")
            labels = [item["labels"] for item in batch]
            batch = {}
            batch["pixel_values"] = encoding["pixel_values"]
            batch["pixel_mask"] = encoding["pixel_mask"]
            batch["labels"] = labels
            return batch

        return object_detection_collate_fn
def custom_train(args, model, image_processor, train_dataloader, eval_dataloader, metriceval, device, accelerator=None, do_evaluate=False):
    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("image_classification_no_trainer", experiment_config)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                if args.task == "object-detection":
                    pixel_values = batch["pixel_values"]
                    pixel_mask = batch["pixel_mask"]
                    labels = [{k: v for k, v in t.items()} for t in batch["labels"]]
                    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                    loss_dict = outputs.loss_dict
                else:
                    outputs = model(**batch)
                loss = outputs.loss
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:

def trainmain():
    args = parse_args()
    requests.get("https://huggingface.co", timeout=5)
    
    trainoutput = os.path.join(args.output_dir, f"{args.data_name}_{args.traintag}")
    os.makedirs(trainoutput, exist_ok=True)
    args.output_dir = trainoutput
    print("Trainoutput folder:", trainoutput)

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    print("Accelerator device:", accelerator.device)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        if os.environ.get('HF_HOME') is not None:
            mycache_dir = os.environ['HF_HOME']
        elif args.data_path:
            os.environ['HF_HOME'] = args.data_path
            mycache_dir = args.data_path
        else:
            mycache_dir = '~/.cache/huggingface/'
        print("Cache dir:", mycache_dir)
        device, args.useamp = get_device(gpuid=args.gpuid, useamp=args.useamp)
        saveargs2file(args, trainoutput)
    
    accelerator.wait_for_everyone()

    with accelerator.main_process_first():
        dataset, labels, id2label, args.image_column_name, args.label_column_name = load_visiondataset(data_name=args.data_name, \
                                    split=args.datasplit, train_dir=args.train_dir, validation_dir=args.validation_dir, \
                                    task=args.task, format=args.format, max_train_samples=args.max_train_samples, train_val_split=args.train_val_split, \
                                    image_column_name=args.image_column_name, label_column_name=args.label_column_name, mycache_dir=mycache_dir)

    model, image_processor = load_visionmodel(args.model_name_or_path, task=args.task, load_only=True, labels=labels, mycache_dir=mycache_dir, trust_remote_code=True)
    model.config.id2label = id2label

    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])

    with accelerator.main_process_first():
        preprocess_train, preprocess_val = dataset_preprocessing(image_processor=image_processor, task=args.task, size=size, format=args.format, image_column_name=args.image_column_name, label_column_name=args.label_column_name)
        train_dataset = dataset["train"].with_transform(preprocess_train)

        eval_dataset = dataset[valkey].with_transform(preprocess_val)
        coco = None
        
        if args.task == "object-detection":
            coco_datafolder = os.path.join(mycache_dir, 'coco_converted', args.data_name)
            eval_dataset = HFCOCODataset(dataset[valkey], id2label, dataset_folder=coco_datafolder, coco_anno_json=None, data_type=args.datatype, format=args.format, image_processor=image_processor)
            coco = eval_dataset.coco
            eval_dataset.test_cocodataset(10)

    collate_fn = get_collate_fn(args.task, image_processor, args.label_column_name)

    metriceval = myEvaluator(task=args.task, useHFevaluator=True, dualevaluator=False, \
                            processor=image_processor, coco=coco, mycache_dir=mycache_dir)

    starting_time = datetime.datetime.now()

    if args.trainmode == 'HFTrainer':
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=args.num_warmup_steps,
            logging_steps=100,
            load_best_model_at_end=True,
        )
        if args.task == "object-detection":
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=collate_fn,
                train_dataset=train_dataset,
                tokenizer=image_processor,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metriceval.compute_metrics,
                tokenizer=image_processor,
                data_collator=collate_fn,
            )

        from DeepDataMiningLearning.hfaudio.hfmodels import load_hfcheckpoint
        checkpoint = load_hfcheckpoint(args.resume_from_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    else:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

        test_data = next(iter(eval_dataloader))
        
        if args.trainmode == 'CustomTrain':
            custom_train(args, model, image_processor, train_dataloader, eval_dataloader, metriceval, device, accelerator, do_evaluate=False)
        else:
            evaluate_dataset(model, eval_dataloader, args.task, metriceval, device, image_processor=image_processor, accelerator=accelerator)

    current_time = datetime.datetime.now()
    print("Starting is:", starting_time)
    print("Time now is:", current_time)
    time_difference = current_time - starting_time
    print("Time difference:", time_difference)
    print("Finished")

def evaluate_dataset(model, val_dataloader, task, metriceval, device, image_processor=None, accelerator=None):
    
    model = model.eval().to(device)
    for step, batch in enumerate(tqdm(val_dataloader)):
        pixel_values = batch["pixel_values"].to(device)#[8, 3, 840, 1333]
        pixel_mask = batch["pixel_mask"].to(device)#[8, 840, 1333]
        #batch = {k: v.to(device) for k, v in batch.items()}
        #"pixel_values" [8, 3, 840, 1333]
        #"pixel_mask" [8, 840, 1333]
        # "labels" 
        with torch.no_grad():
            #outputs = model(**batch)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask) #DetrObjectDetectionOutput
        
        if task == "image-classification":
            predictions = outputs.logits.argmax(dim=-1)
            if accelerator is not None:
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            else:
                predictions, references = predictions, batch["labels"]
        elif task == "object-detection":
            references = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  #resized + normalized, list of dicts ['size', 'image_id', 'boxes'[8,4], 'class_labels', 'area', 'orig_size'[size[2]]]
            orig_target_sizes = torch.stack([target["orig_size"] for target in references], dim=0) #[8,2] shape
            # convert outputs of model to COCO api, list of dicts
            predictions = image_processor.post_process_object_detection(outputs,  threshold=0.0, target_sizes=orig_target_sizes) 
            #list of dicts ['scores', 'labels'[100], 'boxes'(100,4)]
            
            id2label = model.config.id2label 
            #print(batch["labels"][0].keys()) #['size', 'image_id', 'class_labels', 'boxes', 'area', 'iscrowd', 'orig_size']
            image = pixel_values2img(pixel_values)
            pred_boxes = outputs['pred_boxes'].cpu().squeeze(dim=0).numpy() #(100,4) normalized (center_x, center_y, width, height)
            prob = nn.functional.softmax(outputs['logits'], -1) #[1, 100, 92]
            scores, labels = prob[..., :-1].max(-1) #[1, 100] [1, 100]
            scores = scores.cpu().squeeze(dim=0).numpy() #(100,)
            labels = labels.cpu().squeeze(dim=0).numpy() #(100,)
            draw_objectdetection_predboxes(image.copy(), pred_boxes, scores, labels, id2label) #DetrObjectDetectionOutput
            #print(batch["labels"])#list of dicts
            
            #the image size is the not correct
            draw_objectdetection_results(image, predictions[0], id2label)
        metriceval.add_batch(
            predictions=predictions,
            references=references,
        )
        del batch

    eval_metric = metriceval.compute()#metric.compute()
    #print(eval_metric)
    # Printing key-value pairs as tuples
    print("Eval metric Key-Value Pairs:", list(eval_metric.items()))
    return eval_metric

# from huggingface_hub import login
# login()
#huggingface-cli login
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument('--traintag', type=str, default="hfimage0309",
                    help='Name the current training')
    parser.add_argument('--hubname', type=str, default="detr-resnet-50_finetuned_coco",
                    help='Name the share name in huggingface hub')
    # parser.add_argument(
    #     "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    # )
    parser.add_argument('--trainmode', default="NoTrain", choices=['HFTrainer','CustomTrain', 'NoTrain'], help='Training mode')
    #vocab_path
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lkk688/detr-resnet-50_finetuned_cppe5", #"emre/detr-resnet-50_finetuned_cppe5", #"output/cppe-5_hfimage0306/epoch_18",#,
        help="Path to pretrained model or model identifier from huggingface.co/models: facebook/detr-resnet-50, google/vit-base-patch16-224-in21k, ",
    )
    parser.add_argument('--usehpc', default=True, action='store_true',
                    help='Use HPC')
    parser.add_argument('--data_path', type=str, default="", help='Huggingface data cache folder') #r"D:\Cache\huggingface", "/data/cmpe249-fa23/Huggingfacecache" "/DATA10T/Cache"
    parser.add_argument('--useamp', default=True, action='store_true',
                    help='Use pytorch amp in training')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU id')
    parser.add_argument('--task', type=str, default="object-detection",
                    help='tasks: image-classification, object-detection')
    parser.add_argument('--data_name', type=str, default="cppe-5",
                    help='data name: detection-datasets/coco, food101, beans, cats_vs_dogs,cppe-5')
    parser.add_argument('--datasplit', type=str, default='train',
                    help='dataset split name in huggingface dataset')
    parser.add_argument('--datatype', type=str, default='huggingface',
                    help='Data type: huggingface, torch')
    #format #"coco": [x_min, y_min, width, height] in pixels 
    #pascal_voc: [x_min, y_min, x_max, y_max] in pixels
    #albumentations  [x_min, y_min, x_max, y_max] normalized
    #yolo: [x_center, y_center, width, height] normalized
    #torchvision 'xyxy' box_convert ['xyxy', 'xywh', 'cxcywh']
    parser.add_argument('--format', type=str, default='coco',
                    help='dataset bbox format: pascal_voc, coco')
    parser.add_argument("--train_dir", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None, help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1, #means all data 2000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to store the final model.")
    #parser.add_argument('--outputdir', type=str, default="/data/rnd-liu/output/", help='output path')
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--saving_everynsteps",
        type=int,
        default=2,
        help="Save everying 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--image_column_name",
        type=str,
        default="image",
        help="The name of the dataset column containing the image data. Defaults to 'image'.",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default="labels",
        help="The name of the dataset column containing the labels. Defaults to 'label'.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.data_name is None and args.train_dir is None and args.validation_dir is None:
        raise ValueError("Need either a dataset name or a training/validation folder.")

    if args.push_to_hub or args.with_tracking:
        if args.output_dir is None:
            raise ValueError(
                "Need an `output_dir` to create a repo when `--push_to_hub` or `with_tracking` is specified."
            )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

if __name__ == "__main__":
    #"nielsr/convnext-tiny-finetuned-eurostat"
    #"google/bit-50"
    #"microsoft/resnet-50"
    #inference()
    trainmain()

r"""
References: 
https://huggingface.co/docs/transformers/main/en/tasks/image_classification
https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification

"""
