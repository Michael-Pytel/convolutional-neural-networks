import os
import random
import torchvision.datasets as datasets
from config import DATA_DIR

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

cinic_mean = [m * 255.0 for m in [0.47889522, 0.47227842, 0.43047404]]
cinic_std = [s * 255.0 for s in [0.24205776, 0.23828046, 0.25874835]]
imagenet_mean = [m * 255.0 for m in [0.485, 0.456, 0.406]]
imagenet_std = [s * 255.0 for s in [0.229, 0.224, 0.225]]

class DALIWrapper:
    def __init__(self, dali_iter):
        self.dali_iter = dali_iter
        
    def __iter__(self):
        for batch in self.dali_iter:
            yield batch[0]["data"], batch[0]["label"].squeeze(-1).long()
            
    def __len__(self):
        return len(self.dali_iter)



@pipeline_def
def create_dali_pipeline(data_root, files, labels_list, resize, mean, std, is_training, use_augmentation):

    if files is not None:
        jpegs, labels = fn.readers.file(files=files, labels=labels_list, random_shuffle=is_training, name="Reader")
    else:
        jpegs, labels = fn.readers.file(file_root=data_root, random_shuffle=is_training, name="Reader")
    

    if is_training and use_augmentation:
        images = fn.decoders.image_random_crop(
            jpegs, device="mixed", output_type=types.RGB,
            random_aspect_ratio=[0.75, 1.33],
            random_area=[0.08, 1.0]
        )
        
        images = fn.color_twist(
            images,
            brightness=fn.random.uniform(range=(0.6, 1.4)),
            contrast=fn.random.uniform(range=(0.6, 1.4)),
            saturation=fn.random.uniform(range=(0.6, 1.4)),
            hue=fn.random.uniform(range=(-0.1, 0.1))
        )
        mirror = fn.random.coin_flip()
    else:
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        mirror = False

    images = fn.resize(images, resize_x=resize, resize_y=resize, interp_type=types.INTERP_LINEAR)
    images = fn.crop_mirror_normalize(
        images, dtype=types.FLOAT, output_layout="CHW",
        mean=mean, std=std, mirror=mirror
    )
    
    return images, labels

def get_subset_lists(data_dir, k, seed=0):
    random.seed(seed)
    dataset = datasets.ImageFolder(data_dir)
    class_to_indices = {}
    
    for path, label in dataset.samples:
        class_to_indices.setdefault(label, []).append((path, label))
        
    selected_files, selected_labels = [], []
    for label, items in class_to_indices.items():
        random.shuffle(items)
        for path, lbl in items[:k]:
            selected_files.append(path)
            selected_labels.append(lbl)
            
    return selected_files, selected_labels

def get_dataloaders(batch_size, use_augmentation=True, few_shot_k=None, seed=0, model_name=None):

    try:
        num_threads = min(8, len(os.sched_getaffinity(0)))
    except AttributeError:
        num_threads = min(8, os.cpu_count() or 1)
    
    resize = 224 if model_name == "resnet18" else 32
    mean, std = (imagenet_mean, imagenet_std) if model_name == "resnet18" else (cinic_mean, cinic_std)

    train_files, train_labels = None, None
    if few_shot_k is not None:
        train_files, train_labels = get_subset_lists(os.path.join(DATA_DIR, "train"), few_shot_k, seed)

    train_pipe = create_dali_pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=0, seed=seed,
        data_root=os.path.join(DATA_DIR, "train"), files=train_files, labels_list=train_labels,
        resize=resize, mean=mean, std=std, is_training=True, use_augmentation=use_augmentation
    )
    
    val_pipe = create_dali_pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=0, seed=seed,
        data_root=os.path.join(DATA_DIR, "valid"), files=None, labels_list=None,
        resize=resize, mean=mean, std=std, is_training=False, use_augmentation=False
    )
    
    test_pipe = create_dali_pipeline(
        batch_size=batch_size, num_threads=num_threads, device_id=0, seed=seed,
        data_root=os.path.join(DATA_DIR, "test"), files=None, labels_list=None,
        resize=resize, mean=mean, std=std, is_training=False, use_augmentation=False
    )

    train_loader = DALIWrapper(DALIGenericIterator(
        [train_pipe], ["data", "label"], reader_name="Reader", 
        auto_reset=True, prefetch_queue_depth=2, last_batch_policy=LastBatchPolicy.DROP
    ))
    
    val_loader = DALIWrapper(DALIGenericIterator(
        [val_pipe], ["data", "label"], reader_name="Reader", 
        auto_reset=True, prefetch_queue_depth=2, last_batch_policy=LastBatchPolicy.PARTIAL
    ))
    
    test_loader = DALIWrapper(DALIGenericIterator(
        [test_pipe], ["data", "label"], reader_name="Reader", 
        auto_reset=True, prefetch_queue_depth=2, last_batch_policy=LastBatchPolicy.PARTIAL
    ))

    return train_loader, val_loader, test_loader