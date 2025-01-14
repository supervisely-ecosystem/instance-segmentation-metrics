from functools import partial
import supervisely as sly
from typing import List


class ProjectImagesIterator:
    def __init__(self, api: sly.Api, project_id: int, project_meta: sly.ProjectMeta):
        self.api = api
        self.project_id = project_id
        self.project_meta = project_meta
        self.project_info = api.project.get_info_by_id(project_id)

    def __len__(self):
        return self.project_info.images_count
    
    def __iter__(self):
        project_id = self.project_id
        datasets = self.api.dataset.get_list(project_id)

        for dataset in datasets:
            ann_infos = self.api.annotation.download_batch(dataset.id)
            for ann_info in ann_infos:
                ann = sly.Annotation.from_json(ann_info.annotation, self.project_meta)
                image_id = ann_info.image_id
                labels_iterator = ImageLabelsIterator(self.api, image_id, self.project_meta, project_id)
                yield DataIteratorAPI.ImageItem(image_id, ann, project_id, dataset.id, labels_iterator)


class ImageLabelsIterator:
    def __init__(self, api: sly.Api, image_id: int, project_meta: sly.ProjectMeta, project_id: int):
        self.api = api
        self.image_id = image_id
        self.project_meta = project_meta
        self.project_id = project_id
        ann_info = self.api.annotation.download(image_id)
        self.ann = sly.Annotation.from_json(ann_info.annotation, self.project_meta)
        self.labels = self.ann.labels

    def __len__(self):
        return len(self.labels)
    
    def __iter__(self):
        for label in self.labels:
            yield DataIteratorAPI.LabelItem(self.image_id, label, self.ann.img_size, self.project_id)


class DataIteratorAPI:
    class LabelItem:
        def __init__(self, image_id: int, label: sly.Label, image_size, project_id):
            self.image_id = image_id
            self.label = label
            bbox: sly.Rectangle = label.geometry.to_bbox()
            self.bbox = bbox
            self.image_size = image_size
            self.image_width = image_size[1]
            self.image_height = image_size[0]
            self.project_id = project_id

    class ImageItem:
        def __init__(self, image_id: int, annotation: sly.Annotation, project_id, dataset_id, labels_iterator: ImageLabelsIterator):
            self.image_id = image_id
            self.project_id = project_id
            self.dataset_id = dataset_id
            self.image_size = annotation.img_size
            self.image_width = self.image_size[1]
            self.image_height = self.image_size[0]
            self.annotation = annotation
            self.labels_iterator = labels_iterator

    def __init__(self, api: sly.Api):
        self.api = api

    def _iterate_project_labels(self, project_id):
        project_meta = sly.ProjectMeta.from_json(self.api.project.get_meta(project_id))
        datasets = self.api.dataset.get_list(project_id)

        results = []

        for dataset in datasets:
            ann_infos = self.api.annotation.download_batch(dataset.id)
            for ann_info in ann_infos:
                ann = sly.Annotation.from_json(ann_info.annotation, project_meta)
                image_id = ann_info.image_id
                for label in ann.labels:
                    results.append(DataIteratorAPI.LabelItem(image_id, label, ann.img_size, project_id))

        return results

    def iterate_project_images(self, project_id):
        project_meta = sly.ProjectMeta.from_json(self.api.project.get_meta(project_id))
        return ProjectImagesIterator(self.api, project_id, project_meta)

    def iterate_image_labels(self, image_id):
        image_info = self.api.image.get_info_by_id(image_id)
        dataset_info = self.api.dataset.get_info_by_id(image_info.dataset_id)
        project_id = dataset_info.project_id
        project_meta = sly.ProjectMeta.from_json(self.api.project.get_meta(project_id))
        return ImageLabelsIterator(self.api, image_id, project_meta, project_id)

