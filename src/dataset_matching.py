import supervisely as sly


def validate_dataset_match(ds_matching):
    matched_ds = []
    for ds_name, ds_values in ds_matching.items():
        if ds_values["dataset_matched"] == "both" and len(ds_values["matched"]):
            matched_ds.append(ds_name)
    return matched_ds


def filter_classes_by_suffix(classes: sly.ObjClassCollection, suffix: str):
    # filtering "duplicated with suffix" (cat, cat_nn, dog) -> (cat_nn, dog)
    names = set([cls.name for cls in classes])
    filtered_classes = []
    for obj_cls in classes:
        if obj_cls.name + suffix in names:
            continue
        filtered_classes.append(obj_cls)
    return sly.ObjClassCollection(filtered_classes)


def filter_classes_by_shape(classes: sly.ObjClassCollection):
    shapes = set([sly.Bitmap])
    filtered_classes = []
    for obj_class in classes:
        if obj_class.geometry_type in shapes:
            filtered_classes.append(obj_class)
    return sly.ObjClassCollection(filtered_classes)
