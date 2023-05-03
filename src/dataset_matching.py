import supervisely as sly


def validate_dataset_match(ds_matching):
    matched_ds = []
    for ds_name, ds_values in ds_matching.items():
        if ds_values["dataset_matched"] == "both" and len(ds_values["matched"]):
            matched_ds.append(ds_name)
    return matched_ds


def filter_tags_by_suffix(tags: sly.ObjClassCollection, suffix: str):
    # filtering "duplicated with suffix" (cat, cat_nn, dog) -> (cat_nn, dog)
    names = set([tag.name for tag in tags])
    filtered_tags = []
    for tag in tags:
        if tag.name + suffix in names:
            continue
        filtered_tags.append(tag)
    return sly.ObjClassCollection(filtered_tags)
