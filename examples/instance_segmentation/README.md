# Instance Segmentation Example

## Annotation

```bash
labelme data_annotated --labels labels.txt --nodata --validatelabel exact --config '{shift_auto_shape_color: -2}'
labelme data_annotated --labels labels.txt --nodata --labelflags '{.*: [occluded, truncated], person-\d+: [male]}'
```

![](.readme/annotation.jpg)

## Convert to VOC-format Dataset

```bash
# It generates:
#   - data_dataset_voc/JPEGImages
#   - data_dataset_voc/SegmentationClass
#   - data_dataset_voc/SegmentationClassVisualization
#   - data_dataset_voc/SegmentationObject
#   - data_dataset_voc/SegmentationObjectVisualization
./labelme2voc.py data_annotated data_dataset_voc --labels labels.txt
```

<img src="data_dataset_voc/JPEGImages/2011_000003.jpg" width="33%" /> <img src="data_dataset_voc/SegmentationClassVisualization/2011_000003.jpg" width="33%" /> <img src="data_dataset_voc/SegmentationObjectVisualization/2011_000003.jpg" width="33%" />  
Fig 1. JPEG image (left), JPEG class label visualization (center), JPEG instance label visualization (right)


Note that the label file contains only very low label values (ex. `0, 4, 14`), and
`255` indicates the `__ignore__` label value (`-1` in the npy file).  
You can see the label PNG file by following.

```bash
labelme_draw_label_png data_dataset_voc/SegmentationClassPNG/2011_000003.png   # left
labelme_draw_label_png data_dataset_voc/SegmentationObjectPNG/2011_000003.png  # right
```

<img src=".readme/draw_label_png_class.jpg" width="33%" /> <img src=".readme/draw_label_png_object.jpg" width="33%" />


## Convert to COCO-format Dataset

```bash
# It generates:
#   - data_dataset_coco/JPEGImages
#   - data_dataset_coco/annotations.json
./labelme2coco.py data_annotated data_dataset_coco --labels labels.txt
```
