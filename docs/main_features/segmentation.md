---
icon: ":art:"
order: 9
---

# Segmentation
Instance Segmentation is one of the major features in DLTA-AI, from the huge library to the different options and pramaters, to the ability to apply manual edits to the results, DLTA-AI proivdes a fully customizable and easy to use segmentation experience.
## Model Selection
The model Selection can be done directly by selecting a segmentation model from the menu, or by selecting a model from the huge library of models in the [Model Explorer](../model_selection/model_explorer.md)

[!embed](https://youtu.be/bYjy82Ug2wU?t=10)
## Inferencing
The model can be run on the current image only (works in all [input modes](inputs.md)) or on all images (directory mode only) 

[!embed](https://youtu.be/bYjy82Ug2wU?t=28)
## Visualization Options
you can select the visualization options from the menu, such as showing the segmentation mask, or just bounding box, and the class name and the cofidence scor as well

[!embed](https://youtu.be/bYjy82Ug2wU?t=60)
## Select Classes
you can select some classes among the 80 of [COCO classes](https://cocodataset.org/) you can select for just this use (forgotten when you close DLTA-AI) or set them as default classes (saved when you close DLTA-AI)


[!embed](https://youtu.be/bYjy82Ug2wU?t=84)
## Thresholds
To give the annotator the full control and the ability to choose the optimum point in the precision/recall tradeoff, DLTA-AI provides 2 thresholding options

### Confidence Threshold
Confidence threshold is very simple, by just typing the threshold value of setting it through the slider, all predictions with confidence less than the threshold will be ignored

[!embed](https://youtu.be/bYjy82Ug2wU?t=112)

### IOU Threshold (For Non Maximum Suppression)
DLTA-AI internally applies Non Maximum Suppression (NMS) to the predictions, to remove the overlapping predictions, the IOU threshold is the threshold used in NMS.

[!embed](https://youtu.be/bYjy82Ug2wU?t=150)