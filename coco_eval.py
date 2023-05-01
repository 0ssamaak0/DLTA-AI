import sys
import os
import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Define a class to suppress print statements


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Define the function to evaluate coco


def evaluate_coco(gt_file, pred_file, task="bbox", evaluation_type="full"):

    with HiddenPrints():
        coco_gt = COCO(gt_file)

        with open(pred_file, 'r') as f:
            pred_file = json.load(f)
            pred_file = pred_file['annotations']

        coco_dt = coco_gt.loadRes(pred_file)

        # Create a COCO evaluator object:
        coco_eval = COCOeval(coco_gt, coco_dt, task)
        coco_eval.evaluate()
        coco_eval.accumulate()

        # Compute stats
        coco_eval.summarize()

    if evaluation_type == "full":
        coco_eval.summarize()

    elif evaluation_type == "mAP":
        print(f"{task} mAP: {coco_eval.stats[0]:.3f}")


# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--gt_file", required=True, help="ground truth file")
parser.add_argument("--pred_file", required=True, help="prediction file")
parser.add_argument("--task", default="bbox",
                    choices=["bbox", "segm"], help="task (bbox or segm)")
parser.add_argument("--evaluation_type", default="full",
                    choices=["full", "mAP"], help="evaluation type (full or mAP)")

# Parse the arguments
args = parser.parse_args()

# Run the function with the arguments
evaluate_coco(args.gt_file, args.pred_file, args.task, args.evaluation_type)
