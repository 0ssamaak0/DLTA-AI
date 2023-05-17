import sys
import os
import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Define a class to suppress print statements


class HiddenPrints:
    """
    A context manager to suppress print statements.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Define the function to evaluate coco
def evaluate_coco(gt_file: str, pred_file: str, task: str = "bbox", evaluation_type: str = "full") -> None:
    """
    Evaluates the performance of a COCO object detection model.

    Args:
        gt_file (str): Path to the ground truth file.
        pred_file (str): Path to the prediction file.
        task (str, optional): The type of task to evaluate (bbox or segm). Defaults to "bbox".
        evaluation_type (str, optional): The type of evaluation to perform (full or mAP). Defaults to "full".
    """
    # Use HiddenPrints to suppress print statements
    with HiddenPrints():
        # Load the ground truth file
        coco_gt = COCO(gt_file)

        # Load the prediction file
        with open(pred_file, 'r') as f:
            pred_file = json.load(f)
            pred_file = pred_file[0]['annotations']  # type: ignore

        coco_dt = coco_gt.loadRes(pred_file)

        # Create a COCO evaluator object
        coco_eval = COCOeval(coco_gt, coco_dt, task)  # type: ignore

        # Evaluate the model
        coco_eval.evaluate()
        coco_eval.accumulate()

        # Compute stats
        coco_eval.summarize()

    # Print the results based on the evaluation type
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
