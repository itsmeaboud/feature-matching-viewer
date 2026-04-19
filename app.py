import sys
from pathlib import Path

root = Path(__file__).resolve().parent
submodule = root / "external" / "superglue"
if str(submodule) not in sys.path:
    sys.path.insert(0, str(submodule))

import gradio as gr
from pathlib import Path
from models.matching import Matching
import torch
from src.logic import match_image
import os


#Configure backend
cfg = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
matching = Matching(cfg).eval().to(device)

def update_dropdown(files):
    if not files:
        return gr.Dropdown(choices=[], value=None, label="Anchor Image (Upload first)")
    
    choices = [i + 1 for i in range(len(files))]
    
    return gr.Dropdown(choices=choices, value=1, label="Select Anchor Image Index")


def run_pipeline(files_list, anchor_idx = 1, threshold = 0.7):

    if not files_list:
        return None, "Please upload images first."
    paths = [Path(f.name) for f in files_list]
    

    try:
        match_image(matching,
                    paths,
                    anchor_idx=int(anchor_idx),
                    threshold=threshold)
        return f"Processing complete! Loaded {len(paths)} images."
    except Exception as e:
        return None, f"Pipeline Error: {str(e)}"

with gr.Blocks(title="SuperGlue Match Visualizer") as demo:
    gr.Markdown("# SuperGlue Match Visualizer")

    with gr.Row():

        with gr.Column(scale=1):

            file_input = gr.File(file_count="multiple",label="Upload Image Sequence",file_types=["image"])
            
            anchor_dropdown = gr.Dropdown(choices=[], label="Anchor Image Index", interactive=True)
            
            threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, label="Matching Threshold")
            
            run_btn = gr.Button("🚀 Match & Visualize", variant="primary")
            status_output = gr.Textbox(label="Status")

        with gr.Column(scale=1):
            
            status_output = gr.Textbox(
                    label = "System Status",
                    interactive = False
            )

    file_input.change(
        fn = update_dropdown,
        inputs = file_input,
        outputs = anchor_dropdown
    )

    run_btn.click(
        fn = run_pipeline,
        inputs = [file_input, anchor_dropdown, threshold_slider],
        outputs = [status_output]
    )



if __name__ == "__main__":

    demo.launch(show_error=True)