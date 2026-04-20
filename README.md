<div align="left">

# Feature Matching Viewer

A simple tool to match features between images using the SuperGlue model, with real-time visualization of correspondences in Rerun. It uses Gradio for an interactive UI to manage image sequences and analyze matching consistency against a reference anchor frame.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Rerun](https://img.shields.io/badge/Rerun-0.31+-ff69b4.svg)](https://rerun.io/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://www.gradio.app/)

<br />
![Feature Matching Demo](assets/media/demo.gif)

<br />

</div>


## Install and Run 

### 1. Clone the Repository

Clone the repo with --recursive flag

```bash
git clone --recursive https://github.com/itsmeaboud/feature-matching-viewer.git
cd Feature-Matching-Viewer
```

### 2. Set Up Environment

Create and activate a Conda environment with Python 3.10:

```bash
conda create -n matching_env python=3.10 -y
conda activate matching_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the Gradio interface from the project root:

```bash
python app.py
```


## Project Structure

| File / Folder | Description |
|---|---|
| `app.py` | Entry point containing the Gradio UI |
| `src/` | Logic for the matching pipeline and Rerun logging |
| `external/superglue/` | The SuperGlue submodule |

## Acknowledgements

[SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork.git)
```bibtex
@inproceedings{sarlin20superglue,
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  booktitle = {CVPR},
  year      = {2020},
  url       = {https://arxiv.org/abs/1911.11763}
}
```