# Satellite Image Super-Resolution

This project uses a Restormer-style PyTorch model to upscale an image by 4x.

## Run the demo

Install the dependencies into the same Python environment that will run
Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, if multiple Python installations are available, use the Python
executable where PyTorch is installed. For example:

```powershell
C:\Python314\python.exe -m pip install -r requirements.txt
C:\Python314\python.exe -m streamlit run app.py
```

The app is then available at `http://localhost:8501`.

The app loads `restormer_weights.pth` when it is present. If the file is
missing, the demo still starts but the model is untrained.

## Training data

Put real satellite images in `data/train/` and run:

```powershell
python train.py
```

The UC Merced Land Use dataset is compatible with this workflow: it contains
real RGB aerial images in class subfolders. The training script creates
4x-super-resolution pairs automatically by downsampling each image to 1/4 of
its width and height. The dataset audit blocks predominantly synthetic or
random-looking training folders, but does not reject natural satellite imagery
just because some images have high pixel variation.

For CPU-only training, the script uses a deterministic balanced subset of up to
17 images per class (357 images for the included 21 classes) so a 30-epoch
fine-tune can finish in roughly one hour. To train on the complete dataset,
remove `max_images_per_class=17` from `train.py`; GPU training is recommended
for that configuration.

The current fine-tuned weights are saved to `restormer_weights.pth` after each
epoch and loaded automatically by the demo.

The training dataset and downloaded archives are intentionally excluded from
version control. Add them locally under `data/train/` when training.

## Troubleshooting

If Streamlit reports `ModuleNotFoundError: No module named 'torch'`, it is
running with a different Python installation than the one used for training.
Check the active interpreter:

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"
```

Run Streamlit with the same interpreter that successfully imports `torch`, for
example:

```powershell
C:\Python314\python.exe -m streamlit run app.py
```

If `streamlit run app.py` uses Anaconda or another Python installation, install
the requirements and launch Streamlit from that exact interpreter instead:

```powershell
C:\Users\asus\anaconda3\python.exe -m pip install -r requirements.txt
C:\Users\asus\anaconda3\python.exe -m streamlit run app.py
```
