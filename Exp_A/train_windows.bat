@echo off
REM Disable libuv for PyTorch on Windows to avoid distributed training issues
set USE_LIBUV=0

REM Run the training script
python main.py --base configs/train/mlsp_embedding_stage1.yaml -t --gpus 1

pause
