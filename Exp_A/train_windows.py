import os
# Set environment variable before importing PyTorch
os.environ['USE_LIBUV'] = '0'

# Now run the main script
import subprocess
import sys

subprocess.run([sys.executable, 'main.py', '--base', 'configs/train/mlsp_embedding_stage1.yaml', '-t', '--gpus', '1'])
