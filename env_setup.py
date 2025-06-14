# Environment variables to fix PyTorch/Streamlit compatibility issues
import os

# Set environment variables before importing torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Disable torch distributed if not needed
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

# Set Streamlit to use less aggressive file watching
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
