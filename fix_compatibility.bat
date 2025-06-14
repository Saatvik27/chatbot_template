@echo off
echo Fixing PyTorch/Streamlit compatibility issues...
echo.

echo Current Python version:
python --version
echo.

echo Uninstalling potentially problematic packages...
pip uninstall -y torch torchvision torchaudio streamlit sentence-transformers numpy

echo.
echo Installing compatible versions...
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install streamlit==1.39.0
pip install sentence-transformers==2.7.0
pip install numpy==1.26.4

echo.
echo Installing remaining requirements...
pip install -r requirements.txt

echo.
echo Installation complete! Testing imports...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"
python -c "import sentence_transformers; print('Sentence transformers: OK')"

echo.
echo Fix complete! Try running your app now.
pause
