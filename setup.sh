git clone https://github.com/google/BIG-bench.git
cd BIG-bench
python setup.py sdist
pip install -e .
pip install transformers==4.24.0
pip install accelerate==0.14.0
pip install tqdm==4.64.1
pip install datasets==2.7.1
pip install torch torchvision torchaudio
