conda create -n trf python=3.10
conda config --add channels conda-forge
conda install spacy
conda install -c huggingface transformers
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm

pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name=trf
pip install pandas
pip install nltk
