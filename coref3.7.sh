python3.7 -m venv coref3.7
source  coref3.7/bin/activate
cd coref3.7
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .


python -m spacy download en_core_web_sm #trf incompatible with spacy version\
pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name=coref
pip install pandas
pip install nltk
