mkdir data
mkdir exp_data
mkdir pretrained_models
conda env create -f environment.yml
python setup_bop.py install

unzip pretrained_models.zip -d pretrained_models/
wget https://huggingface.co/hamacojr/CAT-Seg/resolve/main/model_final_large.pth
mv model_final_large.pth  pretrained_models/catseg.pth