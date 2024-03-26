ROOT=toyl
mkdir $ROOT
mkdir $ROOT/split

unzip tyol_models.zip "models/**" -d "${ROOT}/"
mv $ROOT/models $ROOT/models_bop 
unzip tyol_test_bop19.zip -d "${ROOT}/split/"

python scripts/data/fix_bop_masks.py ${ROOT}

unzip oryon_data.zip "datasets/toyl/**" -d "${ROOT}/"
mv ${ROOT}/datasets/toyl/*.json ${ROOT}/
mv "${ROOT}/datasets/toyl/fixed_split" ${ROOT}/
rm -r "${ROOT}/datasets"