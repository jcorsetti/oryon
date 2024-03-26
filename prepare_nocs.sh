ROOT=data/nocs
mkdir $ROOT
mkdir $ROOT/split

unzip obj_models.zip "obj_models/real_test/**" -d "${ROOT}/"
unzip gts.zip "gts/real_test/**" -d "${ROOT}/"
unzip real_test.zip "real_test/**" -d "${ROOT}/split/"

python scripts/data/make_nocs_obj_normal.py ${ROOT}
python scripts/data/nocs_bop_models.py ${ROOT}
python scripts/data/make_nocs_data.py ${ROOT}

unzip oryon_data.zip "datasets/nocs/**" -d "${ROOT}/"
mv ${ROOT}/datasets/nocs/*.json ${ROOT}/
mv "${ROOT}/datasets/nocs/fixed_split" ${ROOT}/
rm -r "${ROOT}/datasets"