ROOT=data/shapenet6d
mkdir $ROOT

unzip oryon_data.zip "datasets/shapenet6d/**" -d "${ROOT}/"
mv ${ROOT}/datasets/shapenet6d/*.json ${ROOT}/
mv ${ROOT}/datasets/shapenet6d/*.csv ${ROOT}/
mv ${ROOT}/datasets/shapenet6d/*.pkl ${ROOT}/
mv "${ROOT}/datasets/nocs/fixed_split" ${ROOT}/
rm -r "${ROOT}/datasets"