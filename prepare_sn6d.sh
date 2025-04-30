ROOT=data/shapenet6d
mkdir $ROOT
# unzip shapenet data
unzip shapenet6d.zip -d "${ROOT}"
mv ${ROOT}/scenes ${ROOT}/raw_data
mkdir ${ROOT}/raw_data/models

# rename and remove non-necessary files
mv ${ROOT}/raw_data/instance_segmap ${ROOT}/raw_data/mask
rm ${ROOT}/raw_data/*.list
rm ${ROOT}/raw_data/*.pkl

# unzip ShapeNetSem models
unzip -j ShapeNetSem.zip ShapeNetSem-backup/models-OBJ/models/* -d ${ROOT}/raw_data/models/ 

# unzip our annotations
unzip oryon_data.zip .
mv datasets/shapenet6d/* ${ROOT}/
rm -r "datasets"
rm -r "${ROOT}/shapenet6d/templates"