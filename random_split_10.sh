#! /bin/bash

dataset=$1;

if [[ ${dataset} == "" ||  ! -d ${dataset} ]]
then
	echo "Where is the dataset?"
	exit 1
fi

mkdir -p tmp
clone=tmp/copy_dir;
cp -r ${dataset} ${clone};

new_folder=tmp/NWPU10;
if [ -d ${new_folder} ]
then
	rm -rf ${new_folder}
fi
mkdir ${new_folder};
mkdir ${new_folder}/train;
mkdir ${new_folder}/test;

for folder in $(ls ${clone}); do
	mkdir ${new_folder}/train/${folder};
	mkdir ${new_folder}/test/${folder};
	echo -n "Creating folder: " ${new_folder}/$folder;
	echo ""
	for file in $(ls ${clone}/${folder}/ | shuf -n 70); do
		mv ${clone}/${folder}/${file} ${new_folder}/train/${folder}/;
	done
	mv ${clone}/${folder}/* ${new_folder}/test/${folder}/;
done

rm -rf ${clone};

