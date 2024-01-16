#!/usr/bin/env bash

set -e

#----------------------------------------------------------------划分好的数据集-----------------------------------------------------
crop=wheat_rice
dset=data_wheat_rice
label_name=Vcmax
python main_trandition.py name=${crop}_OnedCNN_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=OnedCNN
python main_trandition.py name=${crop}_IndiceCNN_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndiceCNN


label_name=Jmax
python main_trandition.py name=${crop}_OnedCNN_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=OnedCNN
python main_trandition.py name=${crop}_IndiceCNN_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndiceCNN

label_name=Vcmax
python main.py name=${crop}_IndexfindNet_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndexfindNet
label_name=Jmax
python main.py name=${crop}_IndexfindNet_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndexfindNet



label_name=Vcmax
for compress_num in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2; do
  echo $compress_num
  python main.py name=${crop}_IndexfindNet_compress${compress_num}_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=Indexfindnet compress_num=$compress_num
done

label_name=Jmax
for compress_num in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2; do
  echo $compress_num
  python main.py name=${crop}_IndexfindNet_compress${compress_num}_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=Indexfindnet compress_num=$compress_num
done

label_name=Vcmax
for resample in  60 120 180 204 240 300 400 500 600; do
  echo $resample
  python main.py name=${crop}_IndexfindNet_resample${resample}_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=Indexfindnet resample=$resample
done

label_name=Jmax
for resample in 60 120 180 204 240 300 400 500 600; do
  echo $resample
  python main.py name=${crop}_IndexfindNet_resample${resample}_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=Indexfindnet resample=$resample
done


#----------------------------------------------------------------自动划分数据集-----------------------------------------------------
crop=wheat_rice
python data_proceed/get_testdata.py 0 data/alldata/wheat_rice_ref_indice_pho_params.csv
python data_proceed/split-train-val-data.py 0
dset=data

label_name=Vcmax
python main_trandition.py name=${crop}_OnedCNN_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=OnedCNN
python main_trandition.py name=${crop}_IndiceCNN_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndiceCNN


label_name=Jmax
python main_trandition.py name=${crop}_OnedCNN_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=OnedCNN
python main_trandition.py name=${crop}_IndiceCNN_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndiceCNN

label_name=Vcmax
python main.py name=${crop}_IndexfindNet_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndexfindNet
label_name=Jmax
python main.py name=${crop}_IndexfindNet_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=IndexfindNet

label_name=Vcmax
for compress_num in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2; do
  echo $compress_num
  python main.py name=${crop}_IndexfindNet_compress${compress_num}_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=Indexfindnet compress_num=$compress_num
done

label_name=Jmax
for compress_num in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2; do
  echo $compress_num
  python main.py name=${crop}_IndexfindNet_compress${compress_num}_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=Indexfindnet compress_num=$compress_num
done

label_name=Vcmax
for resample in  60 120 180 204 240 300 400 500 600; do
  echo $resample
  python main.py name=${crop}_IndexfindNet_resample${resample}_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=Indexfindnet resample=$resample
done

label_name=Jmax
for resample in 60 120 180 204 240 300 400 500 600; do
  echo $resample
  python main.py name=${crop}_IndexfindNet_resample${resample}_${label_name} dset=${dset} label_name=${label_name} task=hyper data_clean=0 embedding=0 model=Indexfindnet resample=$resample
done

