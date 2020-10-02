The requirements.txt file includes the dependencies.

Extract the rar file.
The model is saved as 'classifier.pkl' within the folder

1. To evaluate the pretrained model run evaluate.py. Keep the "NaturalBlurSet.xlsx" and "DigitalBlurSet.xlsx" within 
the 'submission' folder.

2. To generate the features copy 'CERTH_ImageBlurDataset' folder into the 'submission' folder and run generate.py
This takes about 30 minutes depending on the speed of the system. I have included the genrated .csv file in the
'submission' folder.

3. To train a model run train.py and the model is saved as 'classifier.pkl'.