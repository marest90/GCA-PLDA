import numpy as np
import csv
import sys


#########################################################
####### Este script toma la lista durations de voxceleb dev y arma un metadata como el de train



###############Abro el archivo con los nombres del test
test_direccion = '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/data/eval/voxceleb2_4ch_16sec/durations'

test_row = []
with open(test_direccion, 'r') as file:
    reader = csv.reader(file,delimiter=' ')
    for row in reader:
        test_row.append(row)
        
#crea un metadata
test_id = []
duration = []

for row in test_row:
    test_id.append(row[0])
    duration.append(row[1])

print(test_id)
print(duration)

#'id03041-7LBbfdTR2JY_chunk00_000000_000846'


#Escribe el archivo con los logids del test
with open('metadata.txt', 'w') as f:
    for i in range(len(test_id)):
        f.write(test_id[i] + ' ' + test_id[i][:7] + ' ' + test_id[i][:19] + ' ' + 'VOX' + ' ' + duration[i])
        f.write('\n')

