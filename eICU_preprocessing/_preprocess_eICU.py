import os
import numpy as np
import pandas as pd
import argparse
import progressbar

parser = argparse.ArgumentParser(description='preprocessing version one')
parser.add_argument('--path', help='Path to data directory', required=True, type=str)

#train/test/valid
parser.add_argument('--path_train', help='Path to train', required=True, type=str)
parser.add_argument('--path_test', help='Path to test', required=True, type=str)
parser.add_argument('--path_valid', help='Path to valid', required=True, type=str)

args = parser.parse_args()

assert len(args.path) > 0, 'Empty path'

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('StayID')
            ]

logfile = open(os.path.join('summary', 'preprocess.log'), 'w')

# create patient final file

# read labels
label_test = pd.read_csv(os.path.join(args.path_test, 'labels.csv'))
label_train = pd.read_csv(os.path.join(args.path_train, 'labels.csv'))
label_valid = pd.read_csv(os.path.join(args.path_valid, 'labels.csv'))

# only choose relevant columns
sub_label_test = label_test[['patient', 'unitdischargeoffset', 'unitdischargestatus_Expired', 'unitdischargestatus_Alive']]
sub_label_train = label_train[['patient', 'unitdischargeoffset', 'unitdischargestatus_Expired', 'unitdischargestatus_Alive']]
sub_label_valid = label_valid[['patient', 'unitdischargeoffset', 'unitdischargestatus_Expired', 'unitdischargestatus_Alive']]

# read flats
flat_test = pd.read_csv(os.path.join(args.path_test, 'flat.csv'))
flat_train = pd.read_csv(os.path.join(args.path_train, 'flat.csv'))
flat_valid = pd.read_csv(os.path.join(args.path_valid, 'flat.csv'))

# only choose relevant columns
sub_flat_test = flat_test[['patient', 'gender', 'age', 'admissionheight', 'admissionweight', 'ethnicity_African American',
                           'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic', 'ethnicity_Native American',
                           'ethnicity_Other/Unknown', 'unitvisitnumber_1', 'unitvisitnumber_2',
                           'unitvisitnumber_3', 'unitvisitnumber_misc', 'unitstaytype_admit', 'unitstaytype_readmit',
                           'unitstaytype_transfer', '> 89', 'nullweight', 'nullheight']]

sub_flat_train = flat_train[['patient', 'gender', 'age', 'admissionheight', 'admissionweight', 'ethnicity_African American',
                           'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic', 'ethnicity_Native American',
                           'ethnicity_Other/Unknown', 'unitvisitnumber_1', 'unitvisitnumber_2',
                           'unitvisitnumber_3', 'unitvisitnumber_misc', 'unitstaytype_admit', 'unitstaytype_readmit',
                           'unitstaytype_transfer', '> 89', 'nullweight', 'nullheight']]

sub_flat_valid = flat_valid[['patient', 'gender', 'age', 'admissionheight', 'admissionweight', 'ethnicity_African American',
                           'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic', 'ethnicity_Native American',
                           'ethnicity_Other/Unknown', 'unitvisitnumber_1', 'unitvisitnumber_2',
                           'unitvisitnumber_3', 'unitvisitnumber_misc', 'unitstaytype_admit', 'unitstaytype_readmit',
                           'unitstaytype_transfer', '> 89', 'nullweight', 'nullheight']]


flat_frames = [sub_flat_test, sub_flat_train, sub_flat_valid]
flat_result = pd.concat(flat_frames)

label_frames = [sub_label_test, sub_label_train, sub_label_valid]
label_result = pd.concat(label_frames)

patients = flat_result.merge(label_result, on='patient')
logfile.write("patients has {} records\n".format(patients.shape[0]))

# age--> remove the age above 89 years old
patients_s1 = patients.loc[patients['> 89'] == 0]
logfile.write("patients has {} records after filtering by age\n".format(patients.shape[0]))

# only consider the alive patients
patients_s2 = patients_s1.loc[patients_s1['unitdischargestatus_Alive'] == 1]
logfile.write("patients has {} records after filtering by status\n".format(patients.shape[0]))

# only consider the one visit patients
patients_s3 = patients_s2.loc[patients_s2['unitvisitnumber_1'] == 1]
logfile.write("patients has {} records after filtering by number of stays\n".format(patients.shape[0]))

# Select stayids
stayids = patients['patient']
patients.to_csv(os.path.join(args.path, 'patient_features.csv'), index=False)
del patients

# Read diagnosis file
diag_test = pd.read_csv(os.path.join(args.path_test, 'diagnoses.csv'))
diag_train = pd.read_csv(os.path.join(args.path_train, 'diagnoses.csv'))
diag_valid = pd.read_csv(os.path.join(args.path_valid, 'diagnoses.csv'))

diag_frames = [diag_test, diag_train, diag_valid]
diag_result = pd.concat(diag_frames)

diag_s1 = diag_result[diag_result['patient'].isin(stayids)]
logfile.write("dignos has {} records after filtering by number of patients\n".format(patients.shape[0]))

diag_s1.to_csv(os.path.join(args.path, 'diag_features.csv'), index=False)
del diag_s1

# Read timeseries information
ts_test = pd.read_csv(os.path.join(args.path_test, 'timeseries.csv'))
ts_train = pd.read_csv(os.path.join(args.path_train, 'timeseries.csv'))
ts_valid = pd.read_csv(os.path.join(args.path_valid, 'timeseries.csv'))

lab_test = ts_test[['patient', 'time', 'glucose', 'bedside glucose', 'pH', 'FiO2',
                   'Invasive BP', 'Non-Invasive BP', 'noninvasivediastolic', 'noninvasivemean', 'noninvasivesystolic',
                   'Heart Rate', 'Respiratory Rate', 'Temperature']]

lab_train = ts_train[['patient', 'time', 'glucose', 'bedside glucose', 'pH', 'FiO2',
                   'Invasive BP', 'Non-Invasive BP', 'noninvasivediastolic', 'noninvasivemean', 'noninvasivesystolic',
                   'Heart Rate', 'Respiratory Rate', 'Temperature']]

lab_valid = ts_valid[['patient', 'time', 'glucose', 'bedside glucose', 'pH', 'FiO2',
                   'Invasive BP', 'Non-Invasive BP', 'noninvasivediastolic', 'noninvasivemean', 'noninvasivesystolic',
                   'Heart Rate', 'Respiratory Rate', 'Temperature']]


lab_frames = [lab_test, lab_train, lab_valid]
lab_result = pd.concat(lab_frames)

lab_s1 = lab_result[lab_result['patient'].isin(stayids)]
lab_s1 = lab_s1.reset_index()
lab_s1 = lab_s1.drop(columns=['index'], axis=1)

lab_s1.to_csv(os.path.join(args.path, 'lab_features.csv'), index=False)
logfile.write("lab has {} records after filtering by number of patients\n".format(patients.shape[0]))
del lab_s1

# Combining all features
final_s1 = patients_s3.merge(diag_s1, on='patient')
eicu_final = final_s1.merge(lab_s1, on='patient')
first_column = eicu_final.pop('time') #is the same as offset
eicu_final.insert(1, 'time', first_column)

# Compute RLOS
eicu_final['rlos'] = eicu_final['unitdischargeoffset']/1440 - eicu_final['time']/24

# Write features to CSV
eicu_final.to_csv(os.path.join(args.path, 'eicu_features.csv'), index=False)

logfile.write('Wrote all features to CSV\n')
logfile.close()
