eICU pre-processing
==================================

1) To run the sql files you must have the eICU database set up: https://physionet.org/content/eicu-crd/2.0/. 

2) Follow the instructions: https://eicu-crd.mit.edu/tutorials/install_eicu_locally/ to ensure the correct connection configuration. 

3) Replace the eICU_path in `paths.json` to a convenient location in your computer, and do the same for `eICU_preprocessing/create_all_tables.sql` using find and replace for 
`'/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/'`. Leave the extra '/' at the end.

4) In your terminal, navigate to the project directory, then type the following commands:

    ```
    psql 'dbname=eicu user=eicu options=--search_path=eicu'
    ```
    
    Inside the psql console:
    
    ```
    \i eICU_preprocessing/create_all_tables.sql
    ```
    
    This step might take a couple of hours.
    
    To quit the psql console:
    
    ```
    \q
    ```
    
5) Then run the pre-processing scripts in your terminal. This will need to run overnight:

    ```
    python3 -m eICU_preprocessing.run_all_preprocessing
    ```
  
 It is focusing on more complex criteria and more possible information extraction from original tables in eICU database. resulting in 118,535 unique patients and 146,671 ICU stays.


6) Then run the pre-processing scripts in your terminal to overcome the shortcomings of its complexity (computational time). It creates a compressed csv file in the database directory called _eicu_features.csv.

```
python3 -m eICU_preprocessing._preprocess_eICU --path path-to-database-directory 
```

7) Then run the pre-processing scripts in your terminal to access relevant information using small amount of information as possible (patient, lab, and nurseCharting table). It creates a compressed csv file in the database directory called eicu_features.csv.

```
python3 -m eICU_preprocessing.preprocess_eICU --path path-to-database-directory 
```

It will create the following directory structure:

```bash
eICU_data
├── test
│   ├── diagnoses.csv
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── train
│   ├── diagnoses.csv
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── val
│   ├── diagnoses.csv
│   ├── flat.csv
│   ├── labels.csv
│   ├── stays.txt
│   └── timeseries.csv
├── diagnoses.csv
├── flat_features.csv
├── labels.csv
├── timeseriesaperiodic.csv
├── timeserieslab.csv
├── timeseriesnurse.csv
├── timeseriesperiodic.csv
├── timeseriesresp.csv
├── patient.csv
├── lab.csv
├── nurseCharting.csv
├── _eicu_features.csv
└── eicu_features.csv
```