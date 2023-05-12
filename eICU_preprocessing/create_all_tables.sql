-- creates all the tables and produces csv files
-- takes a while to run (about an hour)

-- change the paths to those in your local computer using find and replace for '/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/'.

\i eICU_preprocessing/labels.sql
\i eICU_preprocessing/diagnoses.sql
\i eICU_preprocessing/flat_features.sql
\i eICU_preprocessing/timeseries.sql

-- we need to make sure that we have at least some form of time series for every patient in diagnoses, flat and labels
drop materialized view if exists ld_timeseries_patients cascade;
create materialized view public.ld_timeseries_patients as
  with repeats as (
    select distinct patientunitstayid
      from public.ld_timeserieslab
    union
    select distinct patientunitstayid
      from public.ld_timeseriesresp
    union
    select distinct patientunitstayid
      from public.ld_timeseriesnurse
    union
    select distinct patientunitstayid
      from public.ld_timeseriesperiodic
    union
    select distinct patientunitstayid
      from public.ld_timeseriesaperiodic)
  select distinct patientunitstayid
    from repeats;

\copy (select * from public.ld_labels as l where l.patientunitstayid in (select * from public.ld_timeseries_patients)) to '/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/labels.csv' with csv header
\copy (select * from public.ld_diagnoses as d where d.patientunitstayid in (select * from public.ld_timeseries_patients)) to '/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/diagnoses.csv' with csv header
\copy (select * from public.ld_flat as f where f.patientunitstayid in (select * from public.ld_timeseries_patients)) to '/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/flat_features.csv' with csv header
\copy (select * from public.ld_timeserieslab) to '/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/timeserieslab.csv' with csv header
\copy (select * from public.ld_timeseriesresp) to '/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/timeseriesresp.csv' with csv header
\copy (select * from public.ld_timeseriesnurse) to '/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/timeseriesnurse.csv' with csv header
\copy (select * from public.ld_timeseriesperiodic) to '/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/timeseriesperiodic.csv' with csv header
\copy (select * from public.ld_timeseriesaperiodic) to '/home/jovyan/storage/LoS-prediction/eICU_preprocessing/eICU_data/timeseriesaperiodic.csv' with csv header