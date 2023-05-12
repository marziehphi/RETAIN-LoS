-- MUST BE RUN AFTER labels.sql and select information of past, current, and admission diagnosis.

-- delete the materialized view diagnoses if it already exists

drop materialized view if exists public.ld_diagnoses cascade;
create materialized view public.ld_diagnoses as
  -- for current diagnoses:
  select d.patientunitstayid, d.diagnosisstring
    from eicu_crd.diagnosis as d
    inner join public.ld_labels as l on l.patientunitstayid = d.patientunitstayid
    inner join eicu_crd.patient as p on p.patientunitstayid = d.patientunitstayid
    where d.diagnosisoffset < 60*5 -- corresponds to 5 hours into the stay
  union
  -- for past medical history:
  select ph.patientunitstayid, ph.pasthistorypath as diagnosisstring
    from eicu_crd.pasthistory as ph
    inner join public.ld_labels as l on l.patientunitstayid = ph.patientunitstayid
    inner join eicu_crd.patient as p on p.patientunitstayid = ph.patientunitstayid
    where ph.pasthistoryoffset < 60*5  -- corresponds to 5 hours into the stay
  union
  -- for admission diagnoses:
  select ad.patientunitstayid, ad.admitdxpath as diagnosisstring
    from eicu_crd.admissiondx as ad
    inner join public.ld_labels as l on l.patientunitstayid = ad.patientunitstayid
    inner join eicu_crd.patient as p on p.patientunitstayid = ad.patientunitstayid
    where ad.admitdxenteredoffset < 60*5;  -- corresponds to 5 hours into the stay