-- creates a materialized view labels which has information regarding patient/hospital identifier

-- delete the materialized view labels if it already exists

drop materialized view if exists public.ld_labels cascade;
create materialized view public.ld_labels as
  select p.uniquepid, p.patienthealthsystemstayid, apr.patientunitstayid, p.unitvisitnumber, p.unitdischargelocation,
    p.unitdischargeoffset, p.unitdischargestatus, apr.predictedhospitalmortality, apr.actualhospitalmortality,
    apr.predictediculos, apr.actualiculos
    from eicu_crd.patient as p
    inner join eicu_crd.apachepatientresult as apr
      on p.patientunitstayid = apr.patientunitstayid
    where apr.apacheversion = 'IVa'  -- most recent apache prediction model
      and apr.actualiculos > (5/24)  -- exclude anyone who doesn't have at least 5 hours of data
      and nullif(replace(p.age, '> 89', '89'), '')::int > 17;  -- only include adults