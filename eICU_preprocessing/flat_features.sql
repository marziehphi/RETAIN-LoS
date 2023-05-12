-- MUST BE RUN AFTER labels.sql
-- creates a materialized view flat which selects all static feature from patientss:

-- delete the materialized view flat if it already exists

drop materialized view if exists public.ld_flat cascade;
create materialized view public.ld_flat as
  select distinct la.patientunitstayid, p.gender, p.age, p.ethnicity, p.admissionheight, p.admissionweight,
    p.apacheadmissiondx, extract(hour from to_timestamp(p.unitadmittime24,'HH24:MI:SS')) as hour, p.unittype,
    p.unitadmitsource, p.unitvisitnumber, p.unitstaytype, apr.physicianspeciality, aps.intubated, aps.vent,
    aps.dialysis, aps.eyes, aps.motor, aps.verbal, aps.meds, apv.bedcount, h.numbedscategory, h.teachingstatus,
    h.region
    from eicu_crd.patient as p
    inner join eicu_crd.apacheapsvar as aps on aps.patientunitstayid = p.patientunitstayid
    inner join eicu_crd.apachepatientresult as apr on apr.patientunitstayid = p.patientunitstayid
    inner join eicu_crd.apachepredvar as apv on apv.patientunitstayid = p.patientunitstayid
    inner join eicu_crd.hospital as h on h.hospitalid = p.hospitalid
    inner join public.ld_labels as la on la.patientunitstayid = p.patientunitstayid;
