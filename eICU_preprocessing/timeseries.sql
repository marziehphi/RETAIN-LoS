-- MUST BE RUN AFTER labels.sql
-- creates a materialized view of: commonlabs, timeserieslab, commonresp, timeseriesresp, timeseriesperiodic, timeseriesaperiodic

-- extract the most common lab tests and the corresponding counts of how many patients have values for those labs

drop materialized view if exists public.ld_commonlabs cascade;
create materialized view public.ld_commonlabs as
  select labsstay.labname, count(distinct la.patientunitstayid) as count
    -- we choose between -1440 and discharge offset in order to consider the labs recorded before the unit time starts.
    from (
      select l.labname, l.patientunitstayid
        from lab as l
        inner join eicu_crd.patient as p
          on p.patientunitstayid = l.patientunitstayid
        where labresultoffset between -1440 and p.unitdischargeoffset
      ) as labsstay
    inner join public.ld_labels as la
      on la.patientunitstayid = labsstay.patientunitstayid
    group by labsstay.labname
    -- only keep data that is present at some point for at least 25% of the patients, this gives us 47 lab features
    having count(distinct la.patientunitstayid) > (select count(distinct patientunitstayid) from 'public'.'ld_labels')*0.25
    order by count desc;

-- get the time series features from the most common lab tests (46 of these)
drop materialized view if exists public.ld_timeserieslab cascade;
create materialized view public.ld_timeserieslab as
  select l.patientunitstayid, l.labresultoffset, l.labname, l.labresult
    from eicu_crd.lab as l
    inner join eicu_crd.patient as p
      on p.patientunitstayid = l.patientunitstayid
    inner join public.ld_commonlabs as cl
      on cl.labname = l.labname 
    inner join public.ld_labels as la
      on la.patientunitstayid = l.patientunitstayid  
    where l.labresultoffset between -1440 and p.unitdischargeoffset;

-- extract the most common respiratory chart entries 
drop materialized view if exists public.ld_commonresp cascade;
create materialized view public.ld_commonresp as
  select respstay.respchartvaluelabel, count(distinct la.patientunitstayid) as count
    from (
      select r.respchartvaluelabel, r.patientunitstayid
        from eicu_crd.respiratorycharting as r
        inner join eicu_crd.patient as p
          on p.patientunitstayid = r.patientunitstayid
        where respchartoffset between -1440 and p.unitdischargeoffset
      ) as respstay
    inner join public.ld_labels as la
      on la.patientunitstayid = respstay.patientunitstayid
    group by respstay.respchartvaluelabel
    -- only keep data that is present at some point for at least 12.5% of the patients
    having count(distinct la.patientunitstayid) > (select count(distinct patientunitstayid) from 'public'.'ld_labels')*0.125
    order by count desc;

-- get the time series features from the most common respiratory chart entries (14 of these)
drop materialized view if exists public.ld_timeseriesresp cascade;
create materialized view public.ld_timeseriesresp as
  select r.patientunitstayid, r.respchartoffset, r.respchartvaluelabel, r.respchartvalue
    from eicu_crd.respiratorycharting as r
    inner join public.ld_commonresp as cr
      on cr.respchartvaluelabel = r.respchartvaluelabel 
    inner join public.ld_labels as la
      on la.patientunitstayid = r.patientunitstayid 
    inner join eicu_crd.patient as p
      on p.patientunitstayid = r.patientunitstayid
    where r.respchartoffset between -1440 and p.unitdischargeoffset;

-- extract the most common nurse chart entries 
drop materialized view if exists public.ld_commonnurse cascade;
create materialized view public.ld_commonnurse as
  select nursestay.nursingchartcelltypevallabel, count(distinct la.patientunitstayid) as count
    from (
      select n.nursingchartcelltypevallabel, n.patientunitstayid
        from eicu_crd.nursecharting as n
        inner join eicu_crd.patient as p
          on p.patientunitstayid = n.patientunitstayid
        where nursingchartoffset between -1440 and p.unitdischargeoffset
      ) as nursestay
    inner join public.ld_labels as la
      on la.patientunitstayid = nursestay.patientunitstayid
    group by nursestay.nursingchartcelltypevallabel
    -- only keep data that is present at some point for at least 25% of the patients
    having count(distinct la.patientunitstayid) > (select count(distinct patientunitstayid) from 'public'.'ld_labels')*0.125
    order by count desc;

-- get the time series features from the most common nursing chart entries
drop materialized view if exists public.ld_timeseriesnurse cascade;
create materialized view public.ld_timeseriesnurse as
  select n.patientunitstayid, n.nursingchartoffset, n.nursingchartcelltypevallabel, n.nursingchartvalue
    from eicu_crd.nursecharting as n
    inner join public.ld_commonnurse as cn
      on cn.nursingchartcelltypevallabel = n.nursingchartcelltypevallabel 
    inner join public.ld_labels as la
      on la.patientunitstayid = n.patientunitstayid  
    inner join eicu_crd.patient as p
      on p.patientunitstayid = n.patientunitstayid
    where n.nursingchartoffset between -1440 and p.unitdischargeoffset;

-- get the periodic (regularly sampled) time series data
drop materialized view if exists public.ld_timeseriesperiodic cascade;
create materialized view public.ld_timeseriesperiodic as
  select vp.patientunitstayid, vp.observationoffset, vp.temperature, vp.sao2, vp.heartrate, vp.respiration, vp.cvp,
    vp.systemicsystolic, vp.systemicdiastolic, vp.systemicmean, vp.st1, vp.st2, vp.st3
    from eicu_crd.vitalperiodic as vp
    inner join public.ld_labels as la
      on la.patientunitstayid = vp.patientunitstayid
    inner join eicu_crd.patient as p
      on p.patientunitstayid = vp.patientunitstayid
    where vp.observationoffset between -1440 and p.unitdischargeoffset
    order by vp.patientunitstayid, vp.observationoffset;

-- get the aperiodic (irregularly sampled) time series data
drop materialized view if exists public.ld_timeseriesaperiodic cascade;
create materialized view public.ld_timeseriesaperiodic as
    select va.patientunitstayid, va.observationoffset, va.noninvasivesystolic, va.noninvasivediastolic, va.noninvasivemean
    from eicu_crd.vitalaperiodic as va
    inner join public.ld_labels as la
      on la.patientunitstayid = va.patientunitstayid
    inner join eicu_crd.patient as p
      on p.patientunitstayid = va.patientunitstayid
    where va.observationoffset between -1440 and p.unitdischargeoffset
    order by va.patientunitstayid, va.observationoffset;
