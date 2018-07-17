set dbname = ;
set tablename1  = dwa_v_m_cus_2g_rns_wide
set tablename2  = dwa_v_m_cus_3g_rns_wide
set tablename3  = dwa_v_m_cus_cb_rns_wide
set tablename4  = dwa_m_ia_basic_label
set tablename5  = cusc_ddh_gas_final_end
set tablename6  = dwa_m_ia_basic_user_app
set tablename7  = dwa_d_ia_basic_user_web
set tablename8  = cusc_ddh_house_price_distance201804
set tablename9  = gps_distance_misidn201804
set path_routh = /tmp/zwkj2
set month_id='201805';
set prov_id='';

---1、用户信息表，包含年龄分箱、性别、月份、省份id
drop table ml_user_age_sex_area;
create table ml_user_age_sex_area as
select distinct device_number,cust_sex,years_old,month_id,prov_id from
(select device_number,cust_sex,(case
when cert_age > 0 and cert_age <= 20 then '0-20old'
when cert_age > 20 and cert_age <= 30 then '20-30old'
when cert_age > 30 and cert_age <= 40 then '30-40old'
when cert_age > 40 and cert_age <= 50 then '40-50old'
when cert_age > 50 and cert_age <= 60 then '50-60old'
when cert_age > 60 and cert_age <= 70 then '60-70old'
else 'unknow'
end)years_old,area_id,month_id,prov_id from dbname.tablename1 where month_id = ${hiveconf:month_id} and prov_id = ${hiveconf:prov_id})
union all
select device_number,cust_sex,(case
when cert_age > 0 and cert_age <= 20 then '0-20old'
when cert_age > 20 and cert_age <= 30 then '20-30old'
when cert_age > 30 and cert_age <= 40 then '30-40old'
when cert_age > 40 and cert_age <= 50 then '40-50old'
when cert_age > 50 and cert_age <= 60 then '50-60old'
when cert_age > 60 and cert_age <= 70 then '60-70old'
else 'unknow'
end)years_old,area_id,month_id,prov_id from dbname.tablename2 where month_id = ${hiveconf:month_id} and prov_id = ${hiveconf:prov_id})
union all
select device_number,cust_sex,(case
when cert_age > 0 and cert_age <= 20 then '0-20old'
when cert_age > 20 and cert_age <= 30 then '20-30old'
when cert_age > 30 and cert_age <= 40 then '30-40old'
when cert_age > 40 and cert_age <= 50 then '40-50old'
when cert_age > 50 and cert_age <= 60 then '50-60old'
when cert_age > 60 and cert_age <= 70 then '60-70old'
else 'unknow'
end)years_old,area_id,month_id,prov_id from dbname.tablename3 where month_id = ${hiveconf:month_id} and prov_id = ${hiveconf:prov_id})t
where cust_sex in ('2','1') and years_old != 'unknow';

---数据清洗
drop table ml_user_age_sex_area_detail;
create table ml_user_age_sex_area_detail as
select t1.device_number,t2.cust_sex,t2.years_old,t2.area_id,t2.month_id,t2.prov_id from 
(select device_number from (select device_number,count(*) as co from ml_user_age_sex_area group by device_number) t where co < 2) t1,
ml_user_age_sex_area t2
where t1.device_number = t2.device_number


---2、处于三个生命阶段中的任意一个都算作1，其他为0
drop table life_stage_is_no;
create table life_stage_is_no as 
select distinct device_number,(
case 
when label_code in ('H004008') and visit_cnt > 50 then 1
when label_code in ('H002002') and visit_cnt > 50 then 1
when label_code in ('H005009') and visit_cnt > 50 then 1
else 0
end
) lift_stage_yn,month_id,prov_id
from dbname.tablename4 where month_id= ${hiveconf:month_id} and prov_id= ${hiveconf:prov_id})


---3、车辆保有者，访问加油站2次及两次以上，10次及10次以下的用户 +使用汽车相关app1次以上（不包含1次）的用户 

drop table ml_car_owner;
create table ml_car_owner as
select distinct device_number,1 car_owner from 
(select misidn as device_number from dbname.tablename5 where 1 < vis_gas_cnt < 11
union all
select distinct device_number from dbname.tablename6 where prod_id in ('C225602' ,'C223569' ,'C227439' ,'C227385' ,'C231058' ,'C227429' ,'C223567' ,'C27696' ,'C222419' ,'C48582' ,'C229606' ,'C231442' ,'C222477' ,'C48441' ,'C224605' ,'C223616' ,'C65229' ,'C21580' ,'C222479' ,'C225422' ,'C32036' ,'C227670' ,'C65311' ,'C227666' ,'C29512' ,'C66077' ,'C29819' ,'C84499' ,'C22175' ,'C224988' ,'C231766' ,'C228238' ,'C225685' ,'C226982' ,'C231601' ,'C231054' ,'C231247' ,'C227156' ,'C229965' ,'C66458' ,'C227858' ,'C228165' ,'C231258' ,'C231250' ,'C226428' ,'C231746' ,'C223849' ,'C231251' ,'C231253' ,'C231252' ,'C226991' ,'C223224' ,'C231254' ,'C231248' ,'C230707' ,'C224629' ,'C231249' ,'C224777' ,'C226626' ,'C76354' ,'C27627' ,'C231328' ,'C227560' ,'C65381' ,'C227856' ,'C222465' ,'C222571' ,'C32194' ,'C230361' ,'C66769' ,'C226310' ,'C226836' ,'C66767' ,'C84363' ,'C224460' ,'C223935' ,'C223228' ,'C20430' ,'C225694' ,'C84362' ,'C224781' ,'C224855' ,'C231594' ,'C223594' ,'C230185' ,'C227668' ,'C227116' ,'C227594' ,'C230806' ,'C225183' ,'C227637' ,'C224506' ,'C32504' ,'C227655' ,'C226239' ,'C227681' ,'C225179' ,'C228537' ,'C230805' ,'C226765' ,'C227117' ,'C230255' ,'C227705' ,'C227980' ,'C227645' ,'C231063' ,'C233120' ,'C230515' ,'C227673' ,'C227716' ,'C227688' ,'C231052') and visit_cnt > 1 and month_id= ${hiveconf:month_id} and prov_id= ${hiveconf:prov_id}) t;


---4、上网行为表，包含导航、滴滴、共享汽车、共享单着访问次数与访问时长
drop table ml_internet_app;
create table ml_internet_app as
select t.device_number,t1.daohang_cnt,t1.daohang_dura,t2.didi_cnt,t2.didi_dura,t3.gongxiangdanche_cnt,t3.gongxiangdanche_dura,t4.gongxiangqiche_cnt,t4.gongxiangqiche_dura,from 
(select distinct device_number from dbname.tablename6 where month_id = ${hiveconf:month_id}) t
left join 
(select device_number, sum(visit_cnt) as daohang_cnt,sum(visit_dura) as daohang_dura from dbname.tablename6 where month_id =  ${hiveconf:month_id} and prod_name in('高德地图','腾讯地图','百度地图') group by device_number) t1
on t.device_number = t1.device_number
left join 
(select device_number, sum(visit_cnt) as didi_cnt,sum(visit_dura) as didi_dura from dbname.tablename6 where month_id =  ${hiveconf:month_id} and prod_name in('滴滴出行') group by device_number) t2
on t.device_number = t2.device_number
left join
(select device_number, sum(visit_cnt) as gongxiangdanche_cnt,sum(visit_dura) as gongxiangdanche_dura from dbname.tablename6 where month_id =  ${hiveconf:month_id} and prod_name in('ofo','摩拜单车') group by device_number) t3
on t.device_number = t3.device_number
left join
(select device_number, sum(visit_cnt) as gongxiangqiche_cnt,sum(visit_dura) as gongxiangqiche_dura from dbname.tablename6 where month_id =  ${hiveconf:month_id} and prod_id in('C223294') group by device_number) t4
on t.device_number = t4.device_number


---5、房价表，测试平台表名：cusc_ddh_house_price_distance，字段名：avgprice。生产平台中表名未知


---6、移动距离表，测试平台表名：dwa_s_gps_matser20，字段名：t7.foot_distance,t7.bike_distance,t7.car_distance。生产平台中表名未知


---7、访问加油站次数  表名：dbname.tablename5  字段名：vis_cnt


---8、家到公司距离表，测试平台表名：cusc_ddh_house_price_distance，字段名：distance。生产平台中表名未知


---9、关联所有表制成宽表

drop table ml_master;
create table ml_master as
select distinct t1.device_number,
t2.cust_sex,t2.years_old,
t3.lift_stage_yn,
t4.car_owner,
t5.area_price,
t5.distance,
t6.daohang_cnt,t6.daohang_dura,t6.didi_cnt,t6.didi_dura,t6.gongxiangdanche_cnt,t6.gongxiangdanche_dura,t6.gongxiangqiche_cnt,t6.gongxiangqiche_dura,
t7.foot_distance,t7.bike_distance,t7.car_distance,
t8.vis_cnt as gas_vis_cnt
from
(select * from dwa_d_ia_basic_user_app where month_id =  ${hiveconf:month_id}) t1
left join
ml_user_age_sex_area_detail t2
on t1.device_number = t2.device_number
left join
life_stage_is_no t3
on t1.device_number = t3.device_number
left join 
ml_car_owner t4
on t1.device_number = t4.device_number
left join 
dbname.tablename8 t5
on t1.device_number = t5.msisdn
left join
ml_internet_app t6
on t1.device_number = t6.device_number
left join
dbname.tablename9 t7
on t1.device_number = t7.misidn
left join
dbname.tablename5 t8
on t1.device_number = t8.misidn;



hive -e "insert overwrite local directory '${hiveconf:path}' row format delimited fields terminated by '\t' select * FROM ml_master;"
cat ${hiveconf:path}/0* >> cust_profile_info_0514.csv








