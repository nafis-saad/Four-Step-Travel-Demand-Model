import pandas as pd
import numpy as np
import time
import os
import random
from urbansim.models import transition, relocation
from urbansim.developer import sqftproforma, developer
from urbansim.utils import misc, networks
import urbansim.sim.simulation as sim
import dataset, variables, utils, transcad
import pandana as pdna


            
@sim.model('travel_model')
def travel_model(year, travel_data, buildings, parcels, households, persons, jobs):
    if year in [2015, 2020, 2025, 2030, 2035, 2040]:
        datatable = 'TAZ Data Table'
        joinfield = 'ZoneID'
        
        input_dir = './/runs//'  ##Where TM expects input
        input_file = input_dir + 'tm_input.tab'
        output_dir = './/data//'  ##Where TM outputs
        output_file = 'tm_output.txt'
        
        def delete_dcc_file(dcc_file):
            if os.path.exists(dcc_file):
                os.remove(dcc_file)
                
        delete_dcc_file(os.path.splitext(input_file)[0] + '.dcc' )
        
        parcels = parcels.to_frame()#(['zone_id','parcel_sqft'])
        hh = households.to_frame()
        persons = persons.to_frame()
        jobs = jobs.to_frame()

        zonal_indicators = pd.DataFrame(index=np.unique(parcels.zone_id.values))
        zonal_indicators['AcresTotal'] = parcels.groupby('zone_id').parcel_sqft.sum()/43560.0
        zonal_indicators['Households'] = hh.groupby('zone_id').size()
        zonal_indicators['HHPop'] = hh.groupby('zone_id').persons.sum()
        zonal_indicators['EmpPrinc'] = jobs.groupby('zone_id').size()
        #zonal_indicators['workers'] = hh.groupby('zone_id').workers.sum()
        zonal_indicators['Agegrp1'] = persons[persons.age<=4].groupby('zone_id').size() #???
        zonal_indicators['Agegrp2'] = persons[(persons.age>=5)*(persons.age<=17)].groupby('zone_id').size() #???
        zonal_indicators['Agegrp3'] = persons[(persons.age>=18)*(persons.age<=34)].groupby('zone_id').size() #???
        zonal_indicators['Age_18to34'] = persons[(persons.age>=18)*(persons.age<=34)].groupby('zone_id').size()
        zonal_indicators['Agegrp4'] = persons[(persons.age>=35)*(persons.age<=64)].groupby('zone_id').size() #???
        zonal_indicators['Agegrp5'] = persons[persons.age>=65].groupby('zone_id').size() #???
        enroll_ratios = pd.read_csv("data/schdic_taz10.csv")
        school_age_by_district = pd.DataFrame({'children':persons[(persons.age>=5)*(persons.age<=17)].groupby('school_district_id').size()})
        enroll_ratios = pd.merge(enroll_ratios,school_age_by_district,left_on='school_district_id',right_index=True)
        enroll_ratios['enrolled'] = enroll_ratios.enroll_ratio*enroll_ratios.children
        enrolled = enroll_ratios.groupby('zone_id').enrolled.sum()
        zonal_indicators['K12Enroll'] = np.round(enrolled)
        zonal_indicators['PopDens'] = zonal_indicators.HHPop/(parcels.groupby('zone_id').parcel_sqft.sum()/43560)
        zonal_indicators['EmpDens'] = zonal_indicators.EmpPrinc/(parcels.groupby('zone_id').parcel_sqft.sum()/43560)
        # zonal_indicators['EmpBasic'] = jobs[jobs.sector_id.isin([1,3])].groupby('zone_id').size()
        # zonal_indicators['EmpNonBas'] = jobs[~jobs.sector_id.isin([1,3])].groupby('zone_id').size()
        zonal_indicators['Natural_Resource_and_Mining'] = jobs[jobs.sector_id==1].groupby('zone_id').size()
        #zonal_indicators['sector2'] = jobs[jobs.sector_id==2].groupby('zone_id').size()
        zonal_indicators['Manufacturing'] = jobs[jobs.sector_id==3].groupby('zone_id').size()
        zonal_indicators['Wholesale_Trade'] = jobs[jobs.sector_id==4].groupby('zone_id').size()
        zonal_indicators['Retail_Trade'] = jobs[jobs.sector_id==5].groupby('zone_id').size()
        zonal_indicators['Transportation_and_Warehousing'] = jobs[jobs.sector_id==6].groupby('zone_id').size()
        zonal_indicators['Utilities'] = jobs[jobs.sector_id==7].groupby('zone_id').size()
        zonal_indicators['Information'] = jobs[jobs.sector_id==8].groupby('zone_id').size()
        zonal_indicators['Financial_Service'] = jobs[jobs.sector_id==9].groupby('zone_id').size()
        zonal_indicators['Professional_Science_Tec'] = jobs[jobs.sector_id==10].groupby('zone_id').size()
        zonal_indicators['Management_of_CompEnt'] = jobs[jobs.sector_id==11].groupby('zone_id').size()
        zonal_indicators['Administrative_Support_and_WM'] = jobs[jobs.sector_id==12].groupby('zone_id').size()
        zonal_indicators['Education_Services'] = jobs[jobs.sector_id==13].groupby('zone_id').size()
        # zonal_indicators['sector14'] = jobs[jobs.sector_id==14].groupby('zone_id').size()
        # zonal_indicators['sector15'] = jobs[jobs.sector_id==15].groupby('zone_id').size()
        zonal_indicators['Health_Care_and_SocialSer'] = jobs[np.in1d(jobs.sector_id,[14,15,19])].groupby('zone_id').size()
        zonal_indicators['Leisure_and_Hospitality'] = jobs[jobs.sector_id==16].groupby('zone_id').size()
        zonal_indicators['Other_Services'] = jobs[jobs.sector_id==17].groupby('zone_id').size()
        zonal_indicators['sector18'] = jobs[jobs.sector_id==18].groupby('zone_id').size()
        zonal_indicators['sector19'] = jobs[jobs.sector_id==19].groupby('zone_id').size()
        zonal_indicators['Public_Administration'] = jobs[jobs.sector_id==20].groupby('zone_id').size()
        
        hh['schoolkids'] = persons[(persons.age>=5)*(persons.age<=17)].groupby('household_id').size()
        hh.schoolkids = hh.schoolkids.fillna(0)
        zonal_indicators['PrCh21'] = hh[(hh.persons==2)*(hh.schoolkids==1)].groupby('zone_id').size()
        zonal_indicators['PrCh31'] = hh[(hh.persons==3)*(hh.schoolkids==1)].groupby('zone_id').size()
        zonal_indicators['PrCh32'] = hh[(hh.persons==3)*(hh.schoolkids==2)].groupby('zone_id').size()
        zonal_indicators['PrCh41'] = hh[(hh.persons==4)*(hh.schoolkids==1)].groupby('zone_id').size()
        zonal_indicators['PrCh42'] = hh[(hh.persons==4)*(hh.schoolkids==2)].groupby('zone_id').size()
        zonal_indicators['PrCh43'] = hh[(hh.persons==4)*(hh.schoolkids>=3)].groupby('zone_id').size()
        zonal_indicators['PrCh51'] = hh[(hh.persons==5)*(hh.schoolkids==1)].groupby('zone_id').size()
        zonal_indicators['PrCh52'] = hh[(hh.persons==5)*(hh.schoolkids==2)].groupby('zone_id').size()
        zonal_indicators['PrCh53'] = hh[(hh.persons==5)*(hh.schoolkids>=3)].groupby('zone_id').size()
        hh['quartile'] = pd.Series(pd.qcut(hh.income,4).labels, index=hh.index)+1
        zonal_indicators['Inc1HHsze1'] = hh[(hh.persons==1)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze1'] = hh[(hh.persons==1)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze1'] = hh[(hh.persons==1)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze1'] = hh[(hh.persons==1)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze2'] = hh[(hh.persons==2)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze2'] = hh[(hh.persons==2)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze2'] = hh[(hh.persons==2)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze2'] = hh[(hh.persons==2)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze3'] = hh[(hh.persons==3)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze3'] = hh[(hh.persons==3)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze3'] = hh[(hh.persons==3)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze3'] = hh[(hh.persons==3)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze4'] = hh[(hh.persons==4)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze4'] = hh[(hh.persons==4)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze4'] = hh[(hh.persons==4)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze4'] = hh[(hh.persons==4)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['Inc1HHsze5p'] = hh[(hh.persons==5)*(hh.quartile==1)].groupby('zone_id').size()
        zonal_indicators['Inc2HHsze5p'] = hh[(hh.persons==5)*(hh.quartile==2)].groupby('zone_id').size()
        zonal_indicators['Inc3HHsze5p'] = hh[(hh.persons==5)*(hh.quartile==3)].groupby('zone_id').size()
        zonal_indicators['Inc4HHsze5p'] = hh[(hh.persons==5)*(hh.quartile==4)].groupby('zone_id').size()
        zonal_indicators['WkAu10'] = hh[(hh.workers==1)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['WkAu11'] = hh[(hh.workers==1)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['WkAu12'] = hh[(hh.workers==1)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['WkAu13'] = hh[(hh.workers==1)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['WkAu20'] = hh[(hh.workers==2)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['WkAu21'] = hh[(hh.workers==2)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['WkAu22'] = hh[(hh.workers==2)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['WkAu23'] = hh[(hh.workers==2)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['WkAu30'] = hh[(hh.workers>=3)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['WkAu31'] = hh[(hh.workers>=3)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['WkAu32'] = hh[(hh.workers>=3)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['WkAu33'] = hh[(hh.workers>=3)*(hh.cars>=3)].groupby('zone_id').size()

        zonal_indicators['PrAu10'] = hh[(hh.persons==1)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu11'] = hh[(hh.persons==1)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu12'] = hh[(hh.persons==1)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu13'] = hh[(hh.persons==1)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['PrAu20'] = hh[(hh.persons==2)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu21'] = hh[(hh.persons==2)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu22'] = hh[(hh.persons==2)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu23'] = hh[(hh.persons==2)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['PrAu30'] = hh[(hh.persons==3)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu31'] = hh[(hh.persons==3)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu32'] = hh[(hh.persons==3)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu33'] = hh[(hh.persons==3)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['PrAu40'] = hh[(hh.persons==4)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu41'] = hh[(hh.persons==4)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu42'] = hh[(hh.persons==4)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu43'] = hh[(hh.persons==4)*(hh.cars>=3)].groupby('zone_id').size()
        zonal_indicators['PrAu50'] = hh[(hh.persons==5)*(hh.cars==0)].groupby('zone_id').size()
        zonal_indicators['PrAu51'] = hh[(hh.persons==5)*(hh.cars==1)].groupby('zone_id').size()
        zonal_indicators['PrAu52'] = hh[(hh.persons==5)*(hh.cars==2)].groupby('zone_id').size()
        zonal_indicators['PrAu53'] = hh[(hh.persons==5)*(hh.cars>=3)].groupby('zone_id').size()
		
        # zonal_indicators['inc1nc'] = hh[(hh.quartile==1)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc1wc'] = hh[(hh.quartile==1)*(hh.children>0)].groupby('zone_id').size()
        # zonal_indicators['inc2nc'] = hh[(hh.quartile==2)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc2wc'] = hh[(hh.quartile==2)*(hh.children>0)].groupby('zone_id').size()
        # zonal_indicators['inc3nc'] = hh[(hh.quartile==3)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc3wc'] = hh[(hh.quartile==3)*(hh.children>0)].groupby('zone_id').size()
        # zonal_indicators['inc4nc'] = hh[(hh.quartile==4)*(hh.children==0)].groupby('zone_id').size()
        # zonal_indicators['inc4wc'] = hh[(hh.quartile==4)*(hh.children>0)].groupby('zone_id').size()

        zonal_indicators['Inc1w0'] = hh[(hh.quartile==1)*(hh.workers==0)].groupby('zone_id').size()
        zonal_indicators['Inc1w1'] = hh[(hh.quartile==1)*(hh.workers==1)].groupby('zone_id').size()
        zonal_indicators['Inc1w2'] = hh[(hh.quartile==1)*(hh.workers==2)].groupby('zone_id').size()
        zonal_indicators['Inc1w3p'] = hh[(hh.quartile==1)*(hh.workers>=3)].groupby('zone_id').size()
        zonal_indicators['Inc2w0'] = hh[(hh.quartile==2)*(hh.workers==0)].groupby('zone_id').size()
        zonal_indicators['Inc2w1'] = hh[(hh.quartile==2)*(hh.workers==1)].groupby('zone_id').size()
        zonal_indicators['Inc2w2'] = hh[(hh.quartile==2)*(hh.workers==2)].groupby('zone_id').size()
        zonal_indicators['Inc2w3p'] = hh[(hh.quartile==2)*(hh.workers>=3)].groupby('zone_id').size()
        zonal_indicators['Inc3w0'] = hh[(hh.quartile==3)*(hh.workers==0)].groupby('zone_id').size()
        zonal_indicators['Inc3w1'] = hh[(hh.quartile==3)*(hh.workers==1)].groupby('zone_id').size()
        zonal_indicators['Inc3w2'] = hh[(hh.quartile==3)*(hh.workers==2)].groupby('zone_id').size()
        zonal_indicators['Inc3w3p'] = hh[(hh.quartile==3)*(hh.workers>=3)].groupby('zone_id').size()
        zonal_indicators['Inc4w0'] = hh[(hh.quartile==4)*(hh.workers==0)].groupby('zone_id').size()
        zonal_indicators['Inc4w1'] = hh[(hh.quartile==4)*(hh.workers==1)].groupby('zone_id').size()
        zonal_indicators['Inc4w2'] = hh[(hh.quartile==4)*(hh.workers==2)].groupby('zone_id').size()
        zonal_indicators['Inc4w3p'] = hh[(hh.quartile==4)*(hh.workers>=3)].groupby('zone_id').size()
        
        zonal_indicators['Workers4HH_IncomeGroup1'] = hh[hh.quartile==1].groupby('zone_id').workers.sum()
        zonal_indicators['Workers4HH_IncomeGroup2'] = hh[hh.quartile==2].groupby('zone_id').workers.sum()
        zonal_indicators['Workers4HH_IncomeGroup3'] = hh[hh.quartile==3].groupby('zone_id').workers.sum()
        zonal_indicators['Workers4HH_IncomeGroup4'] = hh[hh.quartile==4].groupby('zone_id').workers.sum()
        
        if os.path.exists('gq/tazcounts%s.csv'%year):
            gq = pd.read_csv('gq/tazcounts%s.csv'%year).set_index('tazce10')
            gq['GrPop'] = gq.gq04+gq.gq517+gq.gq1834+gq.gq3564+gq.gq65plus
            zonal_indicators['GrPop'] = gq['GrPop']
            zonal_indicators['Population'] = zonal_indicators['GrPop'] + zonal_indicators['HHPop']
            
        ##Update parcel land_use_type_id
        buildings = buildings.to_frame(['parcel_id','building_type_id','year_built'])
        new_construction = buildings[buildings.year_built==year].groupby('parcel_id').building_type_id.median()
        if len(new_construction) > 0:
            parcels.loc[new_construction.index, 'land_use_type_id'] = new_construction.values
            sim.add_table("parcels", parcels)
        
        emp_btypes = sim.get_injectable('emp_btypes')
        emp_parcels = buildings[np.in1d(buildings.building_type_id,emp_btypes)].groupby('parcel_id').size().index.values
        parcels['emp'] = 0
        parcels.emp[np.in1d(parcels.index.values,emp_parcels)] = 1
        parcels['emp_acreage'] = parcels.emp*parcels.parcel_sqft/43560.0
        zonal_indicators['AcresEmp'] = parcels.groupby('zone_id').emp_acreage.sum()
        
        zonal_indicators['TAZCE10_N'] = zonal_indicators.index.values
        #zonal_indicators = zonal_indicators.fillna(0).reset_index().rename({'ZoneID':'TAZCE10_N'})
        
        taz_table = pd.read_csv("data/taz_table.csv")
        
        merged = pd.merge(taz_table, zonal_indicators, left_on='TAZCE10_N', right_on='TAZCE10_N', how='left')
        
        merged.to_csv(input_file, sep='\t', index = False)
        
        #######################################################################
        ####    TRANSCAD INTERACTIONS #########################################
        #######################################################################
        if sim.get_injectable("transcad_available") == True:
            transcad.transcad_interaction(merged, taz_table)
        
