import pandas as pd

import pyodbc as msql

import numpy as np

from pivottablejs import pivot_ui

import math

from sklearn.linear_model import LinearRegression

import snowflake.connector



model = LinearRegression()

conn = msql.connect(driver='{SQL Server}', 

            server='vapafpdf_db.tdisf-prd.eu-west-1.aws.private-pmideep.biz,49600', 

            database='VP_DataMart',

            trusted_connection='yes')

conn2 = snowflake.connector.connect(

    user="ftozoglu@PMINTL.NET", #You can get it by executing in UI: desc user <username>;

    account="pl47603.eu-west-1", #Add all of the account-name between https:// and snowflakecomputing.com

    authenticator="externalbrowser",

    warehouse='WH_PRD_REPORTING',

    role='PMI_EDP_SPK_SNFK_PMI_FDF_PRD_DATAANALYST_IMDL',

    database = "DB_FDF_PRD"

)

#dynamic variables to use as parameter

time_dim = ['Year']

brand_attributes = ['Taste','Thickness','Flavor','Length'] 

IATA_List = ['CJU']

dimension = ['Flavor','Taste','Thickness','Length'] 

domestic_dimensions = ['Market','EBROMId','EBROM','Taste','Thickness', 'Flavor','Length']

current_year = 2023

previous_year = current_year-1

theyearbefore = previous_year-1

MC_per_Product = pd.read_pickle('datasource/MC_per_Product.pkl')

cat_a_df_vols = pd.read_pickle('datasource/cat_a_df_vols.pkl')

# MC_per_Product_SQL = '''

# SELECT a.Market as DF_Market, a.Location,  a.SKU , a.skuid , b.CR_BrandId, a.[Item per Bundle], 

# 	 	ROUND(SUM(CASE WHEN ([Yearid] = {0} AND [PL Item] = 'PMIDF MC') THEN [Amount USD] ELSE 0 END),2) AS '{0} MC',

#    		ROUND(SUM(CASE WHEN ([Yearid] = {0} AND [PL Item] = 'PMIDF NOR') THEN [Amount USD] ELSE 0 END),2) AS '{0} NOR',

# 		ROUND(SUM(CASE WHEN ([Yearid] = {1} AND [PL Item] = 'PMIDF MC') THEN [Amount USD] ELSE 0 END),2) AS '{1} MC',

# 		ROUND(SUM(CASE WHEN ([Yearid] = {1} AND [PL Item] = 'PMIDF NOR') THEN [Amount USD] ELSE 0 END),2) AS '{1} NOR',

#         ROUND(SUM(CASE WHEN ([Yearid] = {2} AND [PL Item] = 'PMIDF MC') THEN [Amount USD] ELSE 0 END),2) AS '{2} MC',

#   		ROUND(SUM(CASE WHEN ([Yearid] = {2} AND [PL Item] = 'PMIDF NOR') THEN [Amount USD] ELSE 0 END),2) AS '{2} NOR'

# from VDM_POS360_PL a left join VDM_Products b on a.SKUId = b.SKUId  

# LEFT JOIN VDM_Touchpoints c ON a.Povid =c.Povid

# WHERE  a.[Trade Channel] = 'airports'

# and a.[Product Category] = 'cigarettes'

# And c.departurearrival != 'A'

# and [PL Item] IN ('PMIDF MC', 'PMIDF NOR')  

# AND DataVersionType ='AC'

# AND [Year] in ('COT {0} ACT','COT {1} ACT','COT {2} ACT')

# GROUP by a.Market, a.Location,a.sku, a.skuid, a.[Item per Bundle],b.CR_BrandId

# '''.format(theyearbefore,previous_year,current_year)

# MC_per_Product = pd.read_sql_query(MC_per_Product_SQL, conn)

# MC_per_Product.to_pickle('datasource/MC_per_Product.pkl')



# df_vols_query = ''' 

# SELECT  a.Market as DF_Market, a.Location , TMO  ,[Brand Family] , CR_BrandId, [SKU/Mkt Brand] as 'SKU',[Item per Bundle],

# 	 	SUM(CASE WHEN [Year] = {0} THEN Volume ELSE 0 END) AS '{0} Volume',

# 		SUM(CASE WHEN [Year] = {1} THEN Volume ELSE 0 END) AS '{1} Volume', 

# 		SUM(CASE WHEN [Year] = {2} THEN Volume ELSE 0 END) AS '{2} Volume',

# 		COUNT(DISTINCT CASE WHEN ([Year] = {0} and Volume > 1) THEN  MONTH([Date])  ELSE NULL END) AS '{0}Month',

# 		COUNT(DISTINCT CASE WHEN ([Year] = {1} and Volume > 1) THEN  MONTH([Date])  ELSE NULL END) AS '{1}Month',

# 		COUNT(DISTINCT CASE WHEN ([Year] = {2} and Volume > 1) THEN  MONTH([Date])  ELSE NULL END) AS '{2}Month',

# 		SUM(CASE WHEN [Year] = {0} and [Item per Bundle] > 0 THEN RSP* (Volume/[Item per Bundle]) ELSE 0 END) AS '{0} Revenue',

# 		SUM(CASE WHEN [Year] = {1} and [Item per Bundle] > 0 THEN RSP* (Volume/[Item per Bundle]) ELSE 0 END) AS '{1} Revenue',

# 		SUM(CASE WHEN [Year] = {2} and [Item per Bundle] > 0 THEN RSP* (Volume/[Item per Bundle]) ELSE 0 END) AS '{2} Revenue'

# FROM VDM_Prices a

# LEFT JOIN VDM_Touchpoints b ON a.Povid =b.Povid

# WHERE a.[Trade Channel] = 'airports' and [Year] in ({0},{1},{2}) 

# and [Product Category] = 'Cigarettes' and QUALITY in ('REAL' , 'Simulated', 'Estimated') 

# and a.Market not in ('France DP, Spain DP, Finland DP')

# and Volume > 0

# And b.departurearrival != 'A'

# GROUP BY  a.Market , a.Location , TMO ,[Brand Family]  ,CR_BrandId, [SKU/Mkt Brand] , [Item per Bundle]

# '''.format(theyearbefore,previous_year,current_year)

# cat_a_df_vols = pd.read_sql_query(df_vols_query, conn)

# cat_a_df_vols.to_pickle('datasource/cat_a_df_vols.pkl')

DF_Vols_w_Financials = cat_a_df_vols.merge(MC_per_Product[['DF_Market','Location','CR_BrandId',"{} MC".format(previous_year),

                                            "{} NOR".format(previous_year),"{} MC".format(current_year),"{} NOR".format(current_year)]], 

                                           how = 'left', on = ['DF_Market','Location','CR_BrandId']).fillna(0)

DF_Vols_w_Financials = DF_Vols_w_Financials[DF_Vols_w_Financials["{} Volume".format(current_year)]>0]



DF_Vols_w_Financials['LYRevenueAvg'] = np.where(DF_Vols_w_Financials["{}Month".format(previous_year)] == 0, 0, 

                            round(DF_Vols_w_Financials["{} Revenue".format(previous_year)] / DF_Vols_w_Financials["{}Month".format(previous_year)],0))

DF_Vols_w_Financials['CYRevenueAvg'] = np.where(DF_Vols_w_Financials["{}Month".format(current_year)] == 0, 0, 

                                         round(DF_Vols_w_Financials["{} Revenue".format(current_year)] / DF_Vols_w_Financials["{}Month".format(current_year)],0))

#Calculating Growth

DF_Vols_w_Financials['Growth'] = (DF_Vols_w_Financials['CYRevenueAvg']-DF_Vols_w_Financials['LYRevenueAvg']) / DF_Vols_w_Financials['LYRevenueAvg']

#Finding Margin:

DF_Vols_w_Financials['Margin'] = np.where(

                                        DF_Vols_w_Financials["{} NOR".format(current_year)]<=0,0,

                                        ((DF_Vols_w_Financials["{} MC".format(current_year)] /DF_Vols_w_Financials["{}Month".format(current_year)]) /  

                                        (DF_Vols_w_Financials["{} NOR".format(current_year)] / DF_Vols_w_Financials["{}Month".format(current_year)])))

pmi_margins = DF_Vols_w_Financials[DF_Vols_w_Financials['TMO'] == 'PMI']



#Brand Family Margins



pmi_margins['Margin_Volume'] = round((pmi_margins["{} Volume".format(current_year)] * pmi_margins['Margin'].fillna(0)),0).astype(int)

pmi_margins = pmi_margins.groupby(['DF_Market','Location','Brand Family']).sum().reset_index()

pmi_margins = pmi_margins[['DF_Market','Location','Brand Family',"{} Volume".format(current_year),"{} MC".format(current_year),'Margin_Volume']]

pmi_margins['Brand Family Margin'] = pmi_margins['Margin_Volume'] / pmi_margins["{} Volume".format(current_year)]



# Merging Brand Family Margins to Original Table

SKU_by_Vols_Margins =  DF_Vols_w_Financials.merge(pmi_margins[['DF_Market','Location','Brand Family','Brand Family Margin']],

                                                  how = 'left', on =['DF_Market','Location','Brand Family']).fillna(0)

SKU_by_Vols_Margins['Margin Comparison'] = np.where(

                                        SKU_by_Vols_Margins['Brand Family Margin']< SKU_by_Vols_Margins['Margin'], 1 , 0)

# SKU_by_Vols_Margins = SKU_by_Vols_Margins[SKU_by_Vols_Margins['TMO'] == 'PMI']

pmi_margins[pmi_margins['Location'] == 'Zurich']

# Calculating the number of SKUs for Green & Red Flag SKUs

no_of_sku = DF_Vols_w_Financials.groupby(['DF_Market', 'Location'])['SKU'].count().reset_index()

no_of_sku = no_of_sku.rename(columns = {'SKU': 'TotalSKU'})

no_of_sku['GreenFlagSKU'] = (no_of_sku['TotalSKU']*0.05).apply(np.ceil)

no_of_sku['RedFlagSKU'] = round(no_of_sku['TotalSKU']*0.25 , 0)

no_of_sku[no_of_sku['Location'] == 'Zurich']

# Rule 1: SKUs in the TOP 5% percentile 

gf = pd.DataFrame(columns=['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU', 'Item per Bundle', 

                           "{} Volume".format(previous_year), "{} Volume".format(current_year),"{}Month".format(previous_year), "{}Month".format(current_year), 

                           "{} Revenue".format(previous_year), "{} Revenue".format(current_year), "{} MC".format(previous_year), "{} NOR".format(previous_year), 

                           "{} MC".format(current_year), "{} NOR".format(current_year),  'LYRevenueAvg','CYRevenueAvg', 'Growth', 'Margin'])



for i in DF_Vols_w_Financials.Location.unique():



    DF_Vols = DF_Vols_w_Financials[DF_Vols_w_Financials['Location'] == i]

    green_flag1 = DF_Vols.sort_values("{} Volume".format(current_year), ascending = False).head(no_of_sku[no_of_sku['Location'] == i].iloc[0,-2].astype('int'))

    green_flag1['Green1'] = 1 

    green_flag1 = green_flag1[green_flag1['TMO'] == 'PMI']

    gf = pd.concat([gf,green_flag1],ignore_index = True)

gf[gf['Location'] == 'Zurich']

#Rule 2: If both:

    # a. Contribution to category revenue growth is in the top 5% percentile of category SKUs, and

    # b. Margin % of SKU is greater than margin % of total relevant brand



# green_flag2 = SKU_by_Vols_Margins[SKU_by_Vols_Margins['Growth']>0]



gf2 = pd.DataFrame(columns=['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU', 'Item per Bundle', 

                             "{} Volume".format(previous_year), "{} Volume".format(current_year),"{}Month".format(previous_year), "{}Month".format(current_year), 

                             "{} Revenue".format(previous_year), "{} Revenue".format(current_year), "{} MC".format(previous_year), "{} NOR".format(previous_year), 

                             "{} MC".format(current_year), "{} NOR".format(current_year),  'LYRevenueAvg','CYRevenueAvg', 'Growth', 'Margin'])



for i in DF_Vols_w_Financials.Location.unique():



    DF_Vols = SKU_by_Vols_Margins[SKU_by_Vols_Margins['Location'] == i]

    green_flag2 = DF_Vols.sort_values('Growth', ascending = False).head(no_of_sku[no_of_sku['Location'] == i].iloc[0,-2].astype('int'))

    green_flag2 = green_flag2[green_flag2['TMO'] == 'PMI']

    green_flag2 = green_flag2[green_flag2['Margin Comparison'] == 1]

    green_flag2['Green Flag2'] = 1

    gf2 = pd.concat([gf2,green_flag2],ignore_index = True)



gf2[gf2['Location'] == 'Zurich']

green_list = pd.concat([gf,gf2])

green_list = green_list[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle']]

green_list = green_list.drop_duplicates()

green_list['Green']  = 1

green_list[green_list['Location'] == 'Zurich']

rf1 = pd.DataFrame(columns=['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle'])



for i in DF_Vols_w_Financials.Location.unique():

    Red_Vols = DF_Vols_w_Financials[DF_Vols_w_Financials['Location'] == i]

    red_flag1 = Red_Vols.sort_values("{} Volume".format(current_year), ascending = True).head(no_of_sku[no_of_sku['Location'] == i].iloc[0,-1].astype('int'))

    red_flag1 = red_flag1[red_flag1['TMO'] == 'PMI']

    

    red_flag1_1 = Red_Vols.sort_values('Growth', ascending = True).head(no_of_sku[no_of_sku['Location'] == i].iloc[0,-1].astype('int'))

    red_flag1_1 = red_flag1_1[red_flag1_1['TMO'] == 'PMI']

    

    red_flag_intersection = np.intersect1d(red_flag1.CR_BrandId, red_flag1_1.CR_BrandId)

    

    red_flag1_2 =  pd.concat([red_flag1,red_flag1_1], ignore_index = True)

    red_flag1_2 = red_flag1_2[red_flag1_2['CR_BrandId'].isin(red_flag_intersection) ].drop_duplicates()

    red_flag1_2 = red_flag1_2[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle']]

    

    rf1 = pd.concat([rf1,red_flag1_2], ignore_index = True)  



rf1[rf1['Location'] == 'Zurich']

rf2 = pd.DataFrame(columns=['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle'])



for i in DF_Vols_w_Financials.Location.unique():

    

    red_flag2_1 = SKU_by_Vols_Margins[SKU_by_Vols_Margins['Location'] ==  i]

    red_flag2_1 = red_flag2_1.sort_values('Growth', ascending = True).head(no_of_sku[no_of_sku['Location'] == i].iloc[0,-1].astype('int'))

    red_flag2_1 = red_flag2_1[red_flag2_1['TMO'] == 'PMI']

    red_flag2_1 = red_flag2_1[red_flag2_1['Margin Comparison'] == 0]

    red_flag2_1 = red_flag2_1[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle']]

    rf2 = pd.concat([rf2,red_flag2_1 ], ignore_index = True)



rf2[rf2['Location'] == 'Zurich']

red_list = pd.concat([rf1, rf2], ignore_index = True).drop_duplicates()

red_list['Red'] = 1



red_list[red_list['Location'] == 'Zurich']

green_red_list = green_list.merge(red_list, how = 'outer', on = ['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle']).fillna(0)

green_red_list['Check'] = np.where(green_red_list['Green'] != green_red_list['Red'], 'OK','Problem')

green_red_list[(green_red_list['Location'] == 'Zurich') & (green_red_list['Check'] == 'Problem')].sort_values('Location')

green_red_list = green_red_list[green_red_list['Check'] != 'Problem']

green_red_list['Status'] = np.where(green_red_list['Green'] == 1 , 'Green', 'Red')


category_a_0 = SKU_by_Vols_Margins[SKU_by_Vols_Margins['TMO'] == 'PMI']



category_a_1 = (category_a_0.merge(

    green_red_list[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle','Status']], 

    how = 'left',on = ['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle'])).fillna(0)


total_sku = category_a_1.groupby(['DF_Market', 'Location'])['CR_BrandId'].count().reset_index()

total_sku = total_sku.rename(columns = {'CR_BrandId': 'TotalSKU'})

ct_green = category_a_1[category_a_1['Status'] == 'Green']

ct_green = ct_green.groupby(['DF_Market', 'Location'])['CR_BrandId'].count().reset_index()

ct_green = ct_green.rename(columns = {'CR_BrandId': 'GreenSKU'})

ct_red = category_a_1[category_a_1['Status'] == 'Red']

ct_red = ct_red.groupby(['DF_Market', 'Location'])['CR_BrandId'].count().reset_index()

ct_red = ct_red.rename(columns = {'CR_BrandId': 'RedSKU'})



ct_gr_red = ct_red.merge(ct_green, how = 'outer', on = ['DF_Market', 'Location'])


calculation_table = total_sku.merge(ct_gr_red, how = 'outer', on =  ['DF_Market', 'Location'])

calculation_table['RedSKU'] = calculation_table['RedSKU'].fillna(0).astype('int')

calculation_table['GreenSKU'] = calculation_table['GreenSKU'].fillna(0).astype(int)

calculation_table[calculation_table['Location'] == 'Zurich']

location = []

score_a= []



for i in calculation_table.Location.unique() :

    

    ct = calculation_table[calculation_table['Location'] == i]

    score = ((ct['GreenSKU'].iloc[0] -(ct['RedSKU'].iloc[0] *2 )) / ct['TotalSKU'].iloc[0])*100

    location.append(i)

    score_a.append(score)



list_of_tuples = list(zip(location, score_a))

cat_a_scores = pd.DataFrame(list_of_tuples,columns=['Location','Score_A'] ).fillna(0)

cat_a_scores['ScaledScore'] = round((cat_a_scores['Score_A']-(-200))*(10/300),2)

cat_a_scores[cat_a_scores['Location'] == 'Zurich']

# SELMA_DF_map = pd.read_excel('datasource/Pmidf_SELMA_output_20230717.xlsx')

SELMA_DF_map = pd.read_pickle('datasource/SELMA_DF_map.pkl')

df_vols = pd.read_pickle('datasource/cat_a_df_vols.pkl')

base_list= pd.read_pickle('datasource/base_list.pkl')

# base_query = ''' SELECT distinct [SKU/Mkt Brand] as SKU ,[Item per Bundle] , CR_BrandId , a.Market as DF_Market, a.Location , TMO

# FROM VDM_Prices a

# LEFT JOIN VDM_Touchpoints b ON a.Povid =b.Povid

# WHERE a.[Trade Channel] = 'airports' and [Date] > '{}-06-01'

# And b.departurearrival != 'A'

# and [Product Category] = 'Cigarettes' and QUALITY in ('REAL' , 'Simulated', 'Estimated') and Volume > 0

# GROUP BY [SKU/Mkt Brand] ,[Item per Bundle] , CR_BrandId , a.Market, a.Location , TMO '''.format(current_year)

# base_list = pd.read_sql_query(base_query, conn)

# base_list.to_pickle('datasource/base_list.pkl')

base_list = base_list[['DF_Market','Location','TMO', 'CR_BrandId', 'SKU', 'Item per Bundle']]

SELMA_DF_map = SELMA_DF_map.drop_duplicates(['CR_BrandId','Location'])

SELMA_DF_map['Length'] = np.where(SELMA_DF_map['Length'].isin(['REGULAR SIZE','REGULAR FILTER','SHORT SIZE','LONG FILTER','NAN']),'KS', 

                         np.where(SELMA_DF_map['Length'].isin(['LONGER THAN KS','100','LONG SIZE','EXTRA LONG','SUPER LONG']),'LONG',

                                  SELMA_DF_map['Length'] ))

duty_free_volumes = df_vols.copy()

duty_free_volumes['key'] = duty_free_volumes['CR_BrandId'].astype('str') + '-' + duty_free_volumes['Item per Bundle'].astype('str')

duty_free_volumes[duty_free_volumes['Location'] == 'Zurich'].head(2)

#All Universe

category_list = base_list.copy()

category_list['key'] =category_list['CR_BrandId'].astype('str') + '-' + category_list['Item per Bundle'].astype('str')

category_list[category_list['Location'] == 'Zurich'].head(3)

duty_free_volumes2 = duty_free_volumes[['DF_Market','Location','TMO','CR_BrandId','SKU',"{} Volume".format(current_year)]]

tobacco_range = category_list.merge(duty_free_volumes[['DF_Market', 'Location','key',"{} Volume".format(current_year)]], how = 'left', 

                                                    on = ['DF_Market', 'Location','key']).fillna(0)

tobacco_range['TMO'] = np.where(tobacco_range['TMO']!= 'PMI', 'Comp','PMI')

tobacco_range = tobacco_range[tobacco_range['CR_BrandId']!=0]

tobacco_range[tobacco_range['Location']=='Zurich'].head(3)

tobacco_range2 = tobacco_range[['DF_Market','Location','TMO','CR_BrandId','SKU',"{} Volume".format(current_year)]]

tobacco_range2 = pd.pivot_table(tobacco_range2, index = ['DF_Market','Location','TMO','CR_BrandId'],

               aggfunc={ "{} Volume".format(current_year): np.sum, 'SKU': np.count_nonzero}).reset_index()

selma_df_products = SELMA_DF_map.copy() 

selma_df_products = selma_df_products[selma_df_products['Product Category'] == 'Cigarettes'].reset_index()

selma_df_products_2 = selma_df_products[['DF_Market','Location', 'CR_BrandId']+brand_attributes ]

selma_df_products_3 = selma_df_products_2.drop_duplicates()

selma_df_products_3 = selma_df_products_3[selma_df_products_3['CR_BrandId']!= 0]

Market_Mix = selma_df_products_3.merge(Market_Mix[['DF_Market', 'Location','CR_BrandId', 'TMO', 'SKU',

                                                    "{} Volume".format(current_year)]], how = 'left' , on = ['DF_Market', 'Location','CR_BrandId'])

Market_Mix = Market_Mix[Market_Mix['SKU'].notnull()]

Market_Mix = Market_Mix[Market_Mix['SKU']!=0]



Market_Mix[Market_Mix['Location'] == 'Zurich']

all_market = pd.pivot_table(tobacco_range, index = ['DF_Market','Location','TMO'],  

                                                   aggfunc={ 'SKU': np.count_nonzero}).reset_index()

all_market = all_market.rename(columns = {'SKU': 'Total TMO'})

Market_Summary0 = pd.pivot_table(Market_Mix, index = ['DF_Market','Location', 'TMO']+brand_attributes,  

                                                   aggfunc={ "{} Volume".format(current_year): np.sum, 'SKU': np.sum}).reset_index()

Market_Summary = Market_Summary0.merge(all_market, how = 'left', on =['DF_Market','Location','TMO'] )

Market_Summary['SoM'] = round(Market_Summary['SKU']*100/Market_Summary['Total TMO'],1)

Market_Summary = Market_Summary[['DF_Market','Location','TMO']+ brand_attributes+['SKU','Total TMO', 'SoM']]

Market_Summary[Market_Summary['Location'] == 'Zurich']

Market_Summary_PMI = Market_Summary[Market_Summary['TMO'] == 'PMI']

Market_Summary_Comp = Market_Summary[Market_Summary['TMO'] == 'Comp']

Market_Summary_PMI = Market_Summary_PMI.rename(columns= {'SoM':'SoM_PMI','SKU':'PMI_Seg_SKU', 'Total TMO': 'PMI Total'})

Market_Summary_Comp = Market_Summary_Comp.rename(columns= {'SoM':'SoM_Comp','SKU':'Comp_Seg_SKU', 'Total TMO': 'Comp Total'})

Market_Summary_Comp = Market_Summary_Comp[['DF_Market','Location']+ brand_attributes+ ['Comp_Seg_SKU','Comp Total','SoM_Comp']]

Market_Summary_PMI = Market_Summary_PMI[['DF_Market','Location']+ brand_attributes+['PMI_Seg_SKU','PMI Total','SoM_PMI']]

Market_Summary_PMI[(Market_Summary_PMI['Location'] == 'Zurich')&(Market_Summary_PMI['Flavor'] == 'Regular') & (Market_Summary_PMI['Taste'] == 'Full Flavor')&

                  (Market_Summary_PMI['Length'] == 'KS') & (Market_Summary_PMI['Thickness'] == 'STD')]

Market_Summary_Comp[(Market_Summary_Comp['Location'] == 'Zurich')&(Market_Summary_Comp['Flavor'] == 'Regular') & (Market_Summary_Comp['Taste'] == 'Full Flavor') & 

                    (Market_Summary_Comp['Length'] == 'KS') & (Market_Summary_Comp['Thickness'] == 'STD')]

Market_Summary_Delta = Market_Summary_Comp.merge(Market_Summary_PMI[['DF_Market','Location','Flavor','Taste','Thickness',

                                                'Length','PMI_Seg_SKU','PMI Total','SoM_PMI']], how = 'outer', 

                                                 on = ['DF_Market','Location','Flavor','Taste','Thickness','Length']).fillna(0)

Market_Volume_Table = Market_Summary0.groupby(['DF_Market', 'Location']+ brand_attributes).sum("{} Volume".format(current_year)).reset_index()

Market_Volume_Table[(Market_Volume_Table['Location'] == 'Zurich') &(Market_Volume_Table['Flavor'] == 'Regular') & (Market_Volume_Table['Taste'] == 'Full Flavor')&

                  (Market_Volume_Table['Length'] == 'KS') & (Market_Volume_Table['Thickness'] == 'STD')]

Market = Market_Summary_Delta.merge(Market_Volume_Table[['DF_Market', 'Location']+ brand_attributes+ ["{} Volume".format(current_year)]] 

                                    ,how = 'left', on = ['DF_Market', 'Location']+ brand_attributes)

Market['SKU_Delta'] = Market['SoM_PMI'] - Market['SoM_Comp']

Market[(Market['Location'] == 'Zurich') &(Market['Flavor'] == 'Regular') & (Market['Taste'] == 'Full Flavor')&

                  (Market['Length'] == 'KS') & (Market['Thickness'] == 'STD')]

location = []

score = []

num_of_pmi_sku = []

num_of_comp_sku = []

pmi_cot = []

comp_cot = []

for i in Market['Location'].unique():

    looped_market = Market[Market['Location']== i]

    X, y = looped_market[["SoM_PMI"]], looped_market[["SoM_Comp"]]

    model.fit(X, y)

    r_squared = model.score(X, y)

    market_score = round(r_squared*10,2)

    

    skunum =  max(Market[(Market['Location']== i)].iloc[:,-4])

    compsku =  max(Market[(Market['Location']== i)].iloc[:,-7])

    pmi_vol = Market_Mix[(Market_Mix['Location']== i) & (Market_Mix['TMO']== 'PMI')].sum().iloc[-1].astype('int')

    comp_vol = Market_Mix[(Market_Mix['Location']== i) & (Market_Mix['TMO']!= 'PMI')].sum().iloc[-1].astype('int')

    

    location.append(i)

    score.append(market_score)

    num_of_pmi_sku.append(skunum)

    num_of_comp_sku.append(compsku)

    pmi_cot.append(pmi_vol)

    comp_cot.append(comp_vol)

#     print(i)

list_of_tuples = list(zip(location, score,num_of_pmi_sku,num_of_comp_sku,pmi_cot,comp_cot))

cat_b_scores = pd.DataFrame(list_of_tuples,columns=['Location', 'RSQ', 'NumPMI_SKU','NumComp_SKU','PMI Volume','Comp Volume']).fillna(0)

cat_b_scores[cat_b_scores['Location'] == 'Zurich']

cat_b_scores[cat_b_scores['Location'] == 'Jeju']

cat_b_scores = cat_b_scores.rename({'RSQ': 'Cat_B'})

mrk_nat_map = pd.read_pickle('datasource/mrk_nat_map.pkl')

IATA_Location = pd.read_pickle('datasource/IATA_Location.pkl')

SELMA_dom_map = pd.read_pickle('datasource/SELMA_dom_map.pkl')

SELMA_DF_map = pd.read_pickle('datasource/SELMA_DF_map.pkl')

Pmidf_products = pd.read_pickle('datasource/Pmidf_products.pkl')

dom_prods_data = pd.read_pickle('datasource/dom_prods_data.pkl')

dom_ims_data = pd.read_pickle('datasource/dom_ims_data.pkl')

DomesticVolumes = pd.read_pickle('datasource/DomesticVolumes.pkl')

CF_data = pd.read_pickle('datasource/CF_data.pkl')

PAX_data = pd.read_pickle('datasource/PAX_data.pkl')

DF_Vol_data = pd.read_pickle('datasource/DF_Vol_data.pkl')

SELMA_dom_map = pd.read_excel('datasource/Dom_SELMA_output_2023120.xlsx')

SELMA_dom_map.to_pickle('datasource/SELMA_dom_map.pkl')

# sql_PMIDF_script = '''SELECT * FROM VDM_Products vp '''

# Pmidf_products = pd.read_sql(sql_PMIDF_script,conn)

# Pmidf_products.to_pickle('datasource/Pmidf_products.pkl')



# sql_dom_script ='''SELECT * FROM VP_DataMart.dbo.VDM_PTF_DOM_PRODUCTS'''

# dom_prods_data = pd.read_sql(sql_dom_script,conn)

# dom_prods_data.to_pickle('datasource/dom_prods_data.pkl')





# sql_Domest_script = '''SELECT [Year], a.EBROMId, sum(Volume) as Volume

# FROM VP_DataMart.dbo.VDM_PTF_DOM_IMS a LEFT JOIN VDM_PTF_DOM_PRODUCTS b

# ON a.EBROMId = b.EBROMId 

# WHERE  Volume > 0 and Year = {} and a.EBROMId != 0

# AND b.[Product Category] = 'Cigarettes'

# GROUP BY [Year], a.EBROMId'''.format(current_year)

# dom_ims_data = pd.read_sql(sql_Domest_script,conn)

# dom_ims_data.to_pickle('datasource/dom_ims_data.pkl')



# DomesticVolumes = '''SELECT [Year], vpdp.EBROMId, Market, EBROM, SUM(Volume) as Volume

# FROM VDM_PTF_DOM_IMS vpdi LEFT JOIN VDM_PTF_DOM_PRODUCTS vpdp on vpdi.EBROMId = vpdp.EBROMId 

# WHERE [Year] = {} AND EBROM IS NOT NULL AND [Product Category] = 'Cigarettes'

# group by [Year], vpdp.EBROMId, Market, EBROM'''.format(current_year)

# DomesticVolumes = pd.read_sql(DomesticVolumes,conn)

# DomesticVolumes.to_pickle('datasource/DomesticVolumes.pkl')



# country_figures = ''' SELECT * FROM VP_DataMart.dbo.VDM_country_figures '''

# CF_data = pd.read_sql(country_figures,conn)

# CF_data.to_pickle('datasource/CF_data.pkl')





# COT_query = ''' SELECT [Year], [Product Category], a.Location, a.Market as DF_Market,a.TMO,a.[Brand Family],a.CR_BrandId, SUM(Volume) AS DF_Vol 

# FROM VDM_Prices a left join VDM_Touchpoints b on a.PovId = b.PovId 

# WHERE a.[Trade Channel] = 'airports' and [Year] = {} 

# and [Product Category] = 'Cigarettes' and QUALITY in ('REAL' , 'Simulated', 'Estimated') and b.DepartureArrival in ('D', 'B')

# GROUP BY [Year],[Product Category], a.Location, a.Market,a.TMO,a.[Brand Family],a.CR_BrandId'''.format(current_year)

# DF_Vol_data = pd.read_sql_query(COT_query, conn)

# DF_Vol_data.to_pickle('datasource/DF_Vol_data.pkl')



Pax_Nat = ''' SELECT YEAR_NUM AS "Year" , IATA_CODE AS "IATA", DF_MARKET_NAME AS "Market",  AIRPORT_NAME , NATIONALITY AS "Nationality"

, sum(PAX_QUANTITY*1000) AS "Pax" FROM DB_FDF_PRD.CS_PAXLANU.PAX_FACT_PAX_QUANTITY 

WHERE DATA_SOURCE_NAME = 'M1ndset Nationalities'

AND YEAR_NUM = {} 

AND ARRIVAL_DEPARTURE = 'D' 

AND validity_desc = 'Actual'

AND DOM_INTL = 'International'

GROUP BY YEAR_NUM, IATA_CODE, DF_MARKET_NAME, AIRPORT_NAME , NATIONALITY '''.format(current_year)

PAX_data = pd.read_sql(Pax_Nat,conn2)

PAX_data.to_pickle('datasource/PAX_data.pkl')

SELMA_dom_map['Length'] = np.where(SELMA_dom_map['Length'].isin(['REGULAR SIZE','REGULAR FILTER','SHORT SIZE','LONG FILTER']), 'KS',

                          np.where(SELMA_dom_map['Length'].isin(['LONGER THAN KS','100','LONG SIZE','EXTRA LONG','SUPER LONG']),'LONG',

                          SELMA_dom_map['Length']))



SELMA_dom_map['Thickness'] = np.where(SELMA_dom_map['Thickness']== 'FAT', 'STD',

                             SELMA_dom_map['Thickness'])



SELMA_dom_map = SELMA_dom_map.merge(dom_prods_data[['EBROMId']], how = 'left', on = 'EBROMId')

SELMA_DF_map = SELMA_DF_map.drop_duplicates(['CR_BrandId','Location'])

SELMA_DF_map['Length'] = np.where(SELMA_DF_map['Length'].isin(['REGULAR SIZE','REGULAR FILTER','SHORT SIZE','LONG FILTER','NAN']),'KS', 

                         np.where(SELMA_DF_map['Length'].isin(['LONGER THAN KS','100','LONG SIZE','EXTRA LONG','SUPER LONG']),'LONG',

                                  SELMA_DF_map['Length'] ))

DF_Vol_data = DF_Vol_data[DF_Vol_data['Year'] == current_year]

DF_Vol_data_cleared = DF_Vol_data[DF_Vol_data['CR_BrandId']!= 0 ]

DF_Vol_data_cleared = DF_Vol_data[DF_Vol_data['Product Category'] == 'Cigarettes' ]

DF_Vol_data_cleared = DF_Vol_data_cleared[DF_Vol_data_cleared['TMO'] == 'PMI']

base_list['key'] = base_list['Location'] + base_list['CR_BrandId'].astype('str')

DF_Vol_data_cleared['key'] = DF_Vol_data_cleared['Location'] + DF_Vol_data_cleared['CR_BrandId'].astype('str')

checklist = list(base_list['key'].unique())

DF_Vol_data_cleared = DF_Vol_data_cleared[DF_Vol_data_cleared['key'].isin(checklist)]

DF_Vol_data_cleared = DF_Vol_data_cleared.drop(columns= 'key')

DF_Vol_data_wLocation = DF_Vol_data_cleared.merge(IATA_Location,how='left', on ='Location')

DF_Vols = DF_Vol_data_wLocation.merge(SELMA_DF_map[['CR_BrandId','Location']+brand_attributes], 

                                      how = 'left', on = ['CR_BrandId','Location'])


#Aggregating domestic data to time-dimension

dom_ims_data = dom_ims_data.groupby(time_dim + ['EBROMId']).sum('Volume').reset_index()

#Aggregating domestic data to time-dimension

DomesticVolumesYearly = DomesticVolumes.groupby(time_dim +['Market', 'EBROMId']).sum('Volume').reset_index()

TotalDomMarkets  = DomesticVolumesYearly.groupby(time_dim + ['Market']).sum('Volume').reset_index()

DomesticSom = DomesticVolumesYearly.merge(TotalDomMarkets, how = 'left', on = time_dim+ ['Market'])

DomesticSom = DomesticSom.rename(columns = {'Volume_x': 'SKU_Vol', 'Volume_y': 'Market_Vol'})

DomesticSom = DomesticSom[time_dim + ['Market','SKU_Vol', 'Market_Vol' ]]

DomesticSom['SodRealDom']  = DomesticSom['SKU_Vol'] / DomesticSom['Market_Vol']

## cleaning the date and adding respective market info

pax_d1 = PAX_data[time_dim + ['IATA', 'Market', 'Nationality', 'Pax']].copy()

pax_d2 = pax_d1.groupby(time_dim + ['IATA', 'Market', 'Nationality']).sum().reset_index()

pax_d3 = pax_d2.merge(mrk_nat_map, how='left', left_on ='Nationality',right_on='Nationalities')



#adding Country Figures data

CF_data = CF_data.rename(columns={'TotalSmokingPrevalence':'SmokingPrevelance'})

cf_d2 = CF_data[['KFYear','Country','SmokingPrevelance','InboundAllowance','ADCStick','PurchaserRate']].copy()



#cleaning country figures data with some assumptions

cf_d2['ADCStick'] = cf_d2['ADCStick'].fillna(15.0)

cf_d2['InboundAllowance'] = cf_d2['InboundAllowance'].fillna(400.0)

cf_d2['PurchaserRate'] = np.where(cf_d2['PurchaserRate'] == cf_d2['PurchaserRate'], 

                                  cf_d2['PurchaserRate'], cf_d2['SmokingPrevelance'])



pax_d4 = pax_d3.merge(cf_d2, how='left',left_on=['Nationalities','Year'],right_on=['Country','KFYear'])



# correctig PAX numbers and rounding to one passenger

pax_d4['Pax'] = np.ceil(pax_d4['Pax'] * 1000)



#model is currently based on the Purchaser Rate (proxied from smoking prevalence) 0.9 coefficient is added

pax_d4['LANU'] = pax_d4['Pax'] * pax_d4['SmokingPrevelance'] * 0.9 # * pax_d4['InboundAllowance']

pax_d4['LANU'] = np.ceil(pax_d4['LANU'])

pax_d4['InboundAllowance'] = pax_d4['InboundAllowance'].astype(float)

pax_d4['StickCons'] =  pax_d4['LANU'] * pax_d4['InboundAllowance']



pax_fin_ = pax_d4[time_dim + ['Market','IATA','Nationality',

                  'Countries','LANU','StickCons']].rename(columns={'Market':'DF_Market'})



# pax_fin - the most granular table that can be aggregated later to needed level



pax_fin = pax_fin_.groupby(time_dim + ['DF_Market','IATA',

                                         'Nationality','Countries']).sum('StickCons').reset_index()

#completing groupping attributes

dom_attr = SELMA_dom_map[domestic_dimensions].merge(dom_prods_data[['EBROMId','TMO','Brand Family']], 

                                                    how='left',on='EBROMId').fillna('NaN')

#adding domestic IMS data

dom_ims2 = dom_ims_data.merge(dom_attr,how='left',on='EBROMId')

dom_ims2 = dom_ims2[dom_ims2['Market'] != 'PMIDF']

dom_ims2 = dom_ims2[dom_ims2['EBROMId'] != 0]

dom_ims2 = dom_ims2[dom_ims2['EBROM'] == dom_ims2['EBROM']]

dom_ims2['Market'] =dom_ims2['Market'].replace('PRC', 'China') 

#dom_ims2['Market'] =dom_ims2['Market'].replace('Spain Mainland', 'Spain') 



dom_totals = dom_ims2.groupby(time_dim+['Market']).sum().reset_index().rename(columns={'Volume':'TotVol'})



#calculating Share of Volumes (Share of Market) by month

dom_sov = dom_ims2.merge(dom_totals[time_dim+['Market','TotVol']], how='left', on=time_dim +['Market'])



dom_sov['SoDom'] = dom_sov['Volume'] / dom_sov['TotVol']



dom_fin = dom_sov[time_dim + domestic_dimensions + ['TMO','SoDom']].rename(columns={'Market':'Dom_Market'})





# dom_fin - the most granular table with domestic volumes that can be aggregated later to needed level

projected_vol_by_sku = pax_fin.merge(dom_fin, how='left',left_on = time_dim +['Countries'] ,

                            right_on= time_dim + ['Dom_Market'])

projected_vol_by_sku['Proj_Vol_bySKU'] = round(projected_vol_by_sku['SoDom'] * projected_vol_by_sku['StickCons'])

projected_vol_by_sku['Proj_LANU_bySKU'] = round(projected_vol_by_sku['SoDom'] * projected_vol_by_sku['LANU'])



# compute projected volumes by dimension

projected_vol_by_prod_dim = projected_vol_by_sku.groupby(['IATA']+ dimension).agg(

    Proj_Vol_PG=('Proj_Vol_bySKU', np.sum), Proj_LANU_PG=('Proj_LANU_bySKU', np.sum)).reset_index()





proj_totVol = projected_vol_by_prod_dim.groupby(['IATA']).sum().reset_index().rename(columns={'Proj_Vol_PG':'Tot_proj_Vol'})

proj_SoM_PG = projected_vol_by_prod_dim.merge(proj_totVol[['IATA','Tot_proj_Vol']], how = 'left', on = ['IATA'])

proj_SoM_PG['Proj_SoM_PG'] =  proj_SoM_PG['Proj_Vol_PG'] / proj_SoM_PG['Tot_proj_Vol']

DFVol_IATA_bySKU = DF_Vols.groupby(['Year', 'IATA'] + dimension).sum('DF_Vol').reset_index()



Total_DFVol_byIATA = DFVol_IATA_bySKU.groupby(['Year','IATA']).sum().reset_index().rename(columns={'DF_Vol':'DFTot_Vol'})



DFSoM_IATA_bySKU = DFVol_IATA_bySKU.merge(Total_DFVol_byIATA[time_dim + ['IATA','DFTot_Vol']],

                                    how='left', on = time_dim + ['IATA'])



DFSoM_IATA_bySKU['DF_SoM_IATA_PG'] = DFSoM_IATA_bySKU['DF_Vol'] / DFSoM_IATA_bySKU['DFTot_Vol']



DFSoM_IATA_bySKU = DFSoM_IATA_bySKU[['IATA'] + dimension+ ['DF_Vol', 'DFTot_Vol', 'DF_SoM_IATA_PG']]

PARIS_output = proj_SoM_PG.merge(DFSoM_IATA_bySKU,how = 'outer',on = dimension + ['IATA']).fillna(0)



PARIS_output['DF_SoM_IATA_PG'] = PARIS_output['DF_SoM_IATA_PG'].fillna(0)

PARIS_output['Proj_SoM_PG'] = PARIS_output['Proj_SoM_PG'].fillna(0)

PARIS_output['Delta_SoS'] =  PARIS_output['Proj_SoM_PG'] - PARIS_output['DF_SoM_IATA_PG']


PARIS_output = proj_SoM_PG.merge(DFSoM_IATA_bySKU,how = 'outer',on = dimension + ['IATA']).fillna(0)



PARIS_output['DF_SoM_IATA_PG'] = PARIS_output['DF_SoM_IATA_PG'].fillna(0)

PARIS_output['Proj_SoM_PG'] = PARIS_output['Proj_SoM_PG'].fillna(0)

PARIS_output['Delta_SoS'] =  PARIS_output['Proj_SoM_PG'] - PARIS_output['DF_SoM_IATA_PG']



# PARIS_output['NLOV'] = np.where(PARIS_output['Delta_SoS'] > 0,

#                                 PARIS_output['Delta_SoS']*PARIS_output['DFTot_Vol'], 0.0)



# PARIS_output['NLOV'] = PARIS_output['NLOV'].fillna(0)



# PARIS_output['SCPI'] = np.where(PARIS_output['Proj_SoM_PG'] > 0.0001,

#                                 (PARIS_output['DF_SoM_IATA_PG'] / PARIS_output['Proj_SoM_PG'] - 1.0),0.0)





PARIS_output = PARIS_output[dimension+['IATA','DF_Vol','Proj_SoM_PG', 'DF_SoM_IATA_PG', 'Delta_SoS']]

PARIS_output = PARIS_output.merge(IATA_Location, how = 'left', on = 'IATA')

PARIS_output = PARIS_output[PARIS_output['Location'].notnull()]

PARIS_output = PARIS_output.rename(columns = {'DF_SoM_IATA_PG' : 'Real_So_Segment' ,'Proj_SoM_PG' : 'Ideal_So_Segment' })

PARIS_output = PARIS_output[PARIS_output['Ideal_So_Segment']>0.001]

PARIS_output = PARIS_output[['Location', 'IATA'] + brand_attributes + 

                            ['DF_Vol','Real_So_Segment','Ideal_So_Segment','Delta_SoS']]

DF_Vols[DF_Vols['Location'] == 'Jeju']

location = []

score = []



for i in PARIS_output['Location'].unique():

    looped_market = PARIS_output[PARIS_output['Location']== i]

    X, y = looped_market[["Real_So_Segment"]], looped_market[["Ideal_So_Segment"]]

    model.fit(X, y)

    r_squared = model.score(X, y)

    market_score = round(r_squared*10,2)

    

    

    location.append(i)

    score.append(market_score)

    # print(i)

list_of_tuples = list(zip(location, score))

cat_c_scores = pd.DataFrame(list_of_tuples,columns=['Location', 'RSQ']).fillna(0)

cat_c_scores[cat_c_scores['Location'] == 'Jeju']

similarity_file = pd.read_excel('datasource/matrix_end_2023.xlsx')

iata_location= pd.read_pickle('datasource/IATA_Location.pkl')

df_vols = pd.read_pickle('datasource/cat_a_df_vols.pkl')

base_list= pd.read_pickle('datasource/base_list.pkl')

similarity_file1 = pd.melt(similarity_file, id_vars =['IATA'], var_name = 'Cluster' , value_name = 'Score')

similarity_file1 = similarity_file1[similarity_file1['Score'] < 1]

similarity_file2 = similarity_file1.sort_values(['IATA','Score'], ascending = False)

similarity_file2['Rank'] = similarity_file2.groupby('IATA').rank(method = 'first', ascending = False)['Score']

clusters = similarity_file2[similarity_file2.Rank <=4 ]

duty_free_volumes = df_vols.copy()

duty_free_volumes['key'] = duty_free_volumes['CR_BrandId'].astype('str') + '-' + duty_free_volumes['Item per Bundle'].astype('str')

#All Universe

category_list = base_list.copy()

category_list['key'] =category_list['CR_BrandId'].astype('str') + '-' + category_list['Item per Bundle'].astype('str')

duty_free_volumes2 = duty_free_volumes[['DF_Market','Location','TMO','CR_BrandId','SKU',"{} Volume".format(current_year)]]

tobacco_range = category_list.merge(duty_free_volumes[['DF_Market', 'Location','key',"{} Volume".format(current_year)]], how = 'left', 

                                                    on = ['DF_Market', 'Location','key']).fillna(0)

tobacco_range['TMO'] = np.where(tobacco_range['TMO']!= 'PMI', 'Comp','PMI')

tobacco_range = tobacco_range[tobacco_range['CR_BrandId']!=0]

tobacco_range[tobacco_range['Location']=='Zurich'].head(3)

tobacco_range2 = tobacco_range[['DF_Market','Location','TMO','CR_BrandId','SKU',"{} Volume".format(current_year)]]

tobacco_range2 = pd.pivot_table(tobacco_range2, index = ['DF_Market','Location','TMO','CR_BrandId'],

               aggfunc={"{} Volume".format(current_year): np.sum, 'SKU': np.count_nonzero}).reset_index()

selma_df_products = SELMA_DF_map.copy() 

selma_df_products = selma_df_products[selma_df_products['Product Category'] == 'Cigarettes'].reset_index()

selma_df_products_2 = selma_df_products[['DF_Market','Location', 'CR_BrandId']+brand_attributes ]

selma_df_products_3 = selma_df_products_2.drop_duplicates()

selma_df_products_3 = selma_df_products_3[selma_df_products_3['CR_BrandId']!= 0]

selma_df_products_3 = selma_df_products_3[~selma_df_products_3['DF_Market'].isna()]

Market_Mix = selma_df_products_3.merge(tobacco_range2[['DF_Market', 'Location','CR_BrandId', 'TMO', 'SKU',

                                                    "{} Volume".format(current_year)]], how = 'left' , on = ['DF_Market', 'Location','CR_BrandId'])

Market_Mix = Market_Mix[Market_Mix['SKU'].notnull()]

Market_Mix = Market_Mix[Market_Mix['SKU']!=0]

Market_Mix = Market_Mix.merge(iata_location, how = 'left', on = 'Location')

Market_Mix[Market_Mix['Location'] == 'Zurich'].head(3)

# base_list.to_excel('baselist.xlsx', index = False)

all_market = pd.pivot_table(tobacco_range, index = ['DF_Market','Location','TMO'],  

                                                   aggfunc={ 'SKU': np.count_nonzero}).reset_index()

all_market = all_market.rename(columns = {'SKU': 'Total TMO'})

Market_Summary0 = pd.pivot_table(Market_Mix, index = ['DF_Market','IATA','Location', 'TMO']+brand_attributes,  

                                                   aggfunc={ "{} Volume".format(current_year): np.sum, 'SKU': np.sum}).reset_index()

Market_Summary = Market_Summary0.merge(all_market, how = 'left', on =['DF_Market','Location','TMO'] )

Market_Summary['SoM'] = round(Market_Summary['SKU']*100/Market_Summary['Total TMO'],1)

Market_Summary = Market_Summary[['DF_Market','IATA','Location','TMO']+ brand_attributes+['SKU','Total TMO', 'SoM']]

Market_Summary_PMI = Market_Summary[Market_Summary['TMO'] == 'PMI']

Market_Summary_PMI = Market_Summary_PMI[(Market_Summary_PMI['DF_Market'] != 'Spain DP') & (Market_Summary_PMI['DF_Market'] != 'France DP')]

Market_Summary_Comp = Market_Summary[Market_Summary['TMO'] == 'Comp']

Market_Summary_Comp = Market_Summary_Comp[(Market_Summary_Comp['DF_Market'] != 'Spain DP') & (Market_Summary_Comp['DF_Market'] != 'France DP')]



Market_Summary_PMI = Market_Summary_PMI.rename(columns= {'SoM':'SoM_PMI','SKU':'PMI_Seg_SKU', 'Total TMO': 'PMI Total'})

Market_Summary_Comp = Market_Summary_Comp.rename(columns= {'SoM':'SoM_Comp','SKU':'Comp_Seg_SKU', 'Total TMO': 'Comp Total'})

Market_Summary_Comp = Market_Summary_Comp[['DF_Market','IATA','Location']+ brand_attributes+ ['Comp_Seg_SKU','Comp Total','SoM_Comp']]

Market_Summary_PMI = Market_Summary_PMI[['DF_Market','IATA','Location']+ brand_attributes+['PMI_Seg_SKU','PMI Total','SoM_PMI']]

Market_Summary_Delta = Market_Summary_Comp.merge(Market_Summary_PMI[['DF_Market','IATA','Location','Flavor','Taste','Thickness',

                                                'Length','PMI_Seg_SKU','PMI Total','SoM_PMI']], how = 'outer', 

                                                 on = ['DF_Market','IATA','Location','Flavor','Taste','Thickness','Length']).fillna(0)

Market_Volume_Table = Market_Summary0.groupby(['DF_Market', 'Location']+ brand_attributes).sum("{} Volume".format(current_year)).reset_index()

Market = Market_Summary_Delta.merge(Market_Volume_Table[['DF_Market', 'Location']+ brand_attributes+ ["{} Volume".format(current_year)]] 

                                    ,how = 'left', on = ['DF_Market', 'Location']+ brand_attributes)

Market['SKU_Delta'] = Market['SoM_PMI'] - Market['SoM_Comp']

clusterlist = pd.DataFrame()



for iata in clusters.IATA.unique():



    a = Market_Summary_PMI[Market_Summary_PMI['IATA'] == iata ]

    a = a.drop(columns = ['SoM_PMI'])

    a = a.rename(columns = {'PMI_Seg_SKU': 'PMI SKU'})

    if len(a)== 0:

        pass

    else:

        selected_iata = clusters[clusters['IATA'] == iata]

        cluster_iata = list(selected_iata.Cluster.unique())

        b = Market_Summary_PMI[Market_Summary_PMI['IATA'].isin(cluster_iata)]

        b['IATA'] = iata

        b = b[['IATA'] + brand_attributes + ['PMI_Seg_SKU']]

        b = b.groupby(by =['IATA'] + brand_attributes).sum(['PMI_Seg_SKU']).reset_index()

        b = b.rename(columns = {'PMI_Seg_SKU':'Cluster Segment'})



        c = Market_Summary_Comp[Market_Summary_Comp['IATA'].isin(cluster_iata)]

        c['IATA'] = iata

        c = c[['IATA'] + brand_attributes + ['Comp_Seg_SKU']]

        c = c.groupby(by = ['IATA'] + brand_attributes).sum(['Comp_Seg_SKU']).reset_index()

        c = c.rename(columns = {'Comp_Seg_SKU':'Cluster Segment'})



        d = pd.concat([b,c], ignore_index = True) 

        d = d.groupby(by = ['IATA'] +  brand_attributes).sum(['Cluster Segment']).reset_index()

        d_x =  d.groupby(by = ['IATA']).sum('Cluster Segment').reset_index()

        d_x = d_x.rename(columns = {'Cluster Segment': 'Cluster_Total'})

        d = d.merge(d_x, how = 'left', on = 'IATA')



        e = a.merge(d, how = 'outer', on = ['IATA'] + brand_attributes).fillna(0)

        e['DF_Market'] = list(a.DF_Market.unique())[0]

        e['IATA'] = list(a.IATA.unique())[0]

        e['Location'] = list(a.Location.unique())[0]



        clusterlist = pd.concat([e,clusterlist])

        clusterlist['PMI SKU %'] = np.where(clusterlist['PMI Total']>0, clusterlist['PMI SKU'] / clusterlist['PMI Total'],0)

        clusterlist['Cluster SKU %'] = np.where(clusterlist['Cluster_Total']>0, clusterlist['Cluster Segment'] / clusterlist['Cluster_Total'],0)



        clusterlist['SKU Delta'] = clusterlist['PMI SKU %'] - clusterlist['Cluster SKU %']

clusterlist = clusterlist.rename(columns = {'Cluster Segment': 'Cluster SKU'})

clusterlist = clusterlist[['DF_Market', 'IATA', 'Location', 

                           'Taste', 'Thickness', 'Flavor','Length', 

                           'PMI SKU', 'PMI Total','PMI SKU %', 'Cluster SKU', 'Cluster_Total', 'Cluster SKU %']]



clusterlist.head(3)

location = []

score = []

num_of_pmi_sku = []

num_of_comp_sku = []

pmi_cot = []

comp_cot = []



for i in clusterlist['Location'].unique():

    looped_market = clusterlist[clusterlist['Location']== i]

    X, y = looped_market[["PMI SKU"]], looped_market[["Cluster SKU"]]

    model.fit(X, y)

    r_squared = model.score(X, y)

    market_score = round(r_squared*10,2)

    

    skunum =  max(Market[(Market['Location']== i)].iloc[:,-4],default=0)

    compsku =  max(Market[(Market['Location']== i)].iloc[:,-7],default=0)

#     pmi_vol = Market_Mix[(Market_Mix['Location']== i) & (Market_Mix['TMO']== 'PMI')].sum().iloc[-1].astype('int')

#     comp_vol = Market_Mix[(Market_Mix['Location']== i) & (Market_Mix['TMO']!= 'PMI')].sum().iloc[-1].astype('int')

    

    location.append(i)

    score.append(market_score)

    num_of_pmi_sku.append(skunum)

    num_of_comp_sku.append(compsku)

#     pmi_cot.append(pmi_vol)

#     comp_cot.append(comp_vol)

#     print(i)

# list_of_tuples = list(zip(location, score,num_of_pmi_sku,num_of_comp_sku,pmi_cot,comp_cot))

list_of_tuples = list(zip(location, score,num_of_pmi_sku,num_of_comp_sku))



# final_list = pd.DataFrame(list_of_tuples,columns=['Location', 'RSQ', 'NumPMI_SKU','NumComp_SKU','PMI Volume','Comp Volume']).fillna(0)

cat_d_scores = pd.DataFrame(list_of_tuples,columns=['Location', 'RSQ', 'NumPMI_SKU','NumComp_SKU']).fillna(0)

cat_d_scores

cat_a_scores = cat_a_scores.rename(columns= {'ScaledScore': 'Cat_A'})

cat_a_scores = cat_a_scores[['Location', 'Cat_A']]

cat_a_scores['Location'] = cat_a_scores['Location'].str.strip()

cat_b_scores = cat_b_scores.rename(columns= {'RSQ': 'Cat_B'})

cat_b_scores = cat_b_scores[['Location', 'Cat_B']]

cat_b_scores['Location'] = cat_b_scores['Location'].str.strip()

cat_c_scores = cat_c_scores.rename(columns= {'RSQ': 'Cat_C'})

cat_c_scores = cat_c_scores[['Location', 'Cat_C']]

cat_c_scores['Location'] = cat_c_scores['Location'].str.strip()

cat_d_scores = cat_d_scores.rename(columns= {'RSQ': 'Cat_D'})

cat_d_scores = cat_d_scores[['Location', 'Cat_D']]

cat_d_scores['Location'] = cat_d_scores['Location'].str.strip()

final_table_0 =  cat_a_scores.merge(cat_b_scores, how = 'left', on = 'Location')

final_table_1 = final_table_0.merge(cat_c_scores, how = 'left', on = 'Location')

final_table_2 = final_table_1.merge(cat_d_scores, how = 'left', on = 'Location')

final_table_3 = final_table_2.fillna(0)

final_table_3[final_table_3['Location'] == 'Zurich']

final_table_4 = final_table_3.copy()

cols_to_average = ['Cat_A','Cat_B','Cat_C','Cat_D']

final_table_4[cols_to_average] = final_table_4[cols_to_average].replace(10, pd.NA)

final_table_4['Avg_Score'] = final_table_4[cols_to_average].mean(axis=1, skipna = True)

final_table_4['Avg_Score'] = final_table_4['Avg_Score'].astype('float')

final_table_4 = final_table_4[final_table_4['Avg_Score']>0]

final_table_4[cols_to_average] = final_table_4[cols_to_average].replace(pd.NA,10)

final_table_4['Avg_Score'] = round(final_table_4['Avg_Score'],2)

final_table_4[final_table_4['Location'] == 'Jeju']

Location_Volumes = pd.pivot_table(cat_a_df_vols, index = ['Location'], aggfunc={ "{} Volume".format(current_year): np.sum}).reset_index()

PMI_Volumes_0 = cat_a_df_vols[cat_a_df_vols['TMO'] == 'PMI']

PMI_Volumes = pd.pivot_table(PMI_Volumes_0, index = ['Location'], aggfunc={ "{} Volume".format(current_year): np.sum}).reset_index()

Location_Volumes = Location_Volumes.rename(columns = {"{} Volume".format(current_year) : 'Market_Volume'})

PMI_Volumes = PMI_Volumes.rename(columns = {"{} Volume".format(current_year) : 'PMI_Volume'})

final_table_5 = final_table_4.merge(Location_Volumes[['Location', 'Market_Volume']], how = 'left', on = 'Location').fillna(0)

final_table_5 = final_table_5.merge(PMI_Volumes[['Location', 'PMI_Volume']], how = 'left', on = 'Location').fillna(0)

final_table_5

final_table_5.to_excel(r'Location_Scores_Outputs\{}scores_1.xlsx'.format(current_year), index = False)





