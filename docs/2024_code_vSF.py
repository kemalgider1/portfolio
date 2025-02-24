import pandas as pd
import pyodbc as msql
import numpy as np
# from pivottablejs import pivot_ui
import math
from sklearn.linear_model import LinearRegression
import snowflake.connector

model = LinearRegression()

conn2 = snowflake.connector.connect(
    user="KGIDERO@PMINTL.NET", #You can get it by executing in UI: desc user <username>;
    account="pl47603.eu-west-1", #Add all of the account-name between https:// and snowflakecomputing.com
    authenticator="externalbrowser",
    warehouse='WH_PRD_REPORTING',
    role='PMI_EDP_SPK_SNFK_PMI_FDF_DEV_DATAENGINEER_IMDL',
    database = "DB_FDF_PRD"
)

#dynamic variables to use as parameter
time_dim = ['Year']
brand_attributes = ['Taste','Thickness','Flavor','Length'] 
IATA_List = [] # Removed specific IATA
dimension = ['Flavor','Taste','Thickness','Length'] 
domestic_dimensions = ['Market','EBROMId','EBROM','Taste','Thickness', 'Flavor','Length']
current_year = 2024
previous_year = current_year-1
theyearbefore = previous_year-1

# Load from CSV files using the query names
MC_per_Product = pd.read_csv('MC_per_Product_SQL.csv')
cat_a_df_vols = pd.read_csv('df_vols_query.csv')
#SELMA files
SELMA_DF_map = pd.read_csv('SELMA_DF_map_query.csv')
base_list = pd.read_csv('base_query.csv')
#Domestic files
Pmidf_products = pd.read_csv('sql_PMIDF_script.csv')
dom_prods_data = pd.read_csv('sql_dom_script.csv')
dom_ims_data = pd.read_csv('sql_Domest_script.csv')
DomesticVolumes = pd.read_csv('DomesticVolumes.csv')
#Lanu Files
CF_data = pd.read_csv('country_figures.csv')
DF_Vol_data = pd.read_csv('COT_query.csv')
PAX_data = pd.read_csv('Pax_Nat.csv')
#map files
mrk_nat_map = pd.read_csv('mrk_nat_map_query.csv')
IATA_Location = pd.read_csv('iata_location_query.csv')
SELMA_dom_map = pd.read_csv('SELMA_dom_map_query.csv')


DF_Vols_w_Financials = cat_a_df_vols.merge(MC_per_Product[['DF_Market','Location','CR_BrandId',
                                            "{} MC".format(previous_year),
                                            "{} NOR".format(previous_year),"{} MC".format(current_year),"{} NOR".format(current_year)]], 
                                           how = 'left', on = ['DF_Market','Location','CR_BrandId']).fillna(0)
DF_Vols_w_Financials = DF_Vols_w_Financials[DF_Vols_w_Financials["{} Volume".format(current_year)]>0]

DF_Vols_w_Financials['LYRevenueAvg'] = np.where(DF_Vols_w_Financials["{}Month".format(previous_year)] == 0, 0, 
                            round(DF_Vols_w_Financials["{} Revenue".format(previous_year)] / DF_Vols_w_Financials["{}Month".format(previous_year)],0))
DF_Vols_w_Financials['CYRevenueAvg'] = np.where(DF_Vols_w_Financials["{}Month".format(current_year)] == 0, 0, 
                                         round(DF_Vols_w_Financials["{} Revenue".format(current_year)] / DF_Vols_w_Financials["{}Month".format(current_year)],0))


DF_Vols_w_Financials['Growth'] = (DF_Vols_w_Financials['CYRevenueAvg']-DF_Vols_w_Financials['LYRevenueAvg']) / DF_Vols_w_Financials['LYRevenueAvg']

DF_Vols_w_Financials['Margin'] = np.where(
                                        DF_Vols_w_Financials["{} NOR".format(current_year)]<=0,0,
                                        ((DF_Vols_w_Financials["{} MC".format(current_year)] /DF_Vols_w_Financials["{}Month".format(current_year)]) /  
                                        (DF_Vols_w_Financials["{} NOR".format(current_year)] / DF_Vols_w_Financials["{}Month".format(current_year)])))
pmi_margins = DF_Vols_w_Financials[DF_Vols_w_Financials['TMO'] == 'PMI']

#Brand Family Margins

pmi_margins['Margin_Volume'] = round((pmi_margins["{} Volume".format(current_year)] * pmi_margins['Margin'].fillna(0)),0).astype(int)
pmi_margins = pmi_margins.groupby(['DF_Market','Location','Brand Family']).sum().reset_index()
pmi_margins = pmi_margins[['DF_Market','Location','Brand Family',
                          "{} Volume".format(current_year),"{} MC".format(current_year),'Margin_Volume']]
pmi_margins['Brand Family Margin'] = pmi_margins['Margin_Volume'] / pmi_margins["{} Volume".format(current_year)]

# Merging Brand Family Margins to Original Table
SKU_by_Vols_Margins =  DF_Vols_w_Financials.merge(pmi_margins[['DF_Market','Location','Brand Family','Brand Family Margin']],
                                                  how = 'left', on =['DF_Market','Location','Brand Family']).fillna(0)
SKU_by_Vols_Margins['Margin Comparison'] = np.where(
                                        SKU_by_Vols_Margins['Brand Family Margin']< SKU_by_Vols_Margins['Margin'], 1 , 0)
# SKU_by_Vols_Margins = SKU_by_Vols_Margins[SKU_by_Vols_Margins['TMO'] == 'PMI']

# Calculating the number of SKUs for Green & Red Flag SKUs
no_of_sku = DF_Vols_w_Financials.groupby(['DF_Market', 'Location'])['SKU'].count().reset_index()
no_of_sku = no_of_sku.rename(columns = {'SKU': 'TotalSKU'})
no_of_sku['GreenFlagSKU'] = (no_of_sku['TotalSKU']*0.05).apply(np.ceil)
no_of_sku['RedFlagSKU'] = round(no_of_sku['TotalSKU']*0.25 , 0)

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

# combining Rule 1 and Rule 2 : Final Green Flag SKUs
green_list = pd.concat([gf,gf2])
green_list = green_list[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle']]
green_list = green_list.drop_duplicates()
green_list['Green']  = 1

# Finding the Red Flag SKUs
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

rf2 = pd.DataFrame(columns=['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle'])

for i in DF_Vols_w_Financials.Location.unique():
    
    red_flag2_1 = SKU_by_Vols_Margins[SKU_by_Vols_Margins['Location'] ==  i]
    red_flag2_1 = red_flag2_1.sort_values('Growth', ascending = True).head(no_of_sku[no_of_sku['Location'] == i].iloc[0,-1].astype('int'))
    red_flag2_1 = red_flag2_1[red_flag2_1['TMO'] == 'PMI']
    red_flag2_1 = red_flag2_1[red_flag2_1['Margin Comparison'] == 0]
    red_flag2_1 = red_flag2_1[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle']]
    rf2 = pd.concat([rf2,red_flag2_1 ], ignore_index = True)

red_list = pd.concat([rf1, rf2], ignore_index = True).drop_duplicates()
red_list['Red'] = 1

green_red_list = green_list.merge(red_list, how = 'outer', on = ['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle']).fillna(0)
green_red_list['Check'] = np.where(green_red_list['Green'] != green_red_list['Red'], 'OK','Problem')
green_red_list = green_red_list[green_red_list['Check'] != 'Problem']
green_red_list['Status'] = np.where(green_red_list['Green'] == 1 , 'Green', 'Red')

category_a_0 = SKU_by_Vols_Margins[SKU_by_Vols_Margins['TMO'] == 'PMI']

category_a_1 = (category_a_0.merge(
    green_red_list[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle','Status']], 
    how = 'left',on = ['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU','Item per Bundle'])).fillna(0)


location = []
score_a = []
for i in category_a_1.Location.unique() :
    
    ct = category_a_1[category_a_1['Location'] == i]
    
    score = ((ct['Green'].fillna(0).sum() -(ct['Red'].fillna(0).sum() *2 )) / ct['SKU'].count())*100
    location.append(i)
    score_a.append(score)

list_of_tuples = list(zip(location, score_a))
cat_a_scores = pd.DataFrame(list_of_tuples,columns=['Location','Score_A'] ).fillna(0)
cat_a_scores['ScaledScore'] = round((cat_a_scores['Score_A']-(-200))*(10/300),2)


#cluster list building
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

location = []
score = []
num_of_pmi_sku = []
num_of_comp_sku = []
pmi_cot = []
comp_cot = []

for i in clusterlist['Location'].unique():
    looped_market = clusterlist[clusterlist['Location']== i]
    X, y = looped_market[[\"PMI SKU\"]], looped_market[[\"Cluster SKU\"]]
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

final_table_4 = final_table_3.copy()

cols_to_average = ['Cat_A','Cat_B','Cat_C','Cat_D']
final_table_4[cols_to_average] = final_table_4[cols_to_average].replace(10, pd.NA)

final_table_4['Avg_Score'] = final_table_4[cols_to_average].mean(axis=1, skipna = True)
final_table_4['Avg_Score'] = final_table_4['Avg_Score'].astype('float')
final_table_4 = final_table_4[final_table_4['Avg_Score']>0]
final_table_4[cols_to_average] = final_table_4[cols_to_average].replace(pd.NA,10)
final_table_4['Avg_Score'] = round(final_table_4['Avg_Score'],2)

Location_Volumes = pd.pivot_table(cat_a_df_vols, index = ['Location'], aggfunc={ "{} Volume".format(current_year): np.sum}).reset_index()
PMI_Volumes_0 = cat_a_df_vols[cat_a_df_vols['TMO'] == 'PMI']
PMI_Volumes = pd.pivot_table(PMI_Volumes_0, index = ['Location'], aggfunc={ "{} Volume".format(current_year): np.sum}).reset_index()
Location_Volumes = Location_Volumes.rename(columns = {"{} Volume".format(current_year) : 'Market_Volume'})
PMI_Volumes = PMI_Volumes.rename(columns = {"{} Volume".format(current_year) : 'PMI_Volume'})
final_table_5 = final_table_4.merge(Location_Volumes[['Location', 'Market_Volume']], how = 'left', on = 'Location').fillna(0)
final_table_5 = final_table_5.merge(PMI_Volumes[['Location', 'PMI_Volume']], how = 'left', on = 'Location').fillna(0)


final_table_5.to_excel(r'output\\{}scores_5.xlsx'.format(current_year), index = False)
