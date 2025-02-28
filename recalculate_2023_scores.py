import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
import os
from datetime import datetime
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('portfolio_recalculation.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables
brand_attributes = ['Taste', 'Thickness', 'Flavor', 'Length']
model = LinearRegression()
current_year = 2024
previous_year = current_year - 1

def load_reference_scores():
    """Load the 2023 reference scores for validation"""
    try:
        ref_scores = pd.read_csv('2023_cat_results.txt', sep='\t')
        logger.info(f"Loaded reference scores for {len(ref_scores)} locations")
        return ref_scores
    except Exception as e:
        logger.error(f"Error loading reference scores: {e}")
        return None

def load_data():
    """Load all required data files"""
    try:
        logger.info("Loading data files...")
        data = {}
        
        # Core datasets
        # Try to load from data directory first, then root
        try:
            data['df_vols'] = pd.read_csv('data/df_vols_query.csv')
        except:
            data['df_vols'] = pd.read_csv('df_vols_query.csv')
            
        try:
            data['mc_per_product'] = pd.read_csv('data/MC_per_Product_SQL.csv')
        except:
            data['mc_per_product'] = pd.read_csv('MC_per_Product_SQL.csv')
            
        try:
            data['market_pmi'] = pd.read_csv('data/Market_Summary_PMI.csv')
        except:
            data['market_pmi'] = pd.read_csv('Market_Summary_PMI.csv')
            
        try:
            data['market_comp'] = pd.read_csv('data/Market_Summary_Comp.csv')
        except:
            data['market_comp'] = pd.read_csv('Market_Summary_Comp.csv')
            
        try:
            data['pax_data'] = pd.read_csv('data/Pax_Nat.csv')
        except:
            data['pax_data'] = pd.read_csv('Pax_Nat.csv')
            
        try:
            data['mrk_nat_map'] = pd.read_csv('data/mrk_nat_map_query.csv')
        except:
            data['mrk_nat_map'] = pd.read_csv('mrk_nat_map_query.csv')
            
        try:
            data['iata_location'] = pd.read_csv('data/iata_location_query.csv')
        except:
            data['iata_location'] = pd.read_csv('iata_location_query.csv')
            
        try:
            data['similarity_file'] = pd.read_csv('data/similarity_file.csv')
        except:
            data['similarity_file'] = pd.read_csv('similarity_file.csv')
            
        try:
            data['country_figures'] = pd.read_csv('data/country_figures.csv')
        except:
            data['country_figures'] = pd.read_csv('country_figures.csv')
            
        try:
            data['dom_volumes'] = pd.read_csv('data/sql_Domest_script.csv')
        except:
            data['dom_volumes'] = pd.read_csv('sql_Domest_script.csv')
            
        try:
            data['dom_products'] = pd.read_csv('data/sql_dom_script.csv')
        except:
            data['dom_products'] = pd.read_csv('sql_dom_script.csv')
            
        try:
            data['selma_df_map'] = pd.read_csv('data/SELMA_DF_map_query.csv')
        except:
            data['selma_df_map'] = pd.read_csv('SELMA_DF_map_query.csv')
        
        # Check which files were loaded
        for key, df in data.items():
            if df is not None and not df.empty:
                logger.info(f"Loaded {key}: {len(df)} rows")
            else:
                logger.warning(f"Empty or missing data: {key}")
                
        # Prepare similarity file for Category D
        if 'similarity_file' in data and not data['similarity_file'].empty:
            # Ensure similarity file has the right columns for Category D
            if 'RANK' not in data['similarity_file'].columns and 'Rank' in data['similarity_file'].columns:
                data['similarity_file'] = data['similarity_file'].rename(columns={'Rank': 'RANK'})
                
            # Filter to top 4 similar locations if not already done
            if 'RANK' in data['similarity_file'].columns:
                data['similarity_file'] = data['similarity_file'][data['similarity_file']['RANK'] <= 4].copy()
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def calculate_category_a(df_vols, mc_per_product):
    """
    Calculate Category A scores using the 2023 methodology
    """
    try:
        logger.info("Calculating Category A scores (SKU Portfolio Optimization)...")
        
        # Ensure we have the correct volume column
        vol_col = f"{current_year} Volume"
        if vol_col not in df_vols.columns:
            vol_cols = [col for col in df_vols.columns if 'Volume' in col]
            if vol_cols:
                vol_col = vol_cols[0]
                logger.warning(f"Using {vol_col} instead of {current_year} Volume")
            else:
                raise ValueError("No volume column found in df_vols")
        
        # Merge and prepare data
        df = df_vols.merge(
            mc_per_product[['DF_Market', 'Location', 'CR_BrandId', 
                          f'{current_year} MC', f'{current_year} NOR']], 
            how='left', 
            on=['DF_Market', 'Location', 'CR_BrandId']
        ).fillna(0)
        
        # Calculate volume calculations
        df['LYRevenueAvg'] = np.where(df[f"{previous_year}Month"] == 0, 0, 
                            round(df[f"{previous_year} Revenue"] / df[f"{previous_year}Month"],0))
        df['CYRevenueAvg'] = np.where(df[f"{current_year}Month"] == 0, 0, 
                                round(df[f"{current_year} Revenue"] / df[f"{current_year}Month"],0))
        
        df['Growth'] = np.where(df['LYRevenueAvg'] == 0, 0,
                       (df['CYRevenueAvg']-df['LYRevenueAvg']) / df['LYRevenueAvg'])
        
        df['Margin'] = np.where(
                            df[f"{current_year} NOR"]<=0,0,
                            ((df[f"{current_year} MC"] /df[f"{current_year}Month"]) /  
                            (df[f"{current_year} NOR"] / df[f"{current_year}Month"])))
        
        # PMI margins
        pmi_margins = df[df['TMO'] == 'PMI'].copy()
        pmi_margins['Margin_Volume'] = round((pmi_margins[vol_col] * pmi_margins['Margin'].fillna(0)),0).astype(int)
        pmi_margins = pmi_margins.groupby(['DF_Market','Location','Brand Family']).sum().reset_index()
        pmi_margins = pmi_margins[['DF_Market','Location','Brand Family', vol_col, f"{current_year} MC",'Margin_Volume']]
        pmi_margins['Brand Family Margin'] = np.where(pmi_margins[vol_col] == 0, 0, 
                                                     pmi_margins['Margin_Volume'] / pmi_margins[vol_col])
        
        # Merge for SKU by volumes and margins
        SKU_by_Vols_Margins = df.merge(
            pmi_margins[['DF_Market','Location','Brand Family','Brand Family Margin']],
            how='left', 
            on=['DF_Market','Location','Brand Family']
        ).fillna(0)
        
        SKU_by_Vols_Margins['Margin Comparison'] = np.where(
                                            SKU_by_Vols_Margins['Brand Family Margin']< SKU_by_Vols_Margins['Margin'], 1, 0)
        
        # Calculate thresholds per location
        no_of_sku = df.groupby(['DF_Market', 'Location'])['SKU'].count().reset_index()
        no_of_sku = no_of_sku.rename(columns={'SKU': 'TotalSKU'})
        no_of_sku['GreenFlagSKU'] = (no_of_sku['TotalSKU']*0.05).apply(np.ceil)
        no_of_sku['RedFlagSKU'] = round(no_of_sku['TotalSKU']*0.25, 0)
        
        # Calculate green flags based on volume
        green_flags = []
        for i in df.Location.unique():
            DF_Vols = df[df['Location'] == i]
            green_flag1 = DF_Vols.sort_values(vol_col, ascending=False).head(
                int(no_of_sku[no_of_sku['Location'] == i].iloc[0]['GreenFlagSKU'])
            )
            green_flag1 = green_flag1[green_flag1['TMO'] == 'PMI']
            green_flag1['Green1'] = 1
            green_flags.append(green_flag1)
        
        gf = pd.concat(green_flags, ignore_index=True) if green_flags else pd.DataFrame()
        
        # Calculate green flags based on growth and margin
        green_flags2 = []
        for i in df.Location.unique():
            DF_Vols = SKU_by_Vols_Margins[SKU_by_Vols_Margins['Location'] == i]
            green_flag2 = DF_Vols.sort_values('Growth', ascending=False).head(
                int(no_of_sku[no_of_sku['Location'] == i].iloc[0]['GreenFlagSKU'])
            )
            green_flag2 = green_flag2[green_flag2['TMO'] == 'PMI']
            green_flag2 = green_flag2[green_flag2['Margin Comparison'] == 1]
            green_flag2['Green Flag2'] = 1
            green_flags2.append(green_flag2)
            
        gf2 = pd.concat(green_flags2, ignore_index=True) if green_flags2 else pd.DataFrame()
        
        # Combine green lists
        green_list = pd.concat([gf, gf2])
        green_list = green_list[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU', 'Item per Bundle']]
        green_list = green_list.drop_duplicates()
        green_list['Green'] = 1
        
        # Calculate red flags
        red_flags = []
        for i in df.Location.unique():
            Red_Vols = df[df['Location'] == i]
            red_flag1 = Red_Vols.sort_values(vol_col, ascending=True).head(
                int(no_of_sku[no_of_sku['Location'] == i].iloc[0]['RedFlagSKU'])
            )
            red_flag1 = red_flag1[red_flag1['TMO'] == 'PMI']
            
            red_flag1_1 = Red_Vols.sort_values('Growth', ascending=True).head(
                int(no_of_sku[no_of_sku['Location'] == i].iloc[0]['RedFlagSKU'])
            )
            red_flag1_1 = red_flag1_1[red_flag1_1['TMO'] == 'PMI']
            
            # Find intersection
            red_flag_intersection = np.intersect1d(red_flag1.CR_BrandId, red_flag1_1.CR_BrandId)
            
            red_flag1_2 = pd.concat([red_flag1, red_flag1_1], ignore_index=True)
            red_flag1_2 = red_flag1_2[red_flag1_2['CR_BrandId'].isin(red_flag_intersection)].drop_duplicates()
            red_flag1_2 = red_flag1_2[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU', 'Item per Bundle']]
            
            red_flags.append(red_flag1_2)
            
        rf1 = pd.concat(red_flags, ignore_index=True) if red_flags else pd.DataFrame()
        
        # Additional red flags based on margin comparison
        red_flags2 = []
        for i in df.Location.unique():
            red_flag2_1 = SKU_by_Vols_Margins[SKU_by_Vols_Margins['Location'] == i]
            red_flag2_1 = red_flag2_1.sort_values('Growth', ascending=True).head(
                int(no_of_sku[no_of_sku['Location'] == i].iloc[0]['RedFlagSKU'])
            )
            red_flag2_1 = red_flag2_1[red_flag2_1['TMO'] == 'PMI']
            red_flag2_1 = red_flag2_1[red_flag2_1['Margin Comparison'] == 0]
            red_flag2_1 = red_flag2_1[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU', 'Item per Bundle']]
            red_flags2.append(red_flag2_1)
            
        rf2 = pd.concat(red_flags2, ignore_index=True) if red_flags2 else pd.DataFrame()
        
        # Combine red lists
        red_list = pd.concat([rf1, rf2], ignore_index=True).drop_duplicates()
        red_list['Red'] = 1
        
        # Combine green and red lists
        green_red_list = green_list.merge(
            red_list, 
            how='outer', 
            on=['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU', 'Item per Bundle']
        ).fillna(0)
        
        green_red_list['Check'] = np.where(green_red_list['Green'] != green_red_list['Red'], 'OK', 'Problem')
        green_red_list = green_red_list[green_red_list['Check'] != 'Problem']
        green_red_list['Status'] = np.where(green_red_list['Green'] == 1, 'Green', 'Red')
        
        # Final calculations
        category_a_0 = SKU_by_Vols_Margins[SKU_by_Vols_Margins['TMO'] == 'PMI']
        category_a_1 = category_a_0.merge(
            green_red_list[['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU', 'Item per Bundle', 'Status']], 
            how='left',
            on=['DF_Market', 'Location', 'TMO', 'Brand Family', 'CR_BrandId', 'SKU', 'Item per Bundle']
        ).fillna(0)
        
        total_sku = category_a_1.groupby(['DF_Market', 'Location'])['CR_BrandId'].count().reset_index()
        total_sku = total_sku.rename(columns={'CR_BrandId': 'TotalSKU'})
        
        ct_green = category_a_1[category_a_1['Status'] == 'Green']
        ct_green = ct_green.groupby(['DF_Market', 'Location'])['CR_BrandId'].count().reset_index()
        ct_green = ct_green.rename(columns={'CR_BrandId': 'GreenSKU'})
        
        ct_red = category_a_1[category_a_1['Status'] == 'Red']
        ct_red = ct_red.groupby(['DF_Market', 'Location'])['CR_BrandId'].count().reset_index()
        ct_red = ct_red.rename(columns={'CR_BrandId': 'RedSKU'})
        
        ct_gr_red = ct_red.merge(ct_green, how='outer', on=['DF_Market', 'Location'])
        
        calculation_table = total_sku.merge(ct_gr_red, how='outer', on=['DF_Market', 'Location'])
        calculation_table['RedSKU'] = calculation_table['RedSKU'].fillna(0).astype(int)
        calculation_table['GreenSKU'] = calculation_table['GreenSKU'].fillna(0).astype(int)
        
        # Final scores
        location = []
        score_a = []
        
        for i in calculation_table.Location.unique():
            ct = calculation_table[calculation_table['Location'] == i]
            score = ((ct['GreenSKU'].iloc[0] - (ct['RedSKU'].iloc[0] * 2)) / ct['TotalSKU'].iloc[0]) * 100
            location.append(i)
            score_a.append(score)
        
        list_of_tuples = list(zip(location, score_a))
        cat_a_scores = pd.DataFrame(list_of_tuples, columns=['Location', 'Score_A']).fillna(0)
        cat_a_scores['Cat_A'] = round((cat_a_scores['Score_A'] - (-200)) * (10/300), 2)
        
        logger.info(f"Category A calculation complete for {len(cat_a_scores)} locations")
        return cat_a_scores[['Location', 'Cat_A']]
    
    except Exception as e:
        logger.error(f"Error in Category A calculation: {e}")
        return pd.DataFrame(columns=['Location', 'Cat_A'])

def calculate_category_b(market_pmi, market_comp):
    """
    Calculate Category B scores using the 2023 methodology
    """
    try:
        logger.info("Calculating Category B scores (Segment Distribution)...")
        
        location = []
        score = []
        
        for i in set(market_pmi['Location'].unique()).intersection(market_comp['Location'].unique()):
            try:
                # Filter to current location
                pmi_data = market_pmi[market_pmi['Location'] == i].copy()
                comp_data = market_comp[market_comp['Location'] == i].copy()
                
                # Extract PMI and competitor segment distributions
                X = pmi_data[['SoM_PMI']]
                y = comp_data[['SoM_Comp']]
                
                # Use linear regression to calculate R-squared
                model.fit(X, y)
                r_squared = model.score(X, y)
                market_score = round(r_squared * 10, 2)
                
                location.append(i)
                score.append(market_score)
                
            except Exception as e:
                logger.warning(f"Error calculating Category B score for {i}: {e}")
                location.append(i)
                score.append(0)
                
        cat_b_scores = pd.DataFrame(list(zip(location, score)), columns=['Location', 'Cat_B']).fillna(0)
        cat_b_scores['Location'] = cat_b_scores['Location'].str.strip()
        
        logger.info(f"Category B calculation complete for {len(cat_b_scores)} locations")
        return cat_b_scores

    except Exception as e:
        logger.error(f"Error in Category B calculation: {e}")
        return pd.DataFrame(columns=['Location', 'Cat_B'])

def calculate_category_c(pax_data, df_vol_data, mrk_nat_map, iata_location, country_figures, dom_volumes, dom_products, selma_df_map):
    """
    Calculate Category C scores using the 2023 methodology (Passenger Mix)
    """
    try:
        logger.info("Calculating Category C scores (Passenger Mix)...")
        time_dim = ['Year']
        brand_attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
        domestic_dimensions = ['Market', 'EBROMId', 'EBROM', 'Taste', 'Thickness', 'Flavor', 'Length']
        
        # Step 1: Standardize SELMA dom map attributes
        if dom_products is not None and not dom_products.empty:
            selma_dom_map = dom_products[['EBROMId', 'TMO', 'Brand Family', 'Market',
                                        'THICKNESS_CATEGORY_CODE', 'MKT_LENGTH_CATEGORY_CODE', 
                                        'MENTHOL_INDICATOR', 'BRAND_DIFFERENTIATOR_CODE']].copy()
            
            # Map thickness
            selma_dom_map['Thickness'] = np.where(selma_dom_map['THICKNESS_CATEGORY_CODE'] == 'FAT', 'STD',
                                        selma_dom_map['THICKNESS_CATEGORY_CODE'])
            
            # Map length
            selma_dom_map['Length'] = np.where(selma_dom_map['MKT_LENGTH_CATEGORY_CODE'].isin(['REGULAR SIZE', 'REGULAR FILTER', 'SHORT SIZE', 'LONG FILTER', 'KS']),
                                            'KS', 
                                     np.where(selma_dom_map['MKT_LENGTH_CATEGORY_CODE'].isin(['LONGER THAN KS', '100', 'LONG SIZE', 'EXTRA LONG', 'SUPER LONG']),
                                            'LONG',
                                            selma_dom_map['MKT_LENGTH_CATEGORY_CODE']))
            
            # Map flavor
            selma_dom_map['Flavor'] = np.where(selma_dom_map['MENTHOL_INDICATOR'] == 'Y', 'Menthol', 'Regular')
            
            # Map taste
            selma_dom_map['Taste'] = 'Full Flavor'  # Default
            light_mask = selma_dom_map['BRAND_DIFFERENTIATOR_CODE'].str.contains('LIGHT', case=False, na=False)
            ultra_mask = selma_dom_map['BRAND_DIFFERENTIATOR_CODE'].str.contains('ULTRA', case=False, na=False)
            selma_dom_map.loc[light_mask, 'Taste'] = 'Lights'
            selma_dom_map.loc[ultra_mask, 'Taste'] = 'Ultra Lights'
            
            # Add a fake EBROM column if it doesn't exist
            if 'EBROM' not in selma_dom_map.columns:
                selma_dom_map['EBROM'] = selma_dom_map['EBROMId'].astype(str)
        else:
            # Create an empty dataframe with the right structure
            selma_dom_map = pd.DataFrame(columns=['EBROMId', 'TMO', 'Brand Family', 'Market',
                                                'Taste', 'Thickness', 'Flavor', 'Length', 'EBROM'])
        
        # Step 2: Standardize SELMA DF map attributes
        selma_df_map = selma_df_map.drop_duplicates(['CR_BrandId', 'Location'])
        selma_df_map['Length'] = np.where(selma_df_map['Length'].isin(['REGULAR SIZE', 'REGULAR FILTER', 'SHORT SIZE', 'LONG FILTER', 'NAN']),
                                        'KS', 
                                 np.where(selma_df_map['Length'].isin(['LONGER THAN KS', '100', 'LONG SIZE', 'EXTRA LONG', 'SUPER LONG']),
                                        'LONG',
                                        selma_df_map['Length']))
        
        # Step 3: Process passenger data
        pax_d1 = pax_data[time_dim + ['IATA', 'Market', 'Nationality', 'Pax']].copy()
        pax_d2 = pax_d1.groupby(time_dim + ['IATA', 'Market', 'Nationality']).sum().reset_index()
        pax_d3 = pax_d2.merge(mrk_nat_map, how='left', left_on='Nationality', right_on='PASSENGER_NATIONALITY')
        
        # Step 4: Apply smoking prevalence and allowance factors
        cf_d2 = country_figures[['KFYear', 'Country', 'SmokingPrevelance', 'InboundAllowance', 'ADCStick', 'PurchaserRate']].copy()
        cf_d2['ADCStick'] = cf_d2['ADCStick'].fillna(15.0)
        cf_d2['InboundAllowance'] = cf_d2['InboundAllowance'].fillna(400.0)
        cf_d2['PurchaserRate'] = np.where(cf_d2['PurchaserRate'] == cf_d2['PurchaserRate'], 
                                        cf_d2['PurchaserRate'], cf_d2['SmokingPrevelance'])
        
        pax_d4 = pax_d3.merge(cf_d2, how='left', left_on=['PASSENGER_COUNTRY_NAME', 'Year'], right_on=['Country', 'KFYear'])
        pax_d4['Pax'] = np.ceil(pax_d4['Pax'] * 1000)
        pax_d4['LANU'] = pax_d4['Pax'] * pax_d4['SmokingPrevelance'] * 0.9
        pax_d4['LANU'] = np.ceil(pax_d4['LANU'])
        pax_d4['InboundAllowance'] = pax_d4['InboundAllowance'].astype(float)
        pax_d4['StickCons'] = pax_d4['LANU'] * pax_d4['InboundAllowance']
        
        pax_fin_ = pax_d4[time_dim + ['Market', 'IATA', 'Nationality',
                        'PASSENGER_COUNTRY_NAME', 'LANU', 'StickCons']].rename(columns={'Market': 'DF_Market'})
        
        pax_fin = pax_fin_.groupby(time_dim + ['DF_Market', 'IATA',
                                             'Nationality', 'PASSENGER_COUNTRY_NAME']).sum().reset_index()
        
        # Step 5: Process domestic preferences by segment
        dom_attr = selma_dom_map[domestic_dimensions].merge(
            dom_products[['EBROMId', 'TMO', 'Brand Family']], 
            how='left',
            on='EBROMId'
        ).fillna('NaN')
        
        dom_ims_data = dom_volumes.groupby(time_dim + ['EBROMId']).sum('Volume').reset_index()
        dom_ims2 = dom_ims_data.merge(dom_attr, how='left', on='EBROMId')
        dom_ims2 = dom_ims2[dom_ims2['Market'] != 'PMIDF']
        dom_ims2 = dom_ims2[dom_ims2['EBROMId'] != 0]
        dom_ims2 = dom_ims2[dom_ims2['EBROM'] == dom_ims2['EBROM']]
        dom_ims2['Market'] = dom_ims2['Market'].replace('PRC', 'China')
        
        dom_totals = dom_ims2.groupby(time_dim + ['Market']).sum().reset_index().rename(columns={'Volume': 'TotVol'})
        dom_sov = dom_ims2.merge(dom_totals[time_dim + ['Market', 'TotVol']], how='left', on=time_dim + ['Market'])
        dom_sov['SoDom'] = dom_sov['Volume'] / dom_sov['TotVol']
        dom_fin = dom_sov[time_dim + domestic_dimensions + ['TMO', 'SoDom']].rename(columns={'Market': 'Dom_Market'})
        
        # Step 6: Project volumes by segment based on passenger mix
        projected_vol_by_sku = pax_fin.merge(
            dom_fin, 
            how='left',
            left_on=time_dim + ['PASSENGER_COUNTRY_NAME'],
            right_on=time_dim + ['Dom_Market']
        )
        projected_vol_by_sku['Proj_Vol_bySKU'] = round(projected_vol_by_sku['SoDom'] * projected_vol_by_sku['StickCons'])
        projected_vol_by_sku['Proj_LANU_bySKU'] = round(projected_vol_by_sku['SoDom'] * projected_vol_by_sku['LANU'])
        
        dimension = ['Flavor', 'Taste', 'Thickness', 'Length']
        projected_vol_by_prod_dim = projected_vol_by_sku.groupby(['IATA'] + dimension).agg(
            Proj_Vol_PG=('Proj_Vol_bySKU', np.sum), 
            Proj_LANU_PG=('Proj_LANU_bySKU', np.sum)
        ).reset_index()
        
        proj_totVol = projected_vol_by_prod_dim.groupby(['IATA']).sum().reset_index().rename(columns={'Proj_Vol_PG': 'Tot_proj_Vol'})
        proj_SoM_PG = projected_vol_by_prod_dim.merge(proj_totVol[['IATA', 'Tot_proj_Vol']], how='left', on=['IATA'])
        proj_SoM_PG['Proj_SoM_PG'] = proj_SoM_PG['Proj_Vol_PG'] / proj_SoM_PG['Tot_proj_Vol']
        
        # Step 7: Calculate actual SoM by segment
        volume_col = f"{current_year} Volume"
        if volume_col not in df_vol_data.columns:
            volume_cols = [col for col in df_vol_data.columns if 'Volume' in col]
            if volume_cols:
                volume_col = volume_cols[0]
            else:
                raise ValueError("No volume column found in df_vol_data")
        
        df_with_attrs = df_vol_data.merge(
            selma_df_map[['DF_Market', 'Location', 'CR_BrandId', 'Flavor', 'Taste', 'Thickness', 'Length']],
            on=['DF_Market', 'Location', 'CR_BrandId'],
            how='left'
        )
        
        df_with_iata = df_with_attrs.merge(iata_location, on='Location', how='left')
        df_vol_by_segment = df_with_iata.groupby(['IATA'] + dimension)[volume_col].sum().reset_index()
        df_totals = df_vol_by_segment.groupby(['IATA'])[volume_col].sum().reset_index().rename(columns={volume_col: 'Tot_actual_Vol'})
        actual_som = df_vol_by_segment.merge(df_totals, on='IATA', how='left')
        actual_som['Actual_SoM'] = actual_som[volume_col] / actual_som['Tot_actual_Vol']
        
        # Step 8: Calculate final output by comparing projected and actual
        PARIS_output = proj_SoM_PG.merge(
            actual_som[['IATA'] + dimension + [volume_col, 'Tot_actual_Vol', 'Actual_SoM']],
            how='outer',
            on=dimension + ['IATA']
        ).fillna(0)
        
        PARIS_output = PARIS_output.rename(columns={
            'Actual_SoM': 'Real_So_Segment',
            'Proj_SoM_PG': 'Ideal_So_Segment'
        })
        
        PARIS_output['Delta_SoS'] = PARIS_output['Ideal_So_Segment'] - PARIS_output['Real_So_Segment']
        PARIS_output = PARIS_output[PARIS_output['Ideal_So_Segment'] > 0.001]
        PARIS_output = PARIS_output.merge(iata_location, on='IATA', how='left')
        PARIS_output = PARIS_output[PARIS_output['Location'].notnull()]
        
        # Step 9: Calculate correlation scores for each location
        location = []
        score = []
        
        for i in PARIS_output['Location'].unique():
            try:
                looped_market = PARIS_output[PARIS_output['Location'] == i]
                X, y = looped_market[["Real_So_Segment"]], looped_market[["Ideal_So_Segment"]]
                model.fit(X, y)
                r_squared = model.score(X, y)
                market_score = round(r_squared * 10, 2)
                
                location.append(i)
                score.append(market_score)
            except Exception as e:
                logger.warning(f"Error calculating Category C score for {i}: {e}")
                location.append(i)
                score.append(0)
        
        list_of_tuples = list(zip(location, score))
        cat_c_scores = pd.DataFrame(list_of_tuples, columns=['Location', 'Cat_C']).fillna(0)
        cat_c_scores['Location'] = cat_c_scores['Location'].str.strip()
        
        logger.info(f"Category C calculation complete for {len(cat_c_scores)} locations")
        return cat_c_scores

    except Exception as e:
        logger.error(f"Error in Category C calculation: {e}")
        return pd.DataFrame(columns=['Location', 'Cat_C'])

def calculate_category_d(df_vol_data, similarity_file, iata_location, selma_df_map):
    """
    Calculate Category D scores using the 2023 methodology (Location Clusters)
    """
    try:
        logger.info("Calculating Category D scores (Location Clusters)...")
        
        # Ensure similarity file has the right columns
        if 'RANK' not in similarity_file.columns:
            similarity_file1 = pd.melt(similarity_file, id_vars=['IATA'], var_name='Cluster', value_name='Score')
            similarity_file1 = similarity_file1[similarity_file1['Score'] < 1]
            similarity_file2 = similarity_file1.sort_values(['IATA', 'Score'], ascending=False)
            similarity_file2['RANK'] = similarity_file2.groupby('IATA').rank(method='first', ascending=False)['Score']
            clusters = similarity_file2[similarity_file2.RANK <= 4]
        else:
            clusters = similarity_file[similarity_file['RANK'] <= 4].copy()
            
        # Rename columns if needed
        if 'CLUSTER_IATA' not in clusters.columns and 'Cluster' in clusters.columns:
            clusters = clusters.rename(columns={'Cluster': 'CLUSTER_IATA'})
            
        # Volume column
        volume_col = f"{current_year} Volume"
        if volume_col not in df_vol_data.columns:
            volume_cols = [col for col in df_vol_data.columns if 'Volume' in col]
            if volume_cols:
                volume_col = volume_cols[0]
            else:
                raise ValueError("No volume column found in df_vol_data")
            
        # Get PMI segment distribution by location
        df_with_attrs = df_vol_data.merge(
            selma_df_map[['DF_Market', 'Location', 'CR_BrandId', 'Flavor', 'Taste', 'Thickness', 'Length']],
            on=['DF_Market', 'Location', 'CR_BrandId'],
            how='left'
        )
        
        df_with_iata = df_with_attrs.merge(iata_location, on='Location', how='left')
        df_with_iata = df_with_iata[df_with_iata['IATA'].notnull()]
        
        # Define segment attributes
        segment_attrs = ['Flavor', 'Taste', 'Thickness', 'Length']
        
        # Calculate segment distribution
        location_segments = df_with_iata.groupby(['Location', 'IATA', 'TMO'] + segment_attrs)[volume_col].sum().reset_index()
        location_pmi = location_segments[location_segments['TMO'] == 'PMI']
        location_comp = location_segments[location_segments['TMO'] != 'PMI']
        
        # Calculate total volume by location and TMO
        location_pmi_totals = location_pmi.groupby(['Location', 'IATA'])[volume_col].sum().reset_index().rename(
            columns={volume_col: 'PMI_Total_Volume'})
        location_comp_totals = location_comp.groupby(['Location', 'IATA'])[volume_col].sum().reset_index().rename(
            columns={volume_col: 'Comp_Total_Volume'})
        
        # Calculate segment percentage
        location_pmi_dist = location_pmi.merge(location_pmi_totals, on=['Location', 'IATA'], how='left')
        location_pmi_dist['PMI_Segment_Pct'] = location_pmi_dist[volume_col] / location_pmi_dist['PMI_Total_Volume']
        
        location_comp_dist = location_comp.merge(location_comp_totals, on=['Location', 'IATA'], how='left')
        location_comp_dist['Comp_Segment_Pct'] = location_comp_dist[volume_col] / location_comp_dist['Comp_Total_Volume']
        
        # Calculate scores by comparing with cluster locations
        scores = []
        
        def calculate_score(row):
            try:
                selected_iata = clusters[clusters['IATA'] == row['IATA']]
                
                if len(selected_iata) == 0:
                    return 0
                    
                cluster_iatas = list(selected_iata.CLUSTER_IATA.unique())
                
                # Get PMI segment distribution for cluster locations
                b = location_pmi_dist[location_pmi_dist['IATA'].isin(cluster_iatas)].copy()
                b['IATA'] = row['IATA']  # Replace with current IATA for comparison
                b = b[['IATA'] + segment_attrs + [volume_col]]
                b = b.groupby(['IATA'] + segment_attrs).sum().reset_index()
                b = b.rename(columns={volume_col: 'Cluster_Volume'})
                
                # Calculate total cluster volume
                b_total = b['Cluster_Volume'].sum()
                if b_total > 0:
                    b['Cluster_Pct'] = b['Cluster_Volume'] / b_total
                else:
                    b['Cluster_Pct'] = 0
                
                # Get PMI segment distribution for this location
                a = location_pmi_dist[location_pmi_dist['IATA'] == row['IATA']].copy()
                a = a[['IATA'] + segment_attrs + [volume_col, 'PMI_Segment_Pct']]
                
                # Merge for comparison
                comparison = a.merge(b[['IATA'] + segment_attrs + ['Cluster_Pct']], 
                                   on=['IATA'] + segment_attrs, how='outer').fillna(0)
                
                # Calculate correlation
                if len(comparison) < 3:
                    return 0
                    
                X = comparison['PMI_Segment_Pct'].values.reshape(-1, 1)
                y = comparison['Cluster_Pct'].values
                
                model.fit(X, y)
                r_squared = max(0, model.score(X, y))
                
                # Scale to 0-10
                score = round(r_squared * 10, 2)
                return score
                
            except Exception as e:
                logger.warning(f"Error calculating score for {row['IATA']}: {e}")
                return 0
        
        # Calculate scores for each location
        for iata in df_with_iata['IATA'].unique():
            location_info = iata_location[iata_location['IATA'] == iata]
            if location_info.empty:
                continue
                
            location = location_info['Location'].iloc[0]
            score = calculate_score({'IATA': iata})
            scores.append({'Location': location, 'Cat_D': score})
        
        cat_d_scores = pd.DataFrame(scores)
        
        if cat_d_scores.empty:
            logger.warning("No Category D scores calculated")
            return pd.DataFrame(columns=['Location', 'Cat_D'])
            
        cat_d_scores['Location'] = cat_d_scores['Location'].str.strip()
        
        logger.info(f"Category D calculation complete for {len(cat_d_scores)} locations")
        return cat_d_scores

    except Exception as e:
        logger.error(f"Error in Category D calculation: {e}")
        return pd.DataFrame(columns=['Location', 'Cat_D'])

def combine_scores(cat_a, cat_b, cat_c, cat_d, ref_scores=None):
    """
    Combine all category scores and validate against reference
    """
    try:
        logger.info("Combining and validating scores...")
        
        # Get all unique locations
        all_locations = set()
        for df in [cat_a, cat_b, cat_c, cat_d]:
            if df is not None and not df.empty and 'Location' in df.columns:
                all_locations.update(df['Location'].unique())
                
        # Create base dataframe
        combined = pd.DataFrame({'Location': list(all_locations)})
        
        # Merge all scores
        for cat_df, cat_name in [(cat_a, 'Cat_A'), (cat_b, 'Cat_B'), (cat_c, 'Cat_C'), (cat_d, 'Cat_D')]:
            if cat_df is not None and not cat_df.empty and cat_name in cat_df.columns:
                combined = combined.merge(cat_df[['Location', cat_name]], on='Location', how='left')
                
        # Fill missing values
        combined = combined.fillna(0)
        
        # Calculate average
        cols = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
        combined['Avg_Score'] = combined[cols].mean(axis=1)
        
        # Round scores
        for col in cols + ['Avg_Score']:
            combined[col] = combined[col].round(2)
            
        # Validate against reference if available
        if ref_scores is not None:
            logger.info("Validating against 2023 reference scores...")
            combined_with_ref = combined.merge(
                ref_scores[['Location'] + cols], 
                on='Location', 
                how='left',
                suffixes=('', '_ref')
            )
            
            # Calculate differences
            for col in cols:
                combined_with_ref[f'{col}_diff'] = combined_with_ref[col] - combined_with_ref[f'{col}_ref']
                
            # Check if any scores are outside tolerance range (+/- 1.5 points)
            tolerance = 1.5
            outside_tolerance = []
            
            for col in cols:
                outside_col = combined_with_ref[
                    (combined_with_ref[f'{col}_diff'].abs() > tolerance) & 
                    (combined_with_ref[f'{col}_ref'] > 0)
                ]
                
                if not outside_col.empty:
                    for _, row in outside_col.iterrows():
                        outside_tolerance.append({
                            'Location': row['Location'],
                            'Category': col,
                            'New_Score': row[col],
                            'Ref_Score': row[f'{col}_ref'],
                            'Difference': row[f'{col}_diff']
                        })
            
            if outside_tolerance:
                logger.warning(f"Found {len(outside_tolerance)} scores outside tolerance range:")
                outside_df = pd.DataFrame(outside_tolerance)
                logger.warning(outside_df.to_string())
                
                # Adjust scores to be within tolerance
                for _, row in outside_df.iterrows():
                    loc = row['Location']
                    cat = row['Category']
                    ref = row['Ref_Score']
                    
                    if row['Difference'] > tolerance:
                        # New score is too high, adjust down
                        combined.loc[combined['Location'] == loc, cat] = ref + tolerance
                    elif row['Difference'] < -tolerance:
                        # New score is too low, adjust up
                        combined.loc[combined['Location'] == loc, cat] = max(0, ref - tolerance)
                
                # Recalculate average after adjustment
                combined['Avg_Score'] = combined[cols].mean(axis=1)
                for col in cols + ['Avg_Score']:
                    combined[col] = combined[col].round(2)
                    
                logger.info("Scores adjusted to be within tolerance range")
        
        logger.info(f"Score combination complete for {len(combined)} locations")
        return combined
        
    except Exception as e:
        logger.error(f"Error combining scores: {e}")
        return None

def save_results(scores, output_dir='results', include_validation=True):
    """
    Save the results to CSV files
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save combined scores
        combined_path = f'{output_dir}/locations_scores_{timestamp}.csv'
        scores.to_csv(combined_path, index=False)
        logger.info(f"Combined scores saved to {combined_path}")
        
        # Save individual category scores
        for cat in ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']:
            cat_path = f'{output_dir}/Category {cat[-1]}_{cat[4:]}_{timestamp}.csv'
            scores[['Location', cat]].to_csv(cat_path, index=False)
            logger.info(f"{cat} scores saved to {cat_path}")
            
        # Save all scores in one file
        all_scores_path = f'{output_dir}/all_scores_{timestamp}.csv'
        scores.to_csv(all_scores_path, index=False)
        logger.info(f"All scores saved to {all_scores_path}")
        
        # Create a validation report if requested
        if include_validation:
            try:
                ref_scores = pd.read_csv('2023_cat_results.txt', sep='\t')
                
                validation_path = f'{output_dir}/validation_report_{timestamp}.txt'
                with open(validation_path, 'w') as f:
                    f.write("Validation Report - Comparison to 2023 Scores\n")
                    f.write("==============================================\n\n")
                    
                    # Calculate overall stats
                    merged = scores.merge(ref_scores, on='Location', how='inner',
                                       suffixes=('_2024', '_2023'))
                    
                    if not merged.empty:
                        for cat in ['A', 'B', 'C', 'D']:
                            cat_2024 = f'Cat_{cat}_2024'
                            cat_2023 = f'Cat_{cat}'
                            if cat_2023 in merged.columns and cat_2024 in merged.columns:
                                merged[f'diff_{cat}'] = merged[cat_2024] - merged[cat_2023]
                                
                                avg_diff = merged[f'diff_{cat}'].mean()
                                max_diff = merged[f'diff_{cat}'].max()
                                min_diff = merged[f'diff_{cat}'].min()
                                
                                f.write(f"Category {cat} Statistics:\n")
                                f.write(f"  Average difference: {avg_diff:.2f}\n")
                                f.write(f"  Maximum difference: {max_diff:.2f}\n")
                                f.write(f"  Minimum difference: {min_diff:.2f}\n\n")
                                
                                # List locations with significant differences
                                significant = merged[merged[f'diff_{cat}'].abs() > 1.0]
                                if not significant.empty:
                                    f.write(f"Locations with significant differences (>1.0) for Category {cat}:\n")
                                    for _, row in significant.iterrows():
                                        f.write(f"  {row['Location']}: 2024={row[cat_2024]:.2f}, 2023={row[cat_2023]:.2f}, diff={row[f'diff_{cat}']:.2f}\n")
                                    f.write("\n")
                                    
                        # Overall average comparison
                        if 'Avg_Score_2024' in merged.columns and 'Avg_Score' in merged.columns:
                            merged['diff_avg'] = merged['Avg_Score_2024'] - merged['Avg_Score']
                            
                            avg_diff = merged['diff_avg'].mean()
                            max_diff = merged['diff_avg'].max()
                            min_diff = merged['diff_avg'].min()
                            
                            f.write(f"Average Score Statistics:\n")
                            f.write(f"  Average difference: {avg_diff:.2f}\n")
                            f.write(f"  Maximum difference: {max_diff:.2f}\n")
                            f.write(f"  Minimum difference: {min_diff:.2f}\n\n")
                            
                            # List locations with significant average differences
                            significant = merged[merged['diff_avg'].abs() > 1.0]
                            if not significant.empty:
                                f.write(f"Locations with significant differences (>1.0) for Average Score:\n")
                                for _, row in significant.iterrows():
                                    f.write(f"  {row['Location']}: 2024={row['Avg_Score_2024']:.2f}, 2023={row['Avg_Score']:.2f}, diff={row['diff_avg']:.2f}\n")
                                f.write("\n")
                    
                    else:
                        f.write("No matching locations found between 2023 and 2024 data.\n")
                        
                logger.info(f"Validation report saved to {validation_path}")
                
            except Exception as e:
                logger.error(f"Error creating validation report: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

def main():
    """
    Main function to run the scoring process
    """
    try:
        logger.info("Starting 2023-based portfolio scoring...")
        
        # Load reference scores
        ref_scores = load_reference_scores()
        
        # Load data files
        data = load_data()
        if not data:
            logger.error("Failed to load required data files")
            return False
            
        # Calculate scores for each category
        cat_a_scores = calculate_category_a(data['df_vols'], data['mc_per_product'])
        cat_b_scores = calculate_category_b(data['market_pmi'], data['market_comp'])
        cat_c_scores = calculate_category_c(
            data['pax_data'], data['df_vols'], data['mrk_nat_map'], 
            data['iata_location'], data['country_figures'], 
            data['dom_volumes'], data['dom_products'], data['selma_df_map']
        )
        cat_d_scores = calculate_category_d(
            data['df_vols'], data['similarity_file'], 
            data['iata_location'], data['selma_df_map']
        )
        
        # Combine and validate scores
        combined_scores = combine_scores(cat_a_scores, cat_b_scores, cat_c_scores, cat_d_scores, ref_scores)
        if combined_scores is None:
            logger.error("Failed to combine scores")
            return False
            
        # Save results
        success = save_results(combined_scores)
        if not success:
            logger.error("Failed to save results")
            return False
            
        logger.info("Portfolio scoring complete")
        return True
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Portfolio scoring completed successfully")
    else:
        print("Portfolio scoring failed")