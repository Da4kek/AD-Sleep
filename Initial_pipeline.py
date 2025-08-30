#!/usr/bin/env python
"""
Comprehensive Sleep-Cognition Analysis Pipeline for Manuscript Preparation
========================================================================

This pipeline integrates crossectional modeling and longitudinal correlation/Granger causality 
analyses to examine relationships between sleep variables and cognitive/neuroimaging outcomes.

Authors: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

# Machine learning libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
import patsy

# Plotting libraries
import matplotlib.patches as patches
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# Set publication-ready plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'Arial'
})

class SleepCognitionPipeline:
    """
    Comprehensive pipeline for sleep-cognition analysis with manuscript-ready outputs.
    """
    
    def __init__(self, data_path):
        """Initialize the pipeline with data."""
        self.data = pd.read_csv(data_path)
        self.results = {}
        self.figures = {}
        self.tables = {}
        
        # Define variable groups
        self.sleep_variables = ['NPIK', 'NPIKSEV', 'MHSleep']
        self.cognitive_outcomes = ['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS', 'ADNI_EF2']
        self.mri_variables = [
            'RightEntorhinal', 'LeftEntorhinal', 'RightTemporalPole', 'LeftTemporalPole',
            'RightParahippocampal', 'LeftParahippocampal', 'RightInferiorTemporal', 
            'LeftInferiorTemporal', 'RightMiddleTemporal', 'LeftMiddleTemporal',
            'RightFusiform', 'LeftFusiform', 'RightInferiorParietal', 'LeftInferiorParietal',
            'RightIsthmusCingulate', 'LeftIsthmusCingulate', 'RightBankssts', 'LeftBankssts',
            'RightPrecuneus', 'LeftPrecuneus', 'RightHippocampus', 'LeftHippocampus',
            'RightAmygdala', 'LeftAmygdala', 'RightAccumbensArea', 'LeftAccumbensArea',
            'RightMedialOrbitofrontal', 'LeftMedialOrbitofrontal', 'RightPallidum', 'LeftPallidum',
            'RightCaudalMiddleFrontal', 'LeftCaudalMiddleFrontal', 'RightPutamen', 'LeftPutamen',
            'RightRostralAnteriorCingulate', 'LeftRostralAnteriorCingulate', 'RightParacentral',
            'LeftParacentral', 'RightPrecentral', 'LeftPrecentral', 'RightLingual', 'LeftLingual',
            'RightInferiorLateralVentricle', 'LeftInferiorLateralVentricle',
            'RightLateralVentricle', 'LeftLateralVentricle'
        ]
        self.sociodemographic_variables = [
            'Adjusted_Age', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 
            'PTMARRY', 'APOE4', 'MH', 'BMI'
        ]
        
        # Predictors for modeling
        self.predictors_cognitive = [
            'PTGENDER', 'Adjusted_Age', 'PTEDUCAT', 'APOE4', 'DX_1', 'DX_2',
            'PTETHCAT_2', 'PTRACCAT', 'PTMARRY', 'MH'
        ]
        self.predictors_mri = [
            'PTGENDER', 'Adjusted_Age', 'PTEDUCAT', 'APOE4', 'DX_1', 'DX_2',
            'PTETHCAT_2', 'PTRACCAT', 'PTMARRY', 'MH'
        ]
        
        print("Pipeline initialized. Data shape:", self.data.shape)
    
    def preprocess_data(self):
        """
        Preprocess data for analysis including sleep variable creation, 
        encoding, and scaling.
        """
        print("\n=== DATA PREPROCESSING ===")
        
        # Create composite sleep variable
        sleep_components = ['Sleep_Apnea', 'Restless_Legs', 'Insomnia', 'Sleep_Disturbance_Other']
        self.data['MHSleep'] = self.data[sleep_components].sum(axis=1)
        
        # Create scaled version for crossectional analysis
        self.data_scaled = self.data.copy()
        
        # Label encoding for categorical variables
        label_encoders = {}
        categorical_cols = self.data_scaled.select_dtypes(include=['object']).columns
        
        for column in categorical_cols:
            le = LabelEncoder()
            self.data_scaled[column] = le.fit_transform(self.data_scaled[column].astype(str))
            label_encoders[column] = le
        
        # Encode sleep variables
        for column in self.sleep_variables:
            le = LabelEncoder()
            self.data_scaled[column] = le.fit_transform(self.data_scaled[column].astype(str))
            label_encoders[column] = le
        
        # Standard scaling for continuous variables
        continuous_vars = (self.cognitive_outcomes + self.mri_variables + ['BMI', 'Adjusted_Age'])
        scaler = StandardScaler()
        self.data_scaled[continuous_vars] = scaler.fit_transform(self.data_scaled[continuous_vars])
        
        # One-hot encoding for diagnosis and ethnicity
        categorical_vars = ['DX', 'PTETHCAT']
        self.data_scaled = pd.get_dummies(self.data_scaled, columns=categorical_vars, drop_first=True)
        
        # Filter for specific visit (crossectional analysis)
        self.data_cross = self.data_scaled[self.data_scaled['VISCODE'] == 3]
        
        # Separate by diagnosis groups for longitudinal analysis
        self.group_data = {
            'CN': self.data[self.data['DX_bl'] == 'CN'].sort_values(['RID', 'VISCODE']),
            'MCI': self.data[self.data['DX_bl'] == 'LMCI'].sort_values(['RID', 'VISCODE']),
            'AD': self.data[self.data['DX_bl'] == 'AD'].sort_values(['RID', 'VISCODE'])
        }
        
        print(f"Preprocessed data shapes:")
        print(f"  Crossectional: {self.data_cross.shape}")
        for group, df in self.group_data.items():
            print(f"  {group}: {df.shape}")
        
        return self
    
    def run_lasso_analysis(self):
        """
        Perform LASSO regression analysis for feature selection and coefficient estimation.
        """
        print("\n=== LASSO REGRESSION ANALYSIS ===")
        
        lasso_results = []
        model_counter = 1
        
        def store_lasso_results(model, outcome, sleep_var, feature_names, X, y):
            coefs = model.coef_
            r2 = r2_score(y, model.predict(X))
            
            records = []
            for coef, name in zip(coefs, feature_names):
                records.append({
                    'Model_ID': f"model_{model_counter}",
                    'Outcome': outcome,
                    'Sleep_Variable': sleep_var,
                    'Feature': name,
                    'Coefficient': coef,
                    'R_Squared': r2
                })
            return records
        
        # Cognitive outcomes
        print("Analyzing cognitive outcomes...")
        for sleep_var in self.sleep_variables:
            for outcome in self.cognitive_outcomes:
                try:
                    formula = f"{outcome} ~ {sleep_var} + {' + '.join(self.predictors_cognitive)}"
                    y, X = patsy.dmatrices(formula, data=self.data_cross, return_type='dataframe')
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y.values.ravel())
                    records = store_lasso_results(lasso, outcome, sleep_var, X.columns, X_scaled, y.values.ravel())
                    lasso_results.extend(records)
                    model_counter += 1
                    
                except Exception as e:
                    print(f"  Error for {outcome} with {sleep_var}: {e}")
        
        # MRI outcomes  
        print("Analyzing MRI outcomes...")
        for sleep_var in self.sleep_variables:
            for outcome in self.mri_variables:
                try:
                    formula = f"{outcome} ~ {sleep_var} + {' + '.join(self.predictors_mri)}"
                    y, X = patsy.dmatrices(formula, data=self.data_cross, return_type='dataframe')
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y.values.ravel())
                    records = store_lasso_results(lasso, outcome, sleep_var, X.columns, X_scaled, y.values.ravel())
                    lasso_results.extend(records)
                    model_counter += 1
                    
                except Exception as e:
                    print(f"  Error for {outcome} with {sleep_var}: {e}")
        
        self.lasso_results_df = pd.DataFrame(lasso_results)
        print(f"Completed LASSO analysis: {len(lasso_results)} results")
        
        return self
    
    def run_significance_testing(self):
        """
        Perform significance testing on LASSO-selected features using OLS.
        """
        print("\n=== SIGNIFICANCE TESTING ===")
        
        def lasso_significance_test(data, outcome, sleep_var, predictors):
            formula = f"{outcome} ~ {sleep_var} + {' + '.join(predictors)}"
            y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
            
            # Fit LASSO to select features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y.values.ravel())
            
            # Get selected features
            selected_columns = X.columns[lasso.coef_ != 0]
            if len(selected_columns) == 0:
                return pd.DataFrame()
            
            # Fit OLS on selected features for p-values
            X_selected = X[selected_columns]
            ols_model = sm.OLS(y, sm.add_constant(X_selected)).fit()
            
            results_df = pd.DataFrame({
                'Feature': ['Intercept'] + selected_columns.tolist(),
                'Coefficient': ols_model.params.values,
                'P_Value': ols_model.pvalues.values,
                'Std_Error': ols_model.bse.values,
                'R_Squared': [ols_model.rsquared] * len(ols_model.params)
            })
            
            return results_df
        
        # Test cognitive outcomes
        cognitive_sig_results = []
        for sleep_var in self.sleep_variables:
            for outcome in self.cognitive_outcomes:
                result = lasso_significance_test(self.data_cross, outcome, sleep_var, self.predictors_cognitive)
                result['Outcome'] = outcome
                result['Sleep_Variable'] = sleep_var
                result['Domain'] = 'Cognitive'
                cognitive_sig_results.append(result)
        
        # Test MRI outcomes
        mri_sig_results = []
        for sleep_var in self.sleep_variables:
            for outcome in self.mri_variables:
                result = lasso_significance_test(self.data_cross, outcome, sleep_var, self.predictors_mri)
                result['Outcome'] = outcome
                result['Sleep_Variable'] = sleep_var
                result['Domain'] = 'MRI'
                mri_sig_results.append(result)
        
        self.cognitive_significance_df = pd.concat(cognitive_sig_results, ignore_index=True)
        self.mri_significance_df = pd.concat(mri_sig_results, ignore_index=True)
        
        print(f"Significance testing completed")
        print(f"  Cognitive results: {len(self.cognitive_significance_df)}")
        print(f"  MRI results: {len(self.mri_significance_df)}")
        
        return self
    
    def run_correlation_analysis(self):
        """
        Perform correlation analysis across diagnostic groups.
        """
        print("\n=== CORRELATION ANALYSIS ===")
        
        def calculate_correlation(df, var1, var2):
            corr_df = df[[var1, var2]].dropna()
            corr_df[var1] = pd.to_numeric(corr_df[var1], errors='coerce')
            corr_df[var2] = pd.to_numeric(corr_df[var2], errors='coerce')
            corr_df = corr_df.dropna()
            
            if len(corr_df) > 10:  # Minimum sample size
                corr_value, p_value = pearsonr(corr_df[var1], corr_df[var2])
                return corr_value, p_value
            return np.nan, np.nan
        
        correlation_results = []
        selected_viscodes = ['bl', 'm24']
        
        all_outcomes = self.cognitive_outcomes + self.mri_variables + self.sociodemographic_variables
        
        for group_name, group_df in self.group_data.items():
            group_df_filtered = group_df[group_df['VISCODE'].isin(selected_viscodes)]
            
            for sleep_var in self.sleep_variables:
                for outcome in all_outcomes:
                    corr_value, corr_p_value = calculate_correlation(group_df_filtered, sleep_var, outcome)
                    
                    correlation_results.append({
                        'Group': group_name,
                        'Sleep_Variable': sleep_var,
                        'Outcome': outcome,
                        'Correlation': corr_value,
                        'P_Value': corr_p_value,
                        'Significant': corr_p_value < 0.05 if not pd.isna(corr_p_value) else False
                    })
        
        self.correlation_results_df = pd.DataFrame(correlation_results)
        print(f"Correlation analysis completed: {len(correlation_results)} comparisons")
        
        return self
    
    def run_granger_causality(self):
        """
        Perform Granger causality testing.
        """
        print("\n=== GRANGER CAUSALITY ANALYSIS ===")
        
        def perform_granger_test(df, var1, var2, max_lag=3):
            try:
                data = df[[var1, var2]].dropna()
                if len(data) < 20:  # Minimum sample size
                    return np.nan, np.nan
                
                # Test var1 -> var2
                result_1 = grangercausalitytests(data[[var1, var2]], max_lag, verbose=False)
                p_val_1 = result_1[1][0]['ssr_ftest'][1]
                
                # Test var2 -> var1
                result_2 = grangercausalitytests(data[[var2, var1]], max_lag, verbose=False)
                p_val_2 = result_2[1][0]['ssr_ftest'][1]
                
                return p_val_1, p_val_2
            except:
                return np.nan, np.nan
        
        granger_results = []
        all_outcomes = self.cognitive_outcomes + self.mri_variables + self.sociodemographic_variables
        
        for group_name, group_df in self.group_data.items():
            for sleep_var in self.sleep_variables:
                for outcome in all_outcomes:
                    p_val_sleep_to_outcome, p_val_outcome_to_sleep = perform_granger_test(
                        group_df, sleep_var, outcome
                    )
                    
                    granger_results.extend([
                        {
                            'Group': group_name,
                            'Sleep_Variable': sleep_var,
                            'Outcome': outcome,
                            'Direction': 'Sleep_to_Outcome',
                            'P_Value': p_val_sleep_to_outcome,
                            'Significant': p_val_sleep_to_outcome < 0.05 if not pd.isna(p_val_sleep_to_outcome) else False
                        },
                        {
                            'Group': group_name,
                            'Sleep_Variable': sleep_var,
                            'Outcome': outcome,
                            'Direction': 'Outcome_to_Sleep',
                            'P_Value': p_val_outcome_to_sleep,
                            'Significant': p_val_outcome_to_sleep < 0.05 if not pd.isna(p_val_outcome_to_sleep) else False
                        }
                    ])
        
        self.granger_results_df = pd.DataFrame(granger_results)
        print(f"Granger causality analysis completed: {len(granger_results)} tests")
        
        return self
    
    def generate_manuscript_tables(self):
        """
        Generate publication-ready tables.
        """
        print("\n=== GENERATING MANUSCRIPT TABLES ===")
        
        # Table 1: Sample Characteristics
        self.create_sample_characteristics_table()
        
        # Table 2: Crossectional Analysis Summary
        self.create_crossectional_summary_table()
        
        # Table 3: Correlation Analysis Summary
        self.create_correlation_summary_table()
        
        # Table 4: Granger Causality Summary
        self.create_granger_causality_table()
        
        print("All manuscript tables generated")
        
        return self
    
    def create_sample_characteristics_table(self):
        """Create Table 1: Sample Characteristics by Diagnostic Group."""
        
        # Calculate statistics by group
        characteristics = []
        
        for group_name, group_df in self.group_data.items():
            baseline_data = group_df[group_df['VISCODE'] == 'bl']
            
            if len(baseline_data) == 0:
                continue
                
            char_dict = {
                'Group': group_name,
                'N': len(baseline_data),
                'Age_Mean_SD': f"{baseline_data['Adjusted_Age'].mean():.1f} ± {baseline_data['Adjusted_Age'].std():.1f}",
                'Female_N_Percent': f"{sum(baseline_data['PTGENDER'] == 0)} ({100*sum(baseline_data['PTGENDER'] == 0)/len(baseline_data):.1f}%)",
                'Education_Mean_SD': f"{baseline_data['PTEDUCAT'].mean():.1f} ± {baseline_data['PTEDUCAT'].std():.1f}",
                'APOE4_Positive_N_Percent': f"{sum(baseline_data['APOE4'] == 1)} ({100*sum(baseline_data['APOE4'] == 1)/len(baseline_data):.1f}%)"
            }
            
            # Add sleep variables
            for sleep_var in self.sleep_variables:
                if sleep_var in baseline_data.columns:
                    char_dict[f'{sleep_var}_Mean_SD'] = f"{baseline_data[sleep_var].mean():.2f} ± {baseline_data[sleep_var].std():.2f}"
            
            characteristics.append(char_dict)
        
        self.tables['sample_characteristics'] = pd.DataFrame(characteristics)
        
    def create_crossectional_summary_table(self):
        """Create crossectional analysis summary table."""
        
        # Process cognitive outcomes
        cognitive_sig_sleep = self.cognitive_significance_df[
            self.cognitive_significance_df['Feature'].isin(self.sleep_variables)
        ].copy()
        
        # Process MRI outcomes  
        mri_sig_sleep = self.mri_significance_df[
            self.mri_significance_df['Feature'].isin(self.sleep_variables)
        ].copy()
        
        # Create summary tables
        def create_domain_summary(df, domain_name):
            pivot_df = df.pivot_table(
                index="Outcome",
                columns="Sleep_Variable", 
                values=["Coefficient", "P_Value"],
                aggfunc="first"
            )
            
            # Flatten column names
            pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()
            
            # Format results
            for sleep_var in self.sleep_variables:
                coef_col = f"{sleep_var}_Coefficient"
                pval_col = f"{sleep_var}_P_Value"
                
                if coef_col in pivot_df.columns and pval_col in pivot_df.columns:
                    pivot_df[sleep_var] = pivot_df.apply(
                        lambda row: (
                            f"{row[coef_col]:.3f} (p={row[pval_col]:.3f}){'*' if row[pval_col] < 0.05 else ''}"
                            if pd.notnull(row[coef_col]) else ""
                        ), axis=1
                    )
            
            # Keep only relevant columns
            result_df = pivot_df[["Outcome"] + self.sleep_variables]
            result_df['Domain'] = domain_name
            
            return result_df
        
        cognitive_summary = create_domain_summary(cognitive_sig_sleep, 'Cognitive')
        mri_summary = create_domain_summary(mri_sig_sleep, 'MRI')
        
        self.tables['crossectional_cognitive'] = cognitive_summary
        self.tables['crossectional_mri'] = mri_summary
        
    def create_correlation_summary_table(self):
        """Create correlation analysis summary table."""
        
        # Filter significant correlations
        sig_corr = self.correlation_results_df[
            self.correlation_results_df['Significant'] == True
        ].copy()
        
        if len(sig_corr) > 0:
            # Create pivot table
            pivot_df = sig_corr.pivot_table(
                index=['Sleep_Variable', 'Outcome'],
                columns='Group',
                values='Correlation',
                aggfunc='first'
            ).reset_index()
            
            # Fill NaN with empty string
            pivot_df = pivot_df.fillna('')
            
            # Format correlation values
            for group in ['CN', 'MCI', 'AD']:
                if group in pivot_df.columns:
                    pivot_df[group] = pivot_df[group].apply(
                        lambda x: f"{x:.3f}" if x != '' else ''
                    )
            
            self.tables['correlation_summary'] = pivot_df
        else:
            self.tables['correlation_summary'] = pd.DataFrame({'Message': ['No significant correlations found']})
    
    def create_granger_causality_table(self):
        """Create Granger causality summary table."""
        
        # Filter significant results
        sig_granger = self.granger_results_df[
            self.granger_results_df['Significant'] == True
        ].copy()
        
        if len(sig_granger) > 0:
            # Separate by direction
            sleep_to_outcome = sig_granger[sig_granger['Direction'] == 'Sleep_to_Outcome']
            outcome_to_sleep = sig_granger[sig_granger['Direction'] == 'Outcome_to_Sleep']
            
            # Create summary
            def summarize_direction(df, direction_name):
                summary = df.groupby(['Sleep_Variable', 'Group']).size().reset_index(name='Count')
                summary['Direction'] = direction_name
                return summary
            
            sleep_to_outcome_summary = summarize_direction(sleep_to_outcome, 'Sleep → Outcome')
            outcome_to_sleep_summary = summarize_direction(outcome_to_sleep, 'Outcome → Sleep')
            
            granger_summary = pd.concat([sleep_to_outcome_summary, outcome_to_sleep_summary], ignore_index=True)
            
            self.tables['granger_causality'] = granger_summary
        else:
            self.tables['granger_causality'] = pd.DataFrame({'Message': ['No significant Granger causality relationships found']})
    
    def generate_manuscript_figures(self):
        """
        Generate all publication-ready figures.
        """
        print("\n=== GENERATING MANUSCRIPT FIGURES ===")
        
        # Figure 1: LASSO Coefficients Heatmap
        self.create_lasso_coefficients_figure()
        
        # Figure 2: Correlation Heatmaps by Diagnostic Group  
        self.create_correlation_heatmaps()
        
        # Figure 3: Granger Causality Network Diagram
        self.create_granger_causality_figure()
        
        # Figure 4: Effect Size Comparison
        self.create_effect_size_comparison()
        
        print("All manuscript figures generated")
        
        return self
    
    def create_lasso_coefficients_figure(self):
        """Create Figure 1: LASSO Coefficients Visualization."""
        
        # Filter for sleep variable effects only
        sleep_effects = self.lasso_results_df[
            self.lasso_results_df['Feature'].isin(self.sleep_variables)
        ].copy()
        
        # Separate cognitive and MRI
        cognitive_effects = sleep_effects[sleep_effects['Outcome'].isin(self.cognitive_outcomes)]
        mri_effects = sleep_effects[sleep_effects['Outcome'].isin(self.mri_variables)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Cognitive outcomes
        if len(cognitive_effects) > 0:
            cog_pivot = cognitive_effects.pivot(
                index='Outcome', 
                columns='Sleep_Variable', 
                values='Coefficient'
            )
            
            sns.heatmap(cog_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                       ax=ax1, cbar_kws={'label': 'LASSO Coefficient'})
            ax1.set_title('Cognitive Outcomes', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Sleep Variables')
            ax1.set_ylabel('Cognitive Measures')
        
        # MRI outcomes (top 15 by absolute coefficient)
        if len(mri_effects) > 0:
            # Select top MRI regions by maximum absolute coefficient
            mri_max_coef = mri_effects.groupby('Outcome')['Coefficient'].apply(
                lambda x: x.abs().max()
            ).sort_values(ascending=False)
            
            top_mri = mri_max_coef.head(15).index
            mri_top_effects = mri_effects[mri_effects['Outcome'].isin(top_mri)]
            
            mri_pivot = mri_top_effects.pivot(
                index='Outcome',
                columns='Sleep_Variable', 
                values='Coefficient'
            )
            
            sns.heatmap(mri_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       ax=ax2, cbar_kws={'label': 'LASSO Coefficient'})
            ax2.set_title('MRI Outcomes (Top 15)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Sleep Variables')
            ax2.set_ylabel('Brain Regions')
        
        plt.tight_layout()
        plt.savefig('Figure1_LASSO_Coefficients.png', dpi=300, bbox_inches='tight')
        self.figures['lasso_coefficients'] = fig
        
    def create_correlation_heatmaps(self):
        """Create Figure 2: Correlation Heatmaps by Diagnostic Group."""
        
        # Filter significant correlations
        sig_correlations = self.correlation_results_df[
            self.correlation_results_df['Significant'] == True
        ].copy()
        
        if len(sig_correlations) == 0:
            print("No significant correlations to plot")
            return
        
        # Categorize outcomes
        def categorize_outcome(outcome):
            if outcome in self.cognitive_outcomes:
                return 'Cognitive'
            elif outcome in self.mri_variables:
                return 'MRI'
            else:
                return 'Sociodemographic'
        
        sig_correlations['Outcome_Category'] = sig_correlations['Outcome'].apply(categorize_outcome)
        
        # Create subplots for each category
        categories = sig_correlations['Outcome_Category'].unique()
        fig, axes = plt.subplots(1, len(categories), figsize=(6*len(categories), 8))
        
        if len(categories) == 1:
            axes = [axes]
        
        for i, category in enumerate(categories):
            cat_data = sig_correlations[sig_correlations['Outcome_Category'] == category]
            
            # Create pivot table
            pivot_data = cat_data.pivot_table(
                index='Outcome',
                columns=['Sleep_Variable', 'Group'],
                values='Correlation',
                aggfunc='first'
            )
            
            if pivot_data.empty:
                continue
            
            # Plot heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       ax=axes[i], cbar_kws={'label': 'Correlation Coefficient'})
            axes[i].set_title(f'{category} Outcomes', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Sleep Variables × Diagnostic Groups')
            axes[i].set_ylabel('Outcomes')
            
            # Rotate x-axis labels for readability
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('Figure2_Correlation_Heatmaps.png', dpi=300, bbox_inches='tight')
        self.figures['correlation_heatmaps'] = fig
        
    def create_granger_causality_figure(self):
        """Create Figure 3: Granger Causality Network Visualization."""
        
        # Filter significant Granger causality results
        sig_granger = self.granger_results_df[
            self.granger_results_df['Significant'] == True
        ].copy()
        
        if len(sig_granger) == 0:
            print("No significant Granger causality relationships to plot")
            return
        
        # Create separate plots for each direction
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Sleep → Outcome
        sleep_to_outcome = sig_granger[sig_granger['Direction'] == 'Sleep_to_Outcome']
        if len(sleep_to_outcome) > 0:
            # Count relationships by sleep variable and group
            sto_counts = sleep_to_outcome.groupby(['Sleep_Variable', 'Group']).size().reset_index(name='Count')
            sto_pivot = sto_counts.pivot(index='Sleep_Variable', columns='Group', values='Count').fillna(0)
            # Convert to int to avoid formatting issues
            sto_pivot = sto_pivot.astype(int)
            
            sns.heatmap(sto_pivot, annot=True, fmt='d', cmap='Reds', ax=ax1,
                       cbar_kws={'label': 'Number of Significant Relationships'})
            ax1.set_title('Sleep → Outcome Causality', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Diagnostic Groups')
            ax1.set_ylabel('Sleep Variables')
        
        # Outcome → Sleep
        outcome_to_sleep = sig_granger[sig_granger['Direction'] == 'Outcome_to_Sleep']
        if len(outcome_to_sleep) > 0:
            # Count relationships by sleep variable and group
            ots_counts = outcome_to_sleep.groupby(['Sleep_Variable', 'Group']).size().reset_index(name='Count')
            ots_pivot = ots_counts.pivot(index='Sleep_Variable', columns='Group', values='Count').fillna(0)
            # Convert to int to avoid formatting issues
            ots_pivot = ots_pivot.astype(int)
            
            sns.heatmap(ots_pivot, annot=True, fmt='d', cmap='Blues', ax=ax2,
                       cbar_kws={'label': 'Number of Significant Relationships'})
            ax2.set_title('Outcome → Sleep Causality', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Diagnostic Groups')
            ax2.set_ylabel('Sleep Variables')
        
        plt.tight_layout()
        plt.savefig('Figure3_Granger_Causality.png', dpi=300, bbox_inches='tight')
        self.figures['granger_causality'] = fig
        
    def create_effect_size_comparison(self):
        """Create Figure 4: Effect Size Comparison Across Methods."""
        
        # Combine effect sizes from different analyses
        effect_sizes = []
        
        # LASSO coefficients (standardized effect sizes)
        lasso_sleep_effects = self.lasso_results_df[
            self.lasso_results_df['Feature'].isin(self.sleep_variables)
        ].copy()
        
        for _, row in lasso_sleep_effects.iterrows():
            effect_sizes.append({
                'Sleep_Variable': row['Sleep_Variable'],
                'Outcome': row['Outcome'],
                'Effect_Size': abs(row['Coefficient']),
                'Method': 'LASSO',
                'Domain': 'Cognitive' if row['Outcome'] in self.cognitive_outcomes else 'MRI'
            })
        
        # Correlation coefficients
        sig_correlations = self.correlation_results_df[
            self.correlation_results_df['Significant'] == True
        ].copy()
        
        for _, row in sig_correlations.iterrows():
            domain = 'Cognitive' if row['Outcome'] in self.cognitive_outcomes else (
                'MRI' if row['Outcome'] in self.mri_variables else 'Sociodemographic'
            )
            effect_sizes.append({
                'Sleep_Variable': row['Sleep_Variable'],
                'Outcome': row['Outcome'],
                'Effect_Size': abs(row['Correlation']),
                'Method': f"Correlation_{row['Group']}",
                'Domain': domain
            })
        
        if len(effect_sizes) == 0:
            print("No effect sizes to compare")
            return
        
        effect_sizes_df = pd.DataFrame(effect_sizes)
        
        # Create violin plots comparing effect sizes
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Effect sizes by sleep variable
        sns.violinplot(data=effect_sizes_df, x='Sleep_Variable', y='Effect_Size', ax=axes[0,0])
        axes[0,0].set_title('Effect Sizes by Sleep Variable', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Absolute Effect Size')
        
        # Plot 2: Effect sizes by method
        sns.violinplot(data=effect_sizes_df, x='Method', y='Effect_Size', ax=axes[0,1])
        axes[0,1].set_title('Effect Sizes by Analysis Method', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Absolute Effect Size')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Effect sizes by domain
        sns.violinplot(data=effect_sizes_df, x='Domain', y='Effect_Size', ax=axes[1,0])
        axes[1,0].set_title('Effect Sizes by Outcome Domain', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('Absolute Effect Size')
        
        # Plot 4: Top 10 strongest effects
        top_effects = effect_sizes_df.nlargest(10, 'Effect_Size')
        sns.barplot(data=top_effects, x='Effect_Size', y='Outcome', hue='Sleep_Variable', ax=axes[1,1])
        axes[1,1].set_title('Top 10 Strongest Effects', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Absolute Effect Size')
        
        plt.tight_layout()
        plt.savefig('Figure4_Effect_Size_Comparison.png', dpi=300, bbox_inches='tight')
        self.figures['effect_size_comparison'] = fig
    
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summary for manuscript."""
        
        print("\n=== GENERATING STATISTICAL SUMMARY ===")
        
        summary = {
            'sample_characteristics': self._summarize_sample(),
            'crossectional_analysis': self._summarize_crossectional(),
            'correlation_analysis': self._summarize_correlations(),
            'granger_causality': self._summarize_granger_causality(),
            'effect_sizes': self._summarize_effect_sizes()
        }
        
        self.statistical_summary = summary
        
        return self
    
    def _summarize_sample(self):
        """Summarize sample characteristics."""
        
        total_n = sum(len(df) for df in self.group_data.values())
        baseline_total = sum(len(df[df['VISCODE'] == 'bl']) for df in self.group_data.values())
        
        return {
            'total_participants': total_n,
            'baseline_participants': baseline_total,
            'group_sizes': {group: len(df) for group, df in self.group_data.items()},
            'age_range': f"{self.data['Adjusted_Age'].min():.1f}-{self.data['Adjusted_Age'].max():.1f}",
            'follow_up_duration': f"{self.data['VISCODE'].nunique()} visits"
        }
    
    def _summarize_crossectional(self):
        """Summarize crossectional analysis results."""
        
        # Count significant relationships
        cog_sig = self.cognitive_significance_df[
            (self.cognitive_significance_df['Feature'].isin(self.sleep_variables)) &
            (self.cognitive_significance_df['P_Value'] < 0.05)
        ]
        
        mri_sig = self.mri_significance_df[
            (self.mri_significance_df['Feature'].isin(self.sleep_variables)) &
            (self.mri_significance_df['P_Value'] < 0.05)
        ]
        
        return {
            'total_models_tested': len(self.sleep_variables) * (len(self.cognitive_outcomes) + len(self.mri_variables)),
            'significant_cognitive_associations': len(cog_sig),
            'significant_mri_associations': len(mri_sig),
            'strongest_cognitive_effect': cog_sig.loc[cog_sig['Coefficient'].abs().idxmax()] if len(cog_sig) > 0 else None,
            'strongest_mri_effect': mri_sig.loc[mri_sig['Coefficient'].abs().idxmax()] if len(mri_sig) > 0 else None
        }
    
    def _summarize_correlations(self):
        """Summarize correlation analysis results."""
        
        sig_correlations = self.correlation_results_df[
            self.correlation_results_df['Significant'] == True
        ]
        
        if len(sig_correlations) == 0:
            return {'total_significant_correlations': 0}
        
        return {
            'total_significant_correlations': len(sig_correlations),
            'by_group': sig_correlations['Group'].value_counts().to_dict(),
            'by_sleep_variable': sig_correlations['Sleep_Variable'].value_counts().to_dict(),
            'strongest_correlation': sig_correlations.loc[sig_correlations['Correlation'].abs().idxmax()],
            'average_effect_size': sig_correlations['Correlation'].abs().mean()
        }
    
    def _summarize_granger_causality(self):
        """Summarize Granger causality results."""
        
        sig_granger = self.granger_results_df[
            self.granger_results_df['Significant'] == True
        ]
        
        if len(sig_granger) == 0:
            return {'total_significant_relationships': 0}
        
        sleep_to_outcome = sig_granger[sig_granger['Direction'] == 'Sleep_to_Outcome']
        outcome_to_sleep = sig_granger[sig_granger['Direction'] == 'Outcome_to_Sleep']
        
        return {
            'total_significant_relationships': len(sig_granger),
            'sleep_to_outcome_count': len(sleep_to_outcome),
            'outcome_to_sleep_count': len(outcome_to_sleep),
            'by_group': sig_granger['Group'].value_counts().to_dict(),
            'by_sleep_variable': sig_granger['Sleep_Variable'].value_counts().to_dict()
        }
    
    def _summarize_effect_sizes(self):
        """Summarize effect sizes across analyses."""
        
        # LASSO effect sizes
        lasso_effects = self.lasso_results_df[
            self.lasso_results_df['Feature'].isin(self.sleep_variables)
        ]['Coefficient'].abs()
        
        # Correlation effect sizes
        correlation_effects = self.correlation_results_df[
            self.correlation_results_df['Significant'] == True
        ]['Correlation'].abs()
        
        return {
            'lasso_effect_sizes': {
                'mean': lasso_effects.mean(),
                'median': lasso_effects.median(),
                'max': lasso_effects.max(),
                'min': lasso_effects.min()
            },
            'correlation_effect_sizes': {
                'mean': correlation_effects.mean(),
                'median': correlation_effects.median(), 
                'max': correlation_effects.max(),
                'min': correlation_effects.min()
            } if len(correlation_effects) > 0 else None
        }
    
    def save_results(self, output_dir='manuscript_outputs'):
        """Save all results, tables, and figures to specified directory."""
        
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== SAVING RESULTS TO {output_dir} ===")
        
        # Save tables
        tables_dir = os.path.join(output_dir, 'tables')
        os.makedirs(tables_dir, exist_ok=True)
        
        for table_name, table_df in self.tables.items():
            filename = os.path.join(tables_dir, f'Table_{table_name}.csv')
            table_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")
        
        # Save figures
        figures_dir = os.path.join(output_dir, 'figures')  
        os.makedirs(figures_dir, exist_ok=True)
        
        for fig_name, fig_obj in self.figures.items():
            filename = os.path.join(figures_dir, f'Figure_{fig_name}.png')
            fig_obj.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        # Save raw results
        results_dir = os.path.join(output_dir, 'raw_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save dataframes
        self.lasso_results_df.to_csv(os.path.join(results_dir, 'lasso_results.csv'), index=False)
        self.cognitive_significance_df.to_csv(os.path.join(results_dir, 'cognitive_significance.csv'), index=False)
        self.mri_significance_df.to_csv(os.path.join(results_dir, 'mri_significance.csv'), index=False)
        self.correlation_results_df.to_csv(os.path.join(results_dir, 'correlation_results.csv'), index=False)
        self.granger_results_df.to_csv(os.path.join(results_dir, 'granger_results.csv'), index=False)
        
        # Save statistical summary
        import json
        with open(os.path.join(output_dir, 'statistical_summary.json'), 'w') as f:
            # Convert non-serializable objects to strings
            summary_serializable = {}
            for key, value in self.statistical_summary.items():
                if isinstance(value, dict):
                    summary_serializable[key] = {k: str(v) for k, v in value.items()}
                else:
                    summary_serializable[key] = str(value)
            json.dump(summary_serializable, f, indent=2)
        
        print(f"All results saved to {output_dir}")
        
        return self
    
    def print_manuscript_summary(self):
        """Print a comprehensive summary for manuscript writing."""
        
        print("\n" + "="*80)
        print("MANUSCRIPT SUMMARY")
        print("="*80)
        
        # Sample characteristics
        print("\n1. SAMPLE CHARACTERISTICS")
        print("-" * 40)
        sample_stats = self.statistical_summary['sample_characteristics']
        print(f"Total participants: {sample_stats['total_participants']}")
        print(f"Baseline participants: {sample_stats['baseline_participants']}")
        print("Group sizes:")
        for group, size in sample_stats['group_sizes'].items():
            print(f"  {group}: {size}")
        print(f"Age range: {sample_stats['age_range']} years")
        print(f"Follow-up: {sample_stats['follow_up_duration']}")
        
        # Crossectional results
        print("\n2. CROSSECTIONAL ANALYSIS RESULTS")
        print("-" * 40)
        cross_stats = self.statistical_summary['crossectional_analysis']
        print(f"Total models tested: {cross_stats['total_models_tested']}")
        print(f"Significant cognitive associations: {cross_stats['significant_cognitive_associations']}")
        print(f"Significant MRI associations: {cross_stats['significant_mri_associations']}")
        
        # Correlation results
        print("\n3. CORRELATION ANALYSIS RESULTS")
        print("-" * 40)
        corr_stats = self.statistical_summary['correlation_analysis']
        print(f"Total significant correlations: {corr_stats['total_significant_correlations']}")
        if corr_stats['total_significant_correlations'] > 0:
            print("By diagnostic group:")
            for group, count in corr_stats['by_group'].items():
                print(f"  {group}: {count}")
        
        # Granger causality results
        print("\n4. GRANGER CAUSALITY RESULTS")
        print("-" * 40)
        granger_stats = self.statistical_summary['granger_causality']
        print(f"Total significant causal relationships: {granger_stats['total_significant_relationships']}")
        if granger_stats['total_significant_relationships'] > 0:
            print(f"Sleep → Outcome: {granger_stats['sleep_to_outcome_count']}")
            print(f"Outcome → Sleep: {granger_stats['outcome_to_sleep_count']}")
        
        # Effect sizes
        print("\n5. EFFECT SIZES SUMMARY")
        print("-" * 40)
        effect_stats = self.statistical_summary['effect_sizes']
        if 'lasso_effect_sizes' in effect_stats:
            lasso_effects = effect_stats['lasso_effect_sizes']
            print(f"LASSO coefficients - Mean: {lasso_effects['mean']:.3f}, Max: {lasso_effects['max']:.3f}")
        
        if effect_stats['correlation_effect_sizes']:
            corr_effects = effect_stats['correlation_effect_sizes']
            print(f"Correlations - Mean: {corr_effects['mean']:.3f}, Max: {corr_effects['max']:.3f}")
        
        print("\n" + "="*80)
        print("FILES GENERATED:")
        print("- Tables: Sample characteristics, crossectional results, correlation summary, Granger causality")
        print("- Figures: LASSO coefficients, correlation heatmaps, Granger causality network, effect size comparison")
        print("- Raw results: All statistical outputs in CSV format")
        print("- Statistical summary: JSON file with all summary statistics")
        print("="*80)
        
        return self
    
    def run_full_pipeline(self, output_dir='manuscript_outputs', skip_figures=False, skip_granger=False):
        """
        Execute the complete analysis pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save all outputs
        skip_figures : bool
            Skip figure generation for faster execution
        skip_granger : bool
            Skip Granger causality analysis (time-intensive)
        """
        
        print("SLEEP-COGNITION MANUSCRIPT PIPELINE")
        print("="*50)
        print(f"Analysis started at: {pd.Timestamp.now()}")
        
        # Execute preprocessing and basic analyses
        (self.preprocess_data()
         .run_lasso_analysis()
         .run_significance_testing()
         .run_correlation_analysis())
        
        # Conditionally run Granger causality
        if not skip_granger:
            self.run_granger_causality()
        else:
            print("Skipping Granger causality analysis...")
            self.granger_results_df = pd.DataFrame()
        
        # Generate tables
        self.generate_manuscript_tables()
        
        # Conditionally generate figures
        if not skip_figures:
            self.generate_manuscript_figures()
        else:
            print("Skipping figure generation...")
            self.figures = {}
        
        # Generate summary and save
        (self.generate_statistical_summary()
         .save_results(output_dir)
         .print_manuscript_summary())
        
        print(f"\nPipeline completed at: {pd.Timestamp.now()}")
        print(f"All outputs saved to: {output_dir}")
        
        return self


# Example usage and command-line interface
if __name__ == "__main__":
    import argparse
    import sys
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Sleep-Cognition Analysis Pipeline for Manuscript Preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Pipeline.py --data_file Latest_all_in_one.csv
  python Pipeline.py --data_file data.csv --output_dir Documents/results
  python Pipeline.py --data_file data.csv --output_dir results --skip_figures
        """
    )
    
    parser.add_argument(
        '--data_file', 
        type=str,
        default='Latest_all_in_one.csv',
        help='Path to the input CSV data file (default: Latest_all_in_one.csv)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str, 
        default='manuscript_outputs',
        help='Directory to save all outputs (default: manuscript_outputs)'
    )
    
    parser.add_argument(
        '--skip_figures',
        action='store_true',
        help='Skip figure generation (useful for quick testing)'
    )
    
    parser.add_argument(
        '--skip_granger',
        action='store_true', 
        help='Skip Granger causality analysis (time-intensive)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if data file exists
    import os
    if not os.path.exists(args.data_file):
        print(f"Error: Data file '{args.data_file}' not found!")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        if args.verbose:
            print(f"Initializing pipeline with data file: {args.data_file}")
            print(f"Output directory: {args.output_dir}")
        
        pipeline = SleepCognitionPipeline(args.data_file)
        
        # Run preprocessing and basic analyses
        print("SLEEP-COGNITION MANUSCRIPT PIPELINE")
        print("="*50)
        print(f"Analysis started at: {pd.Timestamp.now()}")
        
        (pipeline.preprocess_data()
         .run_lasso_analysis()
         .run_significance_testing()
         .run_correlation_analysis())
        
        # Conditionally run Granger causality
        if not args.skip_granger:
            pipeline.run_granger_causality()
        else:
            print("Skipping Granger causality analysis...")
            # Create empty results for consistency
            pipeline.granger_results_df = pd.DataFrame()
        
        # Generate tables and statistical summary
        pipeline.generate_manuscript_tables()
        
        # Conditionally generate figures
        if not args.skip_figures:
            pipeline.generate_manuscript_figures()
        else:
            print("Skipping figure generation...")
            pipeline.figures = {}
        
        # Generate summary and save results
        (pipeline.generate_statistical_summary()
         .save_results(args.output_dir)
         .print_manuscript_summary())
        
        print(f"\nPipeline completed successfully at: {pd.Timestamp.now()}")
        print(f"All outputs saved to: {args.output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Example programmatic usage (commented out when run from command line)
    """
    # For programmatic usage:
    pipeline = SleepCognitionPipeline("Latest_all_in_one.csv")
    results = pipeline.run_full_pipeline(output_dir='manuscript_outputs')
    
    # Access specific results
    print("\nSample characteristics table:")
    print(pipeline.tables['sample_characteristics'])
    
    print("\nTop 10 significant crossectional results (cognitive):")
    if 'crossectional_cognitive' in pipeline.tables:
        print(pipeline.tables['crossectional_cognitive'].head(10))
    """