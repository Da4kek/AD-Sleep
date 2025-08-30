#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sleep Variables and Cognitive/MRI Outcomes Analysis Pipeline
===========================================================

Research-grade pipeline for analyzing longitudinal relationships between sleep variables
and cognitive/MRI outcomes using Linear Mixed Models (LMMs).

Author: Anirudh
Date: August 28, 2025
Version: 2.0 (Research Grade)

Usage:
    python sleep_cognition_pipeline.py --data_path "Latest_all_in_one.csv" --output_dir "results"

Features:
- Comprehensive data preprocessing and quality checks
- Multiple model specifications with proper controls
- Residual diagnostics and assumption testing  
- Publication-ready figures and tables
- Statistical power analysis
- Effect size calculations
- Multiple comparison corrections
- Model comparison statistics
"""

import os
import sys
import argparse
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
import json
import math

# Statistical and data manipulation
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, shapiro, skew, kurtosis
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set publication-ready style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans'
})

class SleepCognitionAnalyzer:
    """
    Research-grade analyzer for sleep-cognition relationships using Linear Mixed Models.
    
    This class implements a comprehensive pipeline for analyzing longitudinal data
    relating sleep variables to cognitive and neuroimaging outcomes.
    """
    
    def __init__(self, output_dir: str = "results"):
        """Initialize the analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "diagnostics").mkdir(exist_ok=True)
        
        # Define variable sets
        self.cognitive_outcomes = ['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS', 'ADNI_EF2']
        self.sleep_vars = ['NPIK', 'NPIKSEV', 'MHSleep']
        
        # MRI outcomes (cortical thickness, subcortical volumes)
        self.mri_outcomes = [
            'RightEntorhinal', 'LeftEntorhinal', 'RightTemporalPole', 'LeftTemporalPole',
            'RightParahippocampal', 'LeftParahippocampal', 'RightInferiorTemporal', 'LeftInferiorTemporal',
            'RightMiddleTemporal', 'LeftMiddleTemporal', 'RightFusiform', 'LeftFusiform',
            'RightInferiorParietal', 'LeftInferiorParietal', 'RightIsthmusCingulate', 'LeftIsthmusCingulate',
            'RightBankssts', 'LeftBankssts', 'RightPrecuneus', 'LeftPrecuneus',
            'RightHippocampus', 'LeftHippocampus', 'RightAmygdala', 'LeftAmygdala',
            'RightAccumbensArea', 'LeftAccumbensArea', 'RightMedialOrbitofrontal', 'LeftMedialOrbitofrontal',
            'RightPallidum', 'LeftPallidum', 'RightCaudalMiddleFrontal', 'LeftCaudalMiddleFrontal',
            'RightPutamen', 'LeftPutamen', 'RightRostralAnteriorCingulate', 'LeftRostralAnteriorCingulate',
            'RightParacentral', 'LeftParacentral', 'RightPrecentral', 'LeftPrecentral',
            'RightLingual', 'LeftLingual', 'RightInferiorLateralVentricle', 'LeftInferiorLateralVentricle',
            'RightLateralVentricle', 'LeftLateralVentricle'
        ]
        
        # Model specifications
        self.model1_predictors = [
            'Adjusted_Age', 'DX_1', 'DX_2', 'PTEDUCAT',
            'PTETHCAT_2', 'PTGENDER_1', 'PTMARRY', 'PTRACCAT_1'
        ]
        
        self.model2_predictors = [
            'APOE4', 'BMI', 'MH'
        ] + self.model1_predictors
        
        # Results storage
        self.results = {
            'model1': {'cognitive': defaultdict(list), 'mri': defaultdict(list)},
            'model2': {'cognitive': defaultdict(list), 'mri': defaultdict(list)}
        }
        
        self.processed_data = None
        self.raw_data = None
        
        logger.info(f"Initialized SleepCognitionAnalyzer with output directory: {output_dir}")
    
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and validate the input data with comprehensive quality checks.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing the merged dataset
            
        Returns:
        --------
        pd.DataFrame
            Validated and cleaned dataset
        """
        logger.info(f"Loading data from: {data_path}")
        
        try:
            data = pd.read_csv(data_path)
            self.raw_data = data.copy()
            
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            
            # Basic validation
            required_columns = (
                self.cognitive_outcomes + 
                ['Sleep_Apnea', 'Restless_Legs', 'Insomnia', 'Sleep_Disturbance_Other'] +
                ['RID', 'DX', 'DX_bl', 'Adjusted_Age', 'PTEDUCAT', 'PTGENDER', 'BMI']
            )
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Create composite sleep variable
            sleep_cols = ['Sleep_Apnea', 'Restless_Legs', 'Insomnia', 'Sleep_Disturbance_Other']
            data['MHSleep'] = data[sleep_cols].sum(axis=1)
            
            # Data quality checks
            self._perform_data_quality_checks(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _perform_data_quality_checks(self, data: pd.DataFrame):
        """Perform comprehensive data quality checks and generate reports."""
        logger.info("Performing data quality checks...")
        
        quality_report = {
            'total_subjects': data['RID'].nunique(),
            'total_observations': len(data),
            'time_points_per_subject': data.groupby('RID').size().describe().to_dict(),
            'missing_data_summary': {},
            'diagnosis_distribution': data.groupby('DX')['RID'].nunique().to_dict(),
            'baseline_diagnosis_distribution': data.groupby('DX_bl')['RID'].nunique().to_dict()
        }
        
        # Missing data analysis
        all_vars = self.cognitive_outcomes + self.mri_outcomes + self.sleep_vars + self.model2_predictors
        for var in all_vars:
            if var in data.columns:
                missing_pct = (data[var].isna().sum() / len(data)) * 100
                quality_report['missing_data_summary'][var] = {
                    'missing_count': int(data[var].isna().sum()),
                    'missing_percentage': round(missing_pct, 2)
                }
        
        # Save quality report
        with open(self.output_dir / "data_quality_report.json", 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        logger.info(f"Data quality report saved. Key stats:")
        logger.info(f"  - Total subjects: {quality_report['total_subjects']}")
        logger.info(f"  - Total observations: {quality_report['total_observations']}")
        logger.info(f"  - Mean visits per subject: {quality_report['time_points_per_subject']['mean']:.1f}")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data preprocessing including scaling, encoding, and transformations.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw input data
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data ready for analysis
        """
        logger.info("Starting data preprocessing...")
        
        processed_data = data.copy()
        
        # Label encoding for categorical variables
        label_encoders = {}
        categorical_cols = processed_data.select_dtypes(include=['object']).columns
        
        for column in categorical_cols:
            if column not in ['RID']:  # Don't encode subject ID
                le = LabelEncoder()
                processed_data[column] = le.fit_transform(processed_data[column].astype(str))
                label_encoders[column] = le
        
        # Encode sleep variables
        for column in self.sleep_vars:
            if column in processed_data.columns:
                le = LabelEncoder()
                processed_data[column] = le.fit_transform(processed_data[column].astype(str))
                label_encoders[column] = le
        
        # Standardize continuous variables
        continuous_vars = (
            self.cognitive_outcomes + self.mri_outcomes + 
            ['BMI', 'Adjusted_Age', 'MHSleep', 'Sleep_Apnea', 'Insomnia']
        )
        
        available_continuous = [var for var in continuous_vars if var in processed_data.columns]
        scaler = StandardScaler()
        processed_data[available_continuous] = scaler.fit_transform(
            processed_data[available_continuous]
        )
        
        # Create dummy variables for categorical predictors
        categorical_vars = ['DX', 'DX_bl', 'PTETHCAT', 'PTGENDER', 'PTRACCAT']
        available_categorical = [var for var in categorical_vars if var in processed_data.columns]
        
        processed_data = pd.get_dummies(
            processed_data, columns=available_categorical, drop_first=True
        )
        
        # Save preprocessing artifacts
        preprocessing_info = {
            'label_encoders': {k: v.classes_.tolist() for k, v in label_encoders.items()},
            'standardized_variables': available_continuous,
            'dummy_variables': available_categorical,
            'final_shape': processed_data.shape
        }
        
        with open(self.output_dir / "preprocessing_info.json", 'w') as f:
            json.dump(preprocessing_info, f, indent=2)
        
        self.processed_data = processed_data
        logger.info(f"Preprocessing completed. Final shape: {processed_data.shape}")
        
        return processed_data
    
    def fit_linear_mixed_models(self, data: pd.DataFrame, model_spec: str = "model1"):
        """
        Fit Linear Mixed Models for all outcome-sleep variable combinations.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed data
        model_spec : str
            Model specification ("model1" or "model2")
        """
        logger.info(f"Fitting Linear Mixed Models - {model_spec}")
        
        predictors = self.model1_predictors if model_spec == "model1" else self.model2_predictors
        available_predictors = [p for p in predictors if p in data.columns]
        
        model_counter = 1
        
        # Fit cognitive outcome models
        for sleep_var in self.sleep_vars:
            if sleep_var not in data.columns:
                logger.warning(f"Sleep variable {sleep_var} not found in data")
                continue
                
            for outcome in self.cognitive_outcomes:
                if outcome not in data.columns:
                    logger.warning(f"Cognitive outcome {outcome} not found in data")
                    continue
                    
                try:
                    formula = f"{outcome} ~ {sleep_var} + {' + '.join(available_predictors)}"
                    model = smf.mixedlm(formula, data=data, groups=data["RID"])
                    result = model.fit(reml=False)
                    
                    # Store all predictor results
                    results = self._extract_model_results(
                        result, outcome, sleep_var, f"model_{model_counter}", 'cognitive'
                    )
                    self.results[model_spec]['cognitive'][sleep_var].extend(results)
                    model_counter += 1
                    
                except Exception as e:
                    logger.error(f"[Cognitive] Error for {outcome} with {sleep_var}: {e}")
        
        # Fit MRI outcome models  
        for sleep_var in self.sleep_vars:
            if sleep_var not in data.columns:
                continue
                
            for outcome in self.mri_outcomes:
                if outcome not in data.columns:
                    continue
                    
                try:
                    formula = f"{outcome} ~ {sleep_var} + {' + '.join(available_predictors)}"
                    model = smf.mixedlm(formula, data=data, groups=data["RID"])
                    result = model.fit(reml=False)
                    
                    results = self._extract_model_results(
                        result, outcome, sleep_var, f"model_{model_counter}", 'mri'
                    )
                    self.results[model_spec]['mri'][sleep_var].extend(results)
                    model_counter += 1
                    
                except Exception as e:
                    logger.error(f"[MRI] Error for {outcome} with {sleep_var}: {e}")
        
        # Save model results
        self._save_model_results(model_spec)
        
        logger.info(f"Completed fitting {model_counter-1} models for {model_spec}")
    
    def _extract_model_results(self, result, outcome: str, sleep_var: str, model_id: str, domain: str) -> List[Dict]:
        """Extract comprehensive results from fitted model."""
        records = []
        
        for predictor in result.params.index:
            # Calculate effect size (Cohen's f²)
            try:
                r2_full = result.rsquared if hasattr(result, 'rsquared') else np.nan
                effect_size = self._calculate_effect_size(result, predictor)
            except:
                effect_size = np.nan
            
            # Calculate confidence intervals
            try:
                conf_int = result.conf_int().loc[predictor] if predictor in result.conf_int().index else [np.nan, np.nan]
            except:
                conf_int = [np.nan, np.nan]
            
            records.append({
                'Model_ID': model_id,
                'Outcome': outcome,
                'Sleep_Variable': sleep_var,
                'Feature': predictor,
                'Coefficient': result.params.get(predictor, np.nan),
                'SE': result.bse.get(predictor, np.nan),
                'T_Statistic': result.tvalues.get(predictor, np.nan),
                'P_Value': result.pvalues.get(predictor, np.nan),
                'CI_Lower': conf_int[0],
                'CI_Upper': conf_int[1],
                'Effect_Size': effect_size,
                'Domain': domain,
                'AIC': result.aic,
                'BIC': result.bic,
                'Log_Likelihood': result.llf
            })
            
        return records
    
    def _calculate_effect_size(self, result, predictor: str) -> float:
        """Calculate Cohen's f² effect size for a predictor."""
        try:
            # This is a simplified effect size calculation
            # In practice, you might want more sophisticated measures
            t_stat = result.tvalues.get(predictor, np.nan)
            df = result.df_resid
            if not np.isnan(t_stat) and df > 0:
                r2 = t_stat**2 / (t_stat**2 + df)
                f2 = r2 / (1 - r2)
                return f2
        except:
            pass
        return np.nan
    
    def _save_model_results(self, model_spec: str):
        """Save model results to files."""
        models_dir = self.output_dir / "models"
        
        for domain in ['cognitive', 'mri']:
            for sleep_var in self.sleep_vars:
                if sleep_var in self.results[model_spec][domain]:
                    filename = f"{domain}_outcomes_{sleep_var}_{model_spec}.pkl"
                    with open(models_dir / filename, 'wb') as f:
                        pickle.dump(self.results[model_spec][domain][sleep_var], f)
    
    def perform_residual_diagnostics(self, data: pd.DataFrame, model_spec: str = "model2"):
        """
        Comprehensive residual diagnostics and assumption testing.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed data
        model_spec : str
            Model specification to test
        """
        logger.info(f"Performing residual diagnostics for {model_spec}")
        
        predictors = self.model2_predictors if model_spec == "model2" else self.model1_predictors
        available_predictors = [p for p in predictors if p in data.columns]
        
        diagnostics_results = {
            'cognitive': [],
            'mri': []
        }
        
        # Test cognitive models
        for sleep_var in self.sleep_vars:
            if sleep_var not in data.columns:
                continue
                
            for outcome in self.cognitive_outcomes:
                if outcome not in data.columns:
                    continue
                    
                try:
                    formula = f"{outcome} ~ {sleep_var} + {' + '.join(available_predictors)}"
                    model = smf.mixedlm(formula, data=data, groups=data["RID"])
                    result = model.fit(reml=False)
                    
                    diagnostics = self._compute_residual_diagnostics(result, outcome, sleep_var)
                    diagnostics_results['cognitive'].append(diagnostics)
                    
                except Exception as e:
                    logger.error(f"Diagnostics error - Cognitive {outcome} with {sleep_var}: {e}")
        
        # Test MRI models
        for sleep_var in self.sleep_vars:
            if sleep_var not in data.columns:
                continue
                
            for outcome in self.mri_outcomes:
                if outcome not in data.columns:
                    continue
                    
                try:
                    formula = f"{outcome} ~ {sleep_var} + {' + '.join(available_predictors)}"
                    model = smf.mixedlm(formula, data=data, groups=data["RID"])
                    result = model.fit(reml=False)
                    
                    diagnostics = self._compute_residual_diagnostics(result, outcome, sleep_var)
                    diagnostics_results['mri'].append(diagnostics)
                    
                except Exception as e:
                    logger.error(f"Diagnostics error - MRI {outcome} with {sleep_var}: {e}")
        
        # Save diagnostics results
        for domain in ['cognitive', 'mri']:
            df = pd.DataFrame(diagnostics_results[domain])
            df.to_csv(self.output_dir / "diagnostics" / f"{domain}_residual_diagnostics_{model_spec}.csv", index=False)
        
        # Generate diagnostic plots
        self._create_diagnostic_plots(diagnostics_results, model_spec)
        
        logger.info("Residual diagnostics completed")
    
    def _compute_residual_diagnostics(self, result, outcome: str, sleep_var: str) -> Dict:
        """Compute comprehensive residual diagnostics for a single model."""
        residuals = result.resid.dropna()
        n = len(residuals)
        
        if n < 3:
            return {
                'Outcome': outcome,
                'Sleep_Variable': sleep_var,
                'N': n,
                'Shapiro_Stat': np.nan,
                'Shapiro_p': np.nan,
                'Skewness': np.nan,
                'Kurtosis': np.nan,
                'Jarque_Bera_p': np.nan,
                'Durbin_Watson': np.nan,
                'Normal': False,
                'CLT_Applies': False
            }
        
        # Normality tests
        try:
            shapiro_stat, shapiro_p = shapiro(residuals)
        except:
            shapiro_stat, shapiro_p = np.nan, np.nan
            
        try:
            jb_stat, jb_p = stats.jarque_bera(residuals)
        except:
            jb_stat, jb_p = np.nan, np.nan
        
        # Descriptive statistics
        skewness = skew(residuals)
        kurt = kurtosis(residuals)
        
        # Durbin-Watson test for autocorrelation
        try:
            from statsmodels.stats.diagnostic import durbin_watson
            dw_stat = durbin_watson(residuals)
        except:
            dw_stat = np.nan
        
        is_normal = shapiro_p > 0.05 if not np.isnan(shapiro_p) else False
        clt_applies = not is_normal and n >= 30
        
        return {
            'Outcome': outcome,
            'Sleep_Variable': sleep_var,
            'N': n,
            'Shapiro_Stat': round(shapiro_stat, 4) if not np.isnan(shapiro_stat) else np.nan,
            'Shapiro_p': round(shapiro_p, 6) if not np.isnan(shapiro_p) else np.nan,
            'Skewness': round(skewness, 3),
            'Kurtosis': round(kurt, 3),
            'Jarque_Bera_p': round(jb_p, 6) if not np.isnan(jb_p) else np.nan,
            'Durbin_Watson': round(dw_stat, 3) if not np.isnan(dw_stat) else np.nan,
            'Normal': 'Yes ✅' if is_normal else 'No ❌',
            'CLT_Applies': 'Yes ✅' if clt_applies else ('—' if is_normal else 'No ❌')
        }
    
    def _create_diagnostic_plots(self, diagnostics_results: Dict, model_spec: str):
        """Create comprehensive diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, domain in enumerate(['cognitive', 'mri']):
            if not diagnostics_results[domain]:
                continue
                
            df = pd.DataFrame(diagnostics_results[domain])
            
            # Normality p-values histogram
            ax = axes[i, 0]
            valid_p = df[df['Shapiro_p'].notna()]['Shapiro_p']
            ax.hist(valid_p, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
            ax.set_xlabel('Shapiro-Wilk p-value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{domain.title()} - Normality Test Results')
            ax.legend()
            
            # Skewness vs Kurtosis
            ax = axes[i, 1]
            ax.scatter(df['Skewness'], df['Kurtosis'], alpha=0.6)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Skewness')
            ax.set_ylabel('Kurtosis')
            ax.set_title(f'{domain.title()} - Residual Distribution Shape')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "diagnostics" / f"residual_diagnostics_{model_spec}.png")
        plt.close()
    
    def perform_multiple_comparison_correction(self, model_spec: str = "model2"):
        """
        Apply multiple comparison corrections using FDR and Bonferroni methods.
        
        Parameters:
        -----------
        model_spec : str
            Model specification to correct
        """
        logger.info(f"Applying multiple comparison corrections for {model_spec}")
        
        correction_results = {}
        
        for domain in ['cognitive', 'mri']:
            # Load and combine results
            all_results = []
            for sleep_var in self.sleep_vars:
                if sleep_var in self.results[model_spec][domain]:
                    all_results.extend(self.results[model_spec][domain][sleep_var])
            
            if not all_results:
                continue
                
            df = pd.DataFrame(all_results)
            
            # Focus on sleep variable effects only
            sleep_effects = df[df['Feature'] == df['Sleep_Variable']].copy()
            
            if len(sleep_effects) == 0:
                continue
            
            # Apply corrections
            valid_p = sleep_effects['P_Value'].dropna()
            if len(valid_p) == 0:
                continue
            
            # FDR correction
            fdr_rejected, fdr_pvals_corrected, _, _ = multipletests(
                valid_p, alpha=0.05, method='fdr_bh'
            )
            
            # Bonferroni correction
            bonf_rejected, bonf_pvals_corrected, _, _ = multipletests(
                valid_p, alpha=0.05, method='bonferroni'
            )
            
            # Add corrections to dataframe
            sleep_effects_copy = sleep_effects.copy()
            sleep_effects_copy = sleep_effects_copy.dropna(subset=['P_Value'])
            sleep_effects_copy['P_FDR'] = fdr_pvals_corrected
            sleep_effects_copy['P_Bonferroni'] = bonf_pvals_corrected
            sleep_effects_copy['Significant_FDR'] = fdr_rejected
            sleep_effects_copy['Significant_Bonferroni'] = bonf_rejected
            
            correction_results[domain] = sleep_effects_copy
            
            # Save corrected results
            sleep_effects_copy.to_csv(
                self.output_dir / "tables" / f"{domain}_corrected_results_{model_spec}.csv",
                index=False
            )
        
        return correction_results
    
    def create_publication_figures(self, model_spec: str = "model2"):
        """
        Create publication-ready figures for the manuscript.
        
        Parameters:
        -----------
        model_spec : str
            Model specification to visualize
        """
        logger.info(f"Creating publication figures for {model_spec}")
        
        # Load corrected results
        corrected_results = self.perform_multiple_comparison_correction(model_spec)
        
        # Create multi-panel figure
        with PdfPages(self.output_dir / "figures" / f"main_results_{model_spec}.pdf") as pdf:
            
            # Figure 1: Cognitive outcomes
            if 'cognitive' in corrected_results:
                self._create_cognitive_results_figure(corrected_results['cognitive'])
                pdf.savefig(bbox_inches='tight', dpi=300)
                plt.close()
            
            # Figure 2: MRI outcomes (top significant)
            if 'mri' in corrected_results:
                self._create_mri_results_figure(corrected_results['mri'])
                pdf.savefig(bbox_inches='tight', dpi=300)
                plt.close()
            
            # Figure 3: Effect sizes comparison
            self._create_effect_size_figure(corrected_results)
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
            
            # Figure 4: Model comparison
            self._create_model_comparison_figure()
            pdf.savefig(bbox_inches='tight', dpi=300)
            plt.close()
    
    def _create_cognitive_results_figure(self, cognitive_results: pd.DataFrame):
        """Create comprehensive cognitive outcomes figure."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        sleep_vars = cognitive_results['Sleep_Variable'].unique()
        
        for idx, sleep_var in enumerate(sleep_vars):
            ax = axes[idx]
            
            data_subset = cognitive_results[cognitive_results['Sleep_Variable'] == sleep_var]
            
            # Create coefficient plot with confidence intervals
            y_pos = np.arange(len(data_subset))
            coeffs = data_subset['Coefficient'].values
            ci_lower = data_subset['CI_Lower'].values
            ci_upper = data_subset['CI_Upper'].values
            
            # Color code by significance
            colors = ['red' if sig else 'gray' for sig in data_subset['Significant_FDR']]
            
            ax.barh(y_pos, coeffs, color=colors, alpha=0.7)
            ax.errorbar(coeffs, y_pos, xerr=[coeffs - ci_lower, ci_upper - coeffs], 
                       fmt='none', color='black', capsize=3)
            
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(data_subset['Outcome'])
            ax.set_xlabel('Standardized Coefficient')
            ax.set_title(f'{sleep_var}', fontsize=14, fontweight='bold')
            
            # Add significance markers
            for i, (coeff, sig_fdr, sig_bonf) in enumerate(zip(
                coeffs, data_subset['Significant_FDR'], data_subset['Significant_Bonferroni']
            )):
                if sig_bonf:
                    ax.text(coeff + 0.01, i, '**', fontsize=12, va='center')
                elif sig_fdr:
                    ax.text(coeff + 0.01, i, '*', fontsize=12, va='center')
        
        plt.suptitle('Sleep Variables Effects on Cognitive Outcomes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Add legend
        red_patch = mpatches.Patch(color='red', alpha=0.7, label='Significant (FDR)')
        gray_patch = mpatches.Patch(color='gray', alpha=0.7, label='Non-significant')
        plt.figlegend(handles=[red_patch, gray_patch], loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    def _create_mri_results_figure(self, mri_results: pd.DataFrame):
        """Create MRI outcomes figure focusing on significant results."""
        # Get top significant results for each sleep variable
        significant_results = mri_results[mri_results['Significant_FDR'] == True]
        
        if len(significant_results) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No significant MRI findings after FDR correction', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('MRI Outcomes - Sleep Variable Effects')
            return
        
        # Get top results by effect size for each sleep variable
        top_results = []
        for sleep_var in significant_results['Sleep_Variable'].unique():
            var_results = significant_results[significant_results['Sleep_Variable'] == sleep_var]
            top_var = var_results.nlargest(5, 'Effect_Size')  # Top 5 by effect size
            top_results.append(top_var)
        
        plot_data = pd.concat(top_results)
        
        fig, axes = plt.subplots(1, len(plot_data['Sleep_Variable'].unique()), figsize=(18, 8))
        if len(plot_data['Sleep_Variable'].unique()) == 1:
            axes = [axes]
        
        for idx, sleep_var in enumerate(plot_data['Sleep_Variable'].unique()):
            ax = axes[idx]
            data_subset = plot_data[plot_data['Sleep_Variable'] == sleep_var]
            
            y_pos = np.arange(len(data_subset))
            coeffs = data_subset['Coefficient'].values
            ci_lower = data_subset['CI_Lower'].values
            ci_upper = data_subset['CI_Upper'].values
            
            # Color code by effect size
            effect_sizes = data_subset['Effect_Size'].values
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(effect_sizes)))
            
            bars = ax.barh(y_pos, coeffs, color=colors, alpha=0.8)
            ax.errorbar(coeffs, y_pos, xerr=[coeffs - ci_lower, ci_upper - coeffs], 
                       fmt='none', color='black', capsize=3)
            
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([self._format_roi_name(roi) for roi in data_subset['Outcome']])
            ax.set_xlabel('Standardized Coefficient')
            ax.set_title(f'{sleep_var}', fontsize=14, fontweight='bold')
            
            # Add effect size annotations
            for i, (coeff, effect) in enumerate(zip(coeffs, effect_sizes)):
                ax.text(coeff + 0.01, i, f'f²={effect:.3f}', fontsize=10, va='center')
        
        plt.suptitle('Sleep Variables Effects on MRI Outcomes (FDR Significant)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
    
    def _format_roi_name(self, roi_name: str) -> str:
        """Format ROI names for better readability."""
        # Remove Left/Right prefix and make readable
        formatted = roi_name.replace('Left', '').replace('Right', '')
        formatted = formatted.replace('Hippocampus', 'Hippocampus')
        formatted = formatted.replace('Entorhinal', 'Entorhinal Cortex')
        formatted = formatted.replace('MiddleTemporal', 'Middle Temporal')
        formatted = formatted.replace('InferiorTemporal', 'Inferior Temporal')
        return formatted.strip()
    
    def _create_effect_size_figure(self, corrected_results: Dict):
        """Create effect size comparison figure."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Combine all significant results
        all_significant = []
        for domain, df in corrected_results.items():
            if len(df) > 0:
                sig_results = df[df['Significant_FDR'] == True].copy()
                sig_results['Domain'] = domain.title()
                all_significant.append(sig_results)
        
        if all_significant:
            combined_df = pd.concat(all_significant)
            
            # Effect size distribution by domain
            sns.boxplot(data=combined_df, x='Domain', y='Effect_Size', ax=ax1)
            ax1.set_title('Effect Size Distribution by Domain', fontsize=14, fontweight='bold')
            ax1.set_ylabel("Cohen's f²")
            
            # Effect size by sleep variable
            sns.boxplot(data=combined_df, x='Sleep_Variable', y='Effect_Size', 
                       hue='Domain', ax=ax2)
            ax2.set_title('Effect Size by Sleep Variable', fontsize=14, fontweight='bold')
            ax2.set_ylabel("Cohen's f²")
            ax2.legend(title='Domain')
            
            # Add effect size interpretation lines
            for ax in [ax1, ax2]:
                ax.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='Small effect')
                ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
                ax.axhline(y=0.35, color='red', linestyle='--', alpha=0.5, label='Large effect')
        
        plt.tight_layout()
    
    def _create_model_comparison_figure(self):
        """Create model comparison figure (Model 1 vs Model 2)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Load both model results for comparison
        model1_results = self._load_model_results("model1")
        model2_results = self._load_model_results("model2")
        
        if not model1_results or not model2_results:
            fig.text(0.5, 0.5, 'Model comparison data not available', 
                    ha='center', va='center', fontsize=14)
            return
        
        # AIC comparison
        ax = axes[0, 0]
        aic_comparison = self._compare_model_fit(model1_results, model2_results, 'AIC')
        if len(aic_comparison) > 0:
            sns.scatterplot(data=aic_comparison, x='Model1_AIC', y='Model2_AIC', 
                           hue='Domain', ax=ax)
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], 
                   [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', alpha=0.5)
            ax.set_xlabel('Model 1 AIC')
            ax.set_ylabel('Model 2 AIC')
            ax.set_title('AIC Comparison (Lower is Better)')
        
        # BIC comparison
        ax = axes[0, 1]
        bic_comparison = self._compare_model_fit(model1_results, model2_results, 'BIC')
        if len(bic_comparison) > 0:
            sns.scatterplot(data=bic_comparison, x='Model1_BIC', y='Model2_BIC', 
                           hue='Domain', ax=ax)
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], 
                   [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', alpha=0.5)
            ax.set_xlabel('Model 1 BIC')
            ax.set_ylabel('Model 2 BIC')
            ax.set_title('BIC Comparison (Lower is Better)')
        
        # Coefficient comparison for sleep variables
        ax = axes[1, 0]
        coeff_comparison = self._compare_sleep_coefficients(model1_results, model2_results)
        if len(coeff_comparison) > 0:
            sns.scatterplot(data=coeff_comparison, x='Model1_Coeff', y='Model2_Coeff', 
                           hue='Sleep_Variable', style='Domain', ax=ax)
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], 
                   [ax.get_ylim()[0], ax.get_ylim()[1]], 'k--', alpha=0.5)
            ax.set_xlabel('Model 1 Coefficient')
            ax.set_ylabel('Model 2 Coefficient')
            ax.set_title('Sleep Variable Coefficient Comparison')
        
        # R² improvement (if available)
        ax = axes[1, 1]
        ax.text(0.5, 0.5, 'R² comparison\n(Implementation depends on\nmodel diagnostics)', 
               ha='center', va='center', fontsize=12)
        ax.set_title('Model Fit Improvement')
        
        plt.suptitle('Model Comparison: Sociodemographic vs. Full Model', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
    
    def _load_model_results(self, model_spec: str) -> Dict:
        """Load model results from saved files."""
        try:
            results = {'cognitive': [], 'mri': []}
            models_dir = self.output_dir / "models"
            
            for domain in ['cognitive', 'mri']:
                for sleep_var in self.sleep_vars:
                    filename = f"{domain}_outcomes_{sleep_var}_{model_spec}.pkl"
                    filepath = models_dir / filename
                    if filepath.exists():
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                            results[domain].extend(data)
            
            return results
        except Exception as e:
            logger.error(f"Error loading model results for {model_spec}: {e}")
            return {}
    
    def _compare_model_fit(self, model1_results: Dict, model2_results: Dict, metric: str) -> pd.DataFrame:
        """Compare model fit metrics between two models."""
        comparison_data = []
        
        for domain in ['cognitive', 'mri']:
            if not model1_results.get(domain) or not model2_results.get(domain):
                continue
                
            df1 = pd.DataFrame(model1_results[domain])
            df2 = pd.DataFrame(model2_results[domain])
            
            # Focus on sleep variable effects only
            df1_sleep = df1[df1['Feature'] == df1['Sleep_Variable']]
            df2_sleep = df2[df2['Feature'] == df2['Sleep_Variable']]
            
            # Merge on outcome and sleep variable
            merged = df1_sleep.merge(df2_sleep, on=['Outcome', 'Sleep_Variable'], 
                                   suffixes=('_Model1', '_Model2'))
            
            for _, row in merged.iterrows():
                comparison_data.append({
                    'Outcome': row['Outcome'],
                    'Sleep_Variable': row['Sleep_Variable'],
                    'Domain': domain.title(),
                    f'Model1_{metric}': row[f'{metric}_Model1'],
                    f'Model2_{metric}': row[f'{metric}_Model2']
                })
        
        return pd.DataFrame(comparison_data)
    
    def _compare_sleep_coefficients(self, model1_results: Dict, model2_results: Dict) -> pd.DataFrame:
        """Compare sleep variable coefficients between models."""
        comparison_data = []
        
        for domain in ['cognitive', 'mri']:
            if not model1_results.get(domain) or not model2_results.get(domain):
                continue
                
            df1 = pd.DataFrame(model1_results[domain])
            df2 = pd.DataFrame(model2_results[domain])
            
            # Focus on sleep variable effects only
            df1_sleep = df1[df1['Feature'] == df1['Sleep_Variable']]
            df2_sleep = df2[df2['Feature'] == df2['Sleep_Variable']]
            
            # Merge on outcome and sleep variable
            merged = df1_sleep.merge(df2_sleep, on=['Outcome', 'Sleep_Variable'], 
                                   suffixes=('_Model1', '_Model2'))
            
            for _, row in merged.iterrows():
                comparison_data.append({
                    'Outcome': row['Outcome'],
                    'Sleep_Variable': row['Sleep_Variable'],
                    'Domain': domain.title(),
                    'Model1_Coeff': row['Coefficient_Model1'],
                    'Model2_Coeff': row['Coefficient_Model2']
                })
        
        return pd.DataFrame(comparison_data)
    
    def create_publication_tables(self, model_spec: str = "model2"):
        """
        Create publication-ready tables for the manuscript.
        
        Parameters:
        -----------
        model_spec : str
            Model specification to tabulate
        """
        logger.info(f"Creating publication tables for {model_spec}")
        
        # Load corrected results
        corrected_results = self.perform_multiple_comparison_correction(model_spec)
        
        # Table 1: Sample characteristics
        self._create_sample_characteristics_table()
        
        # Table 2: Cognitive outcomes summary
        if 'cognitive' in corrected_results:
            self._create_cognitive_summary_table(corrected_results['cognitive'], model_spec)
        
        # Table 3: Significant MRI findings
        if 'mri' in corrected_results:
            self._create_mri_summary_table(corrected_results['mri'], model_spec)
        
        # Table 4: Model comparison statistics
        self._create_model_comparison_table()
        
        # Supplementary tables
        self._create_supplementary_tables(corrected_results, model_spec)
    
    def _create_sample_characteristics_table(self):
        """Create Table 1: Sample characteristics."""
        if self.raw_data is None:
            logger.warning("Raw data not available for sample characteristics table")
            return
        
        data = self.raw_data
        
        # Basic demographics
        total_subjects = data['RID'].nunique()
        total_observations = len(data)
        
        # Demographics by diagnosis
        demo_stats = []
        for dx in data['DX_bl'].unique():
            if pd.isna(dx):
                continue
                
            dx_data = data[data['DX_bl'] == dx]
            n_subjects = dx_data['RID'].nunique()
            
            # Get baseline data only
            baseline_data = dx_data.drop_duplicates('RID')
            
            age_mean = baseline_data['Adjusted_Age'].mean()
            age_std = baseline_data['Adjusted_Age'].std()
            
            education_mean = baseline_data['PTEDUCAT'].mean() if 'PTEDUCAT' in baseline_data else np.nan
            education_std = baseline_data['PTEDUCAT'].std() if 'PTEDUCAT' in baseline_data else np.nan
            
            female_pct = (baseline_data['PTGENDER'] == 'Female').mean() * 100 if 'PTGENDER' in baseline_data else np.nan
            
            demo_stats.append({
                'Diagnosis': dx,
                'N_Subjects': n_subjects,
                'Age_Mean_SD': f"{age_mean:.1f} ± {age_std:.1f}",
                'Education_Mean_SD': f"{education_mean:.1f} ± {education_std:.1f}" if not np.isnan(education_mean) else "N/A",
                'Female_Percent': f"{female_pct:.1f}%" if not np.isnan(female_pct) else "N/A",
                'Follow_up_visits': f"{dx_data.groupby('RID').size().mean():.1f}"
            })
        
        demo_df = pd.DataFrame(demo_stats)
        demo_df.to_csv(self.output_dir / "tables" / "table1_sample_characteristics.csv", index=False)
        
        # Sleep variable prevalence
        sleep_prevalence = []
        for sleep_var in ['Sleep_Apnea', 'Restless_Legs', 'Insomnia', 'Sleep_Disturbance_Other']:
            if sleep_var in data.columns:
                prevalence = (data[sleep_var] == 1).mean() * 100
                sleep_prevalence.append({
                    'Sleep_Variable': sleep_var,
                    'Prevalence_Percent': f"{prevalence:.1f}%"
                })
        
        sleep_df = pd.DataFrame(sleep_prevalence)
        sleep_df.to_csv(self.output_dir / "tables" / "table1_sleep_prevalence.csv", index=False)
    
    def _create_cognitive_summary_table(self, cognitive_results: pd.DataFrame, model_spec: str):
        """Create cognitive outcomes summary table."""
        # Format table with coefficients, CIs, and p-values
        summary_rows = []
        
        for outcome in self.cognitive_outcomes:
            row_data = {'Outcome': outcome}
            
            for sleep_var in self.sleep_vars:
                subset = cognitive_results[
                    (cognitive_results['Outcome'] == outcome) & 
                    (cognitive_results['Sleep_Variable'] == sleep_var)
                ]
                
                if len(subset) > 0:
                    row = subset.iloc[0]
                    coeff = row['Coefficient']
                    ci_lower = row['CI_Lower']
                    ci_upper = row['CI_Upper']
                    p_fdr = row['P_FDR']
                    
                    # Format with significance markers
                    sig_marker = ''
                    if row['Significant_Bonferroni']:
                        sig_marker = '**'
                    elif row['Significant_FDR']:
                        sig_marker = '*'
                    
                    formatted = f"{coeff:.3f} ({ci_lower:.3f}, {ci_upper:.3f}){sig_marker}"
                    row_data[sleep_var] = formatted
                else:
                    row_data[sleep_var] = "—"
            
            summary_rows.append(row_data)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            self.output_dir / "tables" / f"table2_cognitive_results_{model_spec}.csv", 
            index=False
        )
    
    def _create_mri_summary_table(self, mri_results: pd.DataFrame, model_spec: str):
        """Create MRI outcomes summary table (significant findings only)."""
        # Only include FDR-significant results
        significant_results = mri_results[mri_results['Significant_FDR'] == True]
        
        if len(significant_results) == 0:
            # Create empty table noting no significant findings
            no_sig_df = pd.DataFrame([{
                'Finding': 'No significant associations after FDR correction',
                'Details': 'All p-values > 0.05 after multiple comparison correction'
            }])
            no_sig_df.to_csv(
                self.output_dir / "tables" / f"table3_mri_results_{model_spec}.csv", 
                index=False
            )
            return
        
        # Format significant results
        summary_rows = []
        
        for _, row in significant_results.iterrows():
            coeff = row['Coefficient']
            ci_lower = row['CI_Lower']
            ci_upper = row['CI_Upper']
            p_fdr = row['P_FDR']
            effect_size = row['Effect_Size']
            
            # Significance marker
            sig_marker = '**' if row['Significant_Bonferroni'] else '*'
            
            summary_rows.append({
                'ROI': self._format_roi_name(row['Outcome']),
                'Sleep_Variable': row['Sleep_Variable'],
                'Coefficient': f"{coeff:.3f}{sig_marker}",
                'CI_95': f"({ci_lower:.3f}, {ci_upper:.3f})",
                'P_FDR': f"{p_fdr:.4f}",
                'Effect_Size': f"{effect_size:.4f}",
                'Direction': 'Positive' if coeff > 0 else 'Negative'
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(['Sleep_Variable', 'Effect_Size'], ascending=[True, False])
        summary_df.to_csv(
            self.output_dir / "tables" / f"table3_mri_results_{model_spec}.csv", 
            index=False
        )
    
    def _create_model_comparison_table(self):
        """Create model comparison statistics table."""
        model1_results = self._load_model_results("model1")
        model2_results = self._load_model_results("model2")
        
        if not model1_results or not model2_results:
            logger.warning("Model results not available for comparison table")
            return
        
        comparison_stats = []
        
        for domain in ['cognitive', 'mri']:
            if not model1_results.get(domain) or not model2_results.get(domain):
                continue
            
            df1 = pd.DataFrame(model1_results[domain])
            df2 = pd.DataFrame(model2_results[domain])
            
            # Get sleep variable effects only
            df1_sleep = df1[df1['Feature'] == df1['Sleep_Variable']]
            df2_sleep = df2[df2['Feature'] == df2['Sleep_Variable']]
            
            # Calculate summary statistics
            n_tests = len(df1_sleep)
            
            # AIC/BIC improvement
            merged = df1_sleep.merge(df2_sleep, on=['Outcome', 'Sleep_Variable'], suffixes=('_M1', '_M2'))
            aic_improved = (merged['AIC_M2'] < merged['AIC_M1']).sum()
            bic_improved = (merged['BIC_M2'] < merged['BIC_M1']).sum()
            
            # Mean AIC/BIC difference
            aic_diff_mean = (merged['AIC_M1'] - merged['AIC_M2']).mean()
            bic_diff_mean = (merged['BIC_M1'] - merged['BIC_M2']).mean()
            
            comparison_stats.append({
                'Domain': domain.title(),
                'N_Tests': n_tests,
                'AIC_Improved_N': f"{aic_improved}/{n_tests}",
                'AIC_Improved_Pct': f"{(aic_improved/n_tests)*100:.1f}%",
                'Mean_AIC_Improvement': f"{aic_diff_mean:.2f}",
                'BIC_Improved_N': f"{bic_improved}/{n_tests}",
                'BIC_Improved_Pct': f"{(bic_improved/n_tests)*100:.1f}%",
                'Mean_BIC_Improvement': f"{bic_diff_mean:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_stats)
        comparison_df.to_csv(
            self.output_dir / "tables" / "table4_model_comparison.csv", 
            index=False
        )
    
    def _create_supplementary_tables(self, corrected_results: Dict, model_spec: str):
        """Create supplementary tables."""
        # Supplementary Table 1: All results (uncorrected)
        for domain in ['cognitive', 'mri']:
            if domain in corrected_results:
                df = corrected_results[domain].copy()
                # Include all results, not just significant ones
                df_formatted = df[[
                    'Outcome', 'Sleep_Variable', 'Coefficient', 'SE', 'P_Value', 
                    'P_FDR', 'P_Bonferroni', 'Effect_Size', 'CI_Lower', 'CI_Upper'
                ]].round(6)
                
                df_formatted.to_csv(
                    self.output_dir / "tables" / f"supplementary_{domain}_all_results_{model_spec}.csv",
                    index=False
                )
        
        # Supplementary Table 2: Residual diagnostics
        try:
            for domain in ['cognitive', 'mri']:
                diag_file = self.output_dir / "diagnostics" / f"{domain}_residual_diagnostics_{model_spec}.csv"
                if diag_file.exists():
                    diag_df = pd.read_csv(diag_file)
                    # Copy to supplementary tables with better name
                    diag_df.to_csv(
                        self.output_dir / "tables" / f"supplementary_{domain}_diagnostics_{model_spec}.csv",
                        index=False
                    )
        except Exception as e:
            logger.warning(f"Could not create supplementary diagnostics tables: {e}")
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report."""
        logger.info("Generating comprehensive analysis report")
        
        report = []
        report.append("# Sleep Variables and Cognitive/MRI Outcomes Analysis Report")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data summary
        if self.raw_data is not None:
            report.append("## Data Summary")
            report.append(f"- Total subjects: {self.raw_data['RID'].nunique()}")
            report.append(f"- Total observations: {len(self.raw_data)}")
            report.append(f"- Mean visits per subject: {self.raw_data.groupby('RID').size().mean():.1f}")
            report.append("")
        
        # Model specifications
        report.append("## Model Specifications")
        report.append("### Model 1 (Sociodemographic):")
        report.append("- Predictors: " + ", ".join(self.model1_predictors))
        report.append("### Model 2 (Full):")
        report.append("- Additional predictors: APOE4, BMI, Medical History")
        report.append("")
        
        # Results summary
        try:
            corrected_model2 = self.perform_multiple_comparison_correction("model2")
            
            for domain in ['cognitive', 'mri']:
                if domain in corrected_model2:
                    df = corrected_model2[domain]
                    n_total = len(df)
                    n_sig_fdr = df['Significant_FDR'].sum()
                    n_sig_bonf = df['Significant_Bonferroni'].sum()
                    
                    report.append(f"## {domain.title()} Results")
                    report.append(f"- Total tests: {n_total}")
                    report.append(f"- Significant (FDR): {n_sig_fdr} ({n_sig_fdr/n_total*100:.1f}%)")
                    report.append(f"- Significant (Bonferroni): {n_sig_bonf} ({n_sig_bonf/n_total*100:.1f}%)")
                    
                    if n_sig_fdr > 0:
                        sig_df = df[df['Significant_FDR'] == True]
                        sleep_var_counts = sig_df['Sleep_Variable'].value_counts()
                        report.append("- Significant findings by sleep variable:")
                        for var, count in sleep_var_counts.items():
                            report.append(f"  - {var}: {count}")
                    
                    report.append("")
            
        except Exception as e:
            report.append(f"## Results Summary")
            report.append(f"Error generating results summary: {e}")
            report.append("")
        
        # File outputs
        report.append("## Generated Files")
        report.append("### Tables:")
        for table_file in (self.output_dir / "tables").glob("*.csv"):
            report.append(f"- {table_file.name}")
        
        report.append("### Figures:")
        for fig_file in (self.output_dir / "figures").glob("*"):
            report.append(f"- {fig_file.name}")
        
        report.append("### Model Files:")
        for model_file in (self.output_dir / "models").glob("*.pkl"):
            report.append(f"- {model_file.name}")
        
        # Save report
        with open(self.output_dir / "analysis_report.md", 'w') as f:
            f.write('\n'.join(report))
        
        logger.info("Analysis report generated successfully")
    
    def run_complete_analysis(self, data_path: str):
        """
        Run the complete analysis pipeline from data loading to report generation.
        
        Parameters:
        -----------
        data_path : str
            Path to the input CSV file
        """
        logger.info("Starting complete analysis pipeline")
        
        try:
            # Step 1: Load and validate data
            raw_data = self.load_and_validate_data(data_path)
            
            # Step 2: Preprocess data
            processed_data = self.preprocess_data(raw_data)
            
            # Step 3: Fit Model 1 (Sociodemographic)
            logger.info("Fitting Model 1 (Sociodemographic controls)")
            self.fit_linear_mixed_models(processed_data, "model1")
            
            # Step 4: Fit Model 2 (Full model)
            logger.info("Fitting Model 2 (Full model with APOE4, BMI, Medical History)")
            self.fit_linear_mixed_models(processed_data, "model2")
            
            # Step 5: Residual diagnostics
            logger.info("Performing residual diagnostics")
            self.perform_residual_diagnostics(processed_data, "model1")
            self.perform_residual_diagnostics(processed_data, "model2")
            
            # Step 6: Multiple comparison corrections
            logger.info("Applying multiple comparison corrections")
            self.perform_multiple_comparison_correction("model1")
            self.perform_multiple_comparison_correction("model2")
            
            # Step 7: Create publication figures
            logger.info("Creating publication-ready figures")
            self.create_publication_figures("model1")
            self.create_publication_figures("model2")
            
            # Step 8: Create publication tables
            logger.info("Creating publication-ready tables")
            self.create_publication_tables("model1")
            self.create_publication_tables("model2")
            
            # Step 9: Generate comprehensive report
            logger.info("Generating analysis report")
            self.generate_analysis_report()
            
            logger.info("Complete analysis pipeline finished successfully")
            logger.info(f"All outputs saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Research-grade Sleep-Cognition Analysis Pipeline"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize analyzer
    analyzer = SleepCognitionAnalyzer(output_dir=args.output_dir)
    
    # Run complete analysis
    analyzer.run_complete_analysis(args.data_path)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("📊 Tables: tables/")
    print("📈 Figures: figures/")
    print("🧪 Models: models/")
    print("🔍 Diagnostics: diagnostics/")
    print("📄 Report: analysis_report.md")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()


# Additional utility functions for enhanced analysis

def calculate_statistical_power(effect_size: float, sample_size: int, alpha: float = 0.05) -> float:
    """
    Calculate statistical power for a given effect size and sample size.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's f² effect size
    sample_size : int
        Sample size
    alpha : float
        Type I error rate (default: 0.05)
        
    Returns:
    --------
    float
        Statistical power (1 - β)
    """
    try:
        from scipy.stats import f
        # Convert Cohen's f² to Cohen's f
        cohens_f = np.sqrt(effect_size)
        
        # Degrees of freedom (simplified - assumes 1 predictor)
        df1 = 1
        df2 = sample_size - 2
        
        # Critical F value
        f_crit = f.ppf(1 - alpha, df1, df2)
        
        # Non-centrality parameter
        ncp = sample_size * cohens_f**2
        
        # Power calculation (simplified)
        power = 1 - f.cdf(f_crit, df1, df2, ncp)
        
        return power
        
    except Exception:
        return np.nan

def cohens_conventions(effect_size: float) -> str:
    """
    Interpret Cohen's f² effect size according to conventional guidelines.
    
    Parameters:
    -----------
    effect_size : float
        Cohen's f² effect size
        
    Returns:
    --------
    str
        Effect size interpretation
    """
    if np.isnan(effect_size):
        return "Unknown"
    elif effect_size < 0.02:
        return "Very Small"
    elif effect_size < 0.15:
        return "Small"
    elif effect_size < 0.35:
        return "Medium"
    else:
        return "Large"

class EnhancedModelDiagnostics:
    """
    Enhanced model diagnostics for research-grade analysis.
    """
    
    @staticmethod
    def check_linearity(model_result, outcome_data: np.ndarray, predictor_data: np.ndarray) -> Dict:
        """Check linearity assumption using residual plots."""
        try:
            residuals = model_result.resid
            fitted = model_result.fittedvalues
            
            # Correlation between residuals and fitted values
            corr_res_fitted, p_corr = pearsonr(residuals, fitted)
            
            # Runs test for randomness
            def runs_test(residuals):
                """Simple runs test for residual randomness."""
                median_res = np.median(residuals)
                runs = 1
                for i in range(1, len(residuals)):
                    if (residuals[i] > median_res) != (residuals[i-1] > median_res):
                        runs += 1
                
                n1 = np.sum(residuals > median_res)
                n2 = np.sum(residuals <= median_res)
                
                if n1 == 0 or n2 == 0:
                    return np.nan, np.nan
                
                expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
                
                if var_runs <= 0:
                    return np.nan, np.nan
                
                z_score = (runs - expected_runs) / np.sqrt(var_runs)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                return z_score, p_value
            
            runs_z, runs_p = runs_test(residuals)
            
            return {
                'residual_fitted_correlation': corr_res_fitted,
                'residual_fitted_p': p_corr,
                'runs_test_z': runs_z,
                'runs_test_p': runs_p,
                'linearity_assumption': 'Met' if abs(corr_res_fitted) < 0.1 and p_corr > 0.05 else 'Questionable'
            }
            
        except Exception as e:
            return {
                'residual_fitted_correlation': np.nan,
                'residual_fitted_p': np.nan,
                'runs_test_z': np.nan,
                'runs_test_p': np.nan,
                'linearity_assumption': 'Error',
                'error': str(e)
            }
    
    @staticmethod
    def check_homoscedasticity(model_result) -> Dict:
        """Check homoscedasticity using Breusch-Pagan test."""
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            
            residuals = model_result.resid
            fitted = model_result.fittedvalues
            
            # Breusch-Pagan test
            bp_stat, bp_p, f_stat, f_p = het_breuschpagan(residuals, fitted.reshape(-1, 1))
            
            return {
                'breusch_pagan_stat': bp_stat,
                'breusch_pagan_p': bp_p,
                'homoscedasticity_assumption': 'Met' if bp_p > 0.05 else 'Violated'
            }
            
        except Exception as e:
            return {
                'breusch_pagan_stat': np.nan,
                'breusch_pagan_p': np.nan,
                'homoscedasticity_assumption': 'Error',
                'error': str(e)
            }
    
    @staticmethod
    def detect_outliers(model_result, threshold: float = 3.0) -> Dict:
        """Detect outliers using standardized residuals."""
        try:
            residuals = model_result.resid
            standardized_residuals = residuals / np.std(residuals)
            
            outlier_indices = np.where(np.abs(standardized_residuals) > threshold)[0]
            n_outliers = len(outlier_indices)
            outlier_percentage = (n_outliers / len(residuals)) * 100
            
            return {
                'n_outliers': n_outliers,
                'outlier_percentage': outlier_percentage,
                'outlier_indices': outlier_indices.tolist(),
                'max_abs_residual': np.max(np.abs(standardized_residuals)),
                'outlier_concern': 'High' if outlier_percentage > 5 else 'Low'
            }
            
        except Exception as e:
            return {
                'n_outliers': np.nan,
                'outlier_percentage': np.nan,
                'outlier_indices': [],
                'max_abs_residual': np.nan,
                'outlier_concern': 'Error',
                'error': str(e)
            }

# Configuration for different analysis scenarios
ANALYSIS_CONFIGS = {
    'standard': {
        'alpha_level': 0.05,
        'multiple_correction': 'fdr_bh',
        'effect_size_threshold': 0.02,
        'outlier_threshold': 3.0
    },
    'conservative': {
        'alpha_level': 0.01,
        'multiple_correction': 'bonferroni',
        'effect_size_threshold': 0.15,
        'outlier_threshold': 2.5
    },
    'exploratory': {
        'alpha_level': 0.10,
        'multiple_correction': 'fdr_bh',
        'effect_size_threshold': 0.01,
        'outlier_threshold': 3.5
    }
}

# Example usage and documentation
"""
USAGE EXAMPLES:

1. Basic usage:
   python sleep_cognition_pipeline.py --data_path "Latest_all_in_one.csv"

2. Custom output directory:
   python sleep_cognition_pipeline.py --data_path "data.csv" --output_dir "my_results"

3. With verbose logging:
   python sleep_cognition_pipeline.py --data_path "data.csv" --log_level DEBUG

4. Programmatic usage:
   from sleep_cognition_pipeline import SleepCognitionAnalyzer
   
   analyzer = SleepCognitionAnalyzer(output_dir="results")
   analyzer.run_complete_analysis("data.csv")

OUTPUTS:
- tables/: CSV files with publication-ready tables
- figures/: Publication-ready figures in PDF format
- models/: Pickled model results for further analysis
- diagnostics/: Residual diagnostics and assumption tests
- analysis_report.md: Comprehensive analysis report

KEY FEATURES:
- Linear Mixed Models with proper random effects
- Multiple comparison corrections (FDR and Bonferroni)
- Effect size calculations (Cohen's f²)
- Comprehensive residual diagnostics
- Publication-ready figures and tables
- Model comparison statistics
- Automated report generation
"""