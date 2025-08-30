#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Sleep-Cognition Analysis Pipeline for Research Manuscripts
================================================================

Comprehensive research pipeline integrating longitudinal, cross-sectional, and correlation
analyses for examining relationships between sleep variables and cognitive/neuroimaging outcomes.

Features:
- Longitudinal Linear Mixed Models (LMMs) with interaction effects
- Cross-sectional LASSO and significance testing
- Correlation analysis across diagnostic groups
- Granger causality testing
- Publication-ready figures and tables
- Comprehensive statistical reporting

Authors: Research Team
Date: August 29, 2025
Version: 3.0 (Unified Research Grade)
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
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import patsy

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# Configure warnings and logging
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Publication-ready plotting parameters
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'Arial',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.alpha': 0.3
})

# Color palettes for research figures
RESEARCH_COLORS = {
    'cognitive': ['#2E86AB', '#A23B72', '#F18F01'],
    'mri': ['#C73E1D', '#F95738', '#FFB627'],
    'sleep': ['#4A5D23', '#87A96B', '#C5D86D'],
    'groups': {'CN': '#2E86AB', 'MCI': '#F18F01', 'AD': '#C73E1D'}
}


class BaseAnalyzer:
    """Base class with common functionality for all analyzers."""
    
    def __init__(self, data: pd.DataFrame, output_dir: str):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define variable sets
        self.cognitive_outcomes = ['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS', 'ADNI_EF2']
        self.sleep_vars = ['NPIK', 'NPIKSEV', 'MHSleep']
        
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
        
        self.sociodemographic_vars = [
            'Adjusted_Age', 'PTEDUCAT', 'PTETHCAT_2', 'PTGENDER_1', 
            'PTMARRY', 'PTRACCAT_1', 'APOE4', 'BMI', 'MH'
        ]
        
        # Model predictors
        self.base_predictors = [
            'Adjusted_Age', 'DX_1', 'DX_2', 'PTEDUCAT',
            'PTETHCAT_2', 'PTGENDER_1', 'PTMARRY', 'PTRACCAT_1'
        ]
        
        self.full_predictors = self.base_predictors + ['APOE4', 'BMI', 'MH']
        
    def preprocess_data(self):
        """Common data preprocessing steps."""
        logger.info("Starting data preprocessing...")
        
        # Create composite sleep variable
        sleep_components = ['Sleep_Apnea', 'Restless_Legs', 'Insomnia', 'Sleep_Disturbance_Other']
        if all(col in self.data.columns for col in sleep_components):
            self.data['MHSleep'] = self.data[sleep_components].sum(axis=1)
        
        # Label encoding for categorical variables
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for column in categorical_cols:
            if column not in ['RID']:
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column].astype(str))
        
        # Create dummy variables for diagnosis and other categoricals
        categorical_vars = ['DX', 'DX_bl', 'PTETHCAT', 'PTGENDER', 'PTRACCAT']
        available_categorical = [var for var in categorical_vars if var in self.data.columns]
        self.data = pd.get_dummies(self.data, columns=available_categorical, drop_first=True)
        
        return self.data
    
    def format_roi_name(self, roi_name: str) -> str:
        """Format ROI names for publication quality."""
        formatted = roi_name.replace('Right', 'R.').replace('Left', 'L.')
        formatted = formatted.replace('Hippocampus', 'Hippocampus')
        formatted = formatted.replace('Entorhinal', 'Entorhinal')
        formatted = formatted.replace('MiddleTemporal', 'Mid. Temporal')
        formatted = formatted.replace('InferiorTemporal', 'Inf. Temporal')
        formatted = formatted.replace('InferiorParietal', 'Inf. Parietal')
        formatted = formatted.replace('MedialOrbitofrontal', 'Med. OFC')
        return formatted.strip()
    
    def save_results_table(self, df: pd.DataFrame, filename: str):
        """Save results table with proper formatting."""
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        df.to_csv(tables_dir / filename, index=False)
        logger.info(f"Saved table: {filename}")


class LongitudinalAnalyzer(BaseAnalyzer):
    """Longitudinal analysis using Linear Mixed Models with interaction effects."""
    
    def __init__(self, data: pd.DataFrame, output_dir: str):
        super().__init__(data, output_dir)
        self.results = {'base_model': {}, 'full_model': {}, 'interaction_model': {}}
        
    def fit_mixed_models(self, include_interactions: bool = True):
        """
        Fit Linear Mixed Models with optional interaction effects.
        
        Parameters:
        -----------
        include_interactions : bool
            Whether to include sleep × diagnosis interactions
        """
        logger.info("Fitting Linear Mixed Models...")
        
        # Create subdirectories
        (self.output_dir / "longitudinal").mkdir(exist_ok=True)
        
        # Fit base model (sociodemographic controls)
        self._fit_model_set('base_model', self.base_predictors)
        
        # Fit full model (with APOE4, BMI, medical history)
        self._fit_model_set('full_model', self.full_predictors)
        
        # Fit interaction model if requested
        if include_interactions:
            self._fit_interaction_models()
        
        # Apply multiple comparison corrections
        self._apply_corrections()
        
        return self
    
    def _fit_model_set(self, model_name: str, predictors: List[str]):
        """Fit a complete set of models for given predictors."""
        available_predictors = [p for p in predictors if p in self.data.columns]
        
        self.results[model_name] = {'cognitive': [], 'mri': []}
        
        # Cognitive outcomes
        for sleep_var in self.sleep_vars:
            if sleep_var not in self.data.columns:
                continue
                
            for outcome in self.cognitive_outcomes:
                if outcome not in self.data.columns:
                    continue
                    
                try:
                    formula = f"{outcome} ~ {sleep_var} + {' + '.join(available_predictors)}"
                    model = smf.mixedlm(formula, data=self.data, groups=self.data["RID"])
                    result = model.fit(reml=False)
                    
                    model_results = self._extract_model_results(result, outcome, sleep_var, model_name)
                    self.results[model_name]['cognitive'].extend(model_results)
                    
                except Exception as e:
                    logger.warning(f"Model fitting failed: {outcome} ~ {sleep_var} ({e})")
        
        # MRI outcomes
        for sleep_var in self.sleep_vars:
            if sleep_var not in self.data.columns:
                continue
                
            for outcome in self.mri_outcomes:
                if outcome not in self.data.columns:
                    continue
                    
                try:
                    formula = f"{outcome} ~ {sleep_var} + {' + '.join(available_predictors)}"
                    model = smf.mixedlm(formula, data=self.data, groups=self.data["RID"])
                    result = model.fit(reml=False)
                    
                    model_results = self._extract_model_results(result, outcome, sleep_var, model_name)
                    self.results[model_name]['mri'].extend(model_results)
                    
                except Exception as e:
                    logger.warning(f"Model fitting failed: {outcome} ~ {sleep_var} ({e})")
        
        logger.info(f"Completed {model_name} fitting")
    
    def _fit_interaction_models(self):
        """Fit models with sleep × diagnosis interactions."""
        logger.info("Fitting interaction models...")
        
        self.results['interaction_model'] = {'cognitive': [], 'mri': []}
        
        # Check for diagnosis variables
        dx_vars = [col for col in self.data.columns if col.startswith('DX_')]
        if not dx_vars:
            logger.warning("No diagnosis variables found for interaction models")
            return
        
        # Cognitive outcomes with interactions
        for sleep_var in self.sleep_vars:
            if sleep_var not in self.data.columns:
                continue
                
            for outcome in self.cognitive_outcomes:
                if outcome not in self.data.columns:
                    continue
                    
                try:
                    # Create interaction terms
                    interaction_terms = [f"{sleep_var}:{dx_var}" for dx_var in dx_vars]
                    
                    formula = (f"{outcome} ~ {sleep_var} + {' + '.join(self.full_predictors)} + "
                             f"{' + '.join(interaction_terms)}")
                    
                    model = smf.mixedlm(formula, data=self.data, groups=self.data["RID"])
                    result = model.fit(reml=False)
                    
                    model_results = self._extract_model_results(result, outcome, sleep_var, 'interaction_model')
                    self.results['interaction_model']['cognitive'].extend(model_results)
                    
                except Exception as e:
                    logger.warning(f"Interaction model failed: {outcome} ~ {sleep_var} ({e})")
        
        # MRI outcomes with interactions
        for sleep_var in self.sleep_vars:
            if sleep_var not in self.data.columns:
                continue
                
            for outcome in self.mri_outcomes:
                if outcome not in self.data.columns:
                    continue
                    
                try:
                    interaction_terms = [f"{sleep_var}:{dx_var}" for dx_var in dx_vars]
                    
                    formula = (f"{outcome} ~ {sleep_var} + {' + '.join(self.full_predictors)} + "
                             f"{' + '.join(interaction_terms)}")
                    
                    model = smf.mixedlm(formula, data=self.data, groups=self.data["RID"])
                    result = model.fit(reml=False)
                    
                    model_results = self._extract_model_results(result, outcome, sleep_var, 'interaction_model')
                    self.results['interaction_model']['mri'].extend(model_results)
                    
                except Exception as e:
                    logger.warning(f"Interaction model failed: {outcome} ~ {sleep_var} ({e})")
        
        logger.info("Interaction models completed")
    
    def _extract_model_results(self, result, outcome: str, sleep_var: str, model_type: str) -> List[Dict]:
        """Extract comprehensive results from fitted model."""
        records = []
        
        for predictor in result.params.index:
            try:
                # Calculate effect size (partial eta-squared approximation)
                t_stat = result.tvalues.get(predictor, np.nan)
                df_resid = result.df_resid if hasattr(result, 'df_resid') else len(result.resid)
                
                if not np.isnan(t_stat) and df_resid > 0:
                    partial_eta_sq = (t_stat**2) / (t_stat**2 + df_resid)
                else:
                    partial_eta_sq = np.nan
                
                # Confidence intervals
                try:
                    conf_int = result.conf_int().loc[predictor]
                except:
                    conf_int = [np.nan, np.nan]
                
                records.append({
                    'Model_Type': model_type,
                    'Outcome': outcome,
                    'Sleep_Variable': sleep_var,
                    'Predictor': predictor,
                    'Coefficient': result.params.get(predictor, np.nan),
                    'SE': result.bse.get(predictor, np.nan),
                    'T_Statistic': t_stat,
                    'P_Value': result.pvalues.get(predictor, np.nan),
                    'CI_Lower': conf_int[0],
                    'CI_Upper': conf_int[1],
                    'Effect_Size': partial_eta_sq,
                    'AIC': result.aic,
                    'BIC': result.bic,
                    'Domain': 'Cognitive' if outcome in self.cognitive_outcomes else 'MRI'
                })
                
            except Exception as e:
                logger.warning(f"Error extracting results for {predictor}: {e}")
        
        return records
    
    def _apply_corrections(self):
        """Apply FDR correction to all sleep variable effects."""
        logger.info("Applying multiple comparison corrections...")
        
        for model_type in self.results:
            for domain in ['cognitive', 'mri']:
                if not self.results[model_type][domain]:
                    continue
                
                df = pd.DataFrame(self.results[model_type][domain])
                
                # Focus on sleep variable main effects and interactions
                sleep_effects = df[
                    (df['Predictor'].isin(self.sleep_vars)) |
                    (df['Predictor'].str.contains(':'))
                ].copy()
                
                if len(sleep_effects) > 0:
                    valid_p = sleep_effects['P_Value'].dropna()
                    if len(valid_p) > 0:
                        # FDR correction
                        fdr_rejected, fdr_pvals, _, _ = multipletests(
                            valid_p, alpha=0.05, method='fdr_bh'
                        )
                        
                        sleep_effects = sleep_effects.dropna(subset=['P_Value']).copy()
                        sleep_effects['P_FDR'] = fdr_pvals
                        sleep_effects['Significant_FDR'] = fdr_rejected
                        
                        # Update original results
                        self.results[model_type][domain] = sleep_effects.to_dict('records')
    
    def create_longitudinal_figures(self):
        """Create publication-ready figures for longitudinal analysis."""
        logger.info("Creating longitudinal analysis figures...")
        
        fig_dir = self.output_dir / "figures" / "longitudinal"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Figure 1: Main effects forest plot
        self._create_forest_plot()
        
        # Figure 2: Interaction effects heatmap
        self._create_interaction_heatmap()
        
        # Figure 3: Model comparison
        self._create_model_comparison_plot()
        
        return self
    
    def _create_forest_plot(self):
        """Create forest plot for main sleep variable effects."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot cognitive outcomes - base model
        self._plot_domain_forest(ax1, 'cognitive', 'base_model', 'Cognitive Outcomes (Base Model)')
        
        # Plot cognitive outcomes - full model
        self._plot_domain_forest(ax2, 'cognitive', 'full_model', 'Cognitive Outcomes (Full Model)')
        
        # Plot MRI outcomes - base model (top 15)
        self._plot_domain_forest(ax3, 'mri', 'base_model', 'MRI Outcomes (Base Model)', top_n=15)
        
        # Plot MRI outcomes - full model (top 15)
        self._plot_domain_forest(ax4, 'mri', 'full_model', 'MRI Outcomes (Full Model)', top_n=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "longitudinal" / "forest_plot_main_effects.png")
        plt.close()
    
    def _plot_domain_forest(self, ax, domain: str, model_type: str, title: str, top_n: int = None):
        """Plot forest plot for a specific domain and model."""
        if not self.results[model_type][domain]:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title)
            return
        
        df = pd.DataFrame(self.results[model_type][domain])
        
        # Filter for sleep variable main effects
        sleep_main_effects = df[df['Predictor'].isin(self.sleep_vars)].copy()
        
        if len(sleep_main_effects) == 0:
            ax.text(0.5, 0.5, 'No sleep effects found', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Select top effects if specified
        if top_n and len(sleep_main_effects) > top_n:
            sleep_main_effects = sleep_main_effects.nlargest(top_n, 'Effect_Size')
        
        # Sort by effect size
        sleep_main_effects = sleep_main_effects.sort_values('Effect_Size', ascending=True)
        
        # Create forest plot
        y_pos = np.arange(len(sleep_main_effects))
        coeffs = sleep_main_effects['Coefficient'].values
        ci_lower = sleep_main_effects['CI_Lower'].values
        ci_upper = sleep_main_effects['CI_Upper'].values
        
        # Color by significance and sleep variable
        colors = []
        for _, row in sleep_main_effects.iterrows():
            if row.get('Significant_FDR', False):
                if row['Sleep_Variable'] == 'NPIK':
                    colors.append(RESEARCH_COLORS['sleep'][0])
                elif row['Sleep_Variable'] == 'NPIKSEV':
                    colors.append(RESEARCH_COLORS['sleep'][1])
                else:
                    colors.append(RESEARCH_COLORS['sleep'][2])
            else:
                colors.append('lightgray')
        
        # Plot bars and error bars
        bars = ax.barh(y_pos, coeffs, color=colors, alpha=0.7, height=0.6)
        ax.errorbar(coeffs, y_pos, xerr=[coeffs - ci_lower, ci_upper - coeffs], 
                   fmt='none', color='black', capsize=3, linewidth=1.2)
        
        # Formatting
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        ax.set_yticks(y_pos)
        
        # Format y-axis labels
        if domain == 'mri':
            y_labels = [f"{self.format_roi_name(row['Outcome'])} ({row['Sleep_Variable']})" 
                       for _, row in sleep_main_effects.iterrows()]
        else:
            y_labels = [f"{row['Outcome']} ({row['Sleep_Variable']})" 
                       for _, row in sleep_main_effects.iterrows()]
        
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel('Standardized Coefficient (95% CI)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add significance indicators
        for i, (coeff, sig_fdr) in enumerate(zip(coeffs, sleep_main_effects.get('Significant_FDR', [False]*len(coeffs)))):
            if sig_fdr:
                ax.text(coeff + max(abs(coeffs))*0.02, i, '*', fontsize=14, va='center', 
                       fontweight='bold', color='red')
    
    def _create_interaction_heatmap(self):
        """Create heatmap showing interaction effects."""
        if 'interaction_model' not in self.results or not self.results['interaction_model']:
            logger.info("No interaction results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        for i, domain in enumerate(['cognitive', 'mri']):
            ax = ax1 if i == 0 else ax2
            
            if not self.results['interaction_model'][domain]:
                ax.text(0.5, 0.5, 'No interaction data', ha='center', va='center')
                ax.set_title(f'{domain.title()} Interactions')
                continue
            
            df = pd.DataFrame(self.results['interaction_model'][domain])
            
            # Filter for interaction terms
            interactions = df[df['Predictor'].str.contains(':')].copy()
            
            if len(interactions) == 0:
                ax.text(0.5, 0.5, 'No significant interactions', ha='center', va='center')
                ax.set_title(f'{domain.title()} Interactions')
                continue
            
            # Create interaction matrix
            interaction_matrix = []
            for _, row in interactions.iterrows():
                parts = row['Predictor'].split(':')
                if len(parts) == 2:
                    interaction_matrix.append({
                        'Sleep_Var': parts[0],
                        'Diagnosis': parts[1],
                        'Outcome': row['Outcome'],
                        'Coefficient': row['Coefficient'],
                        'P_Value': row['P_Value'],
                        'Significant': row.get('Significant_FDR', False)
                    })
            
            if interaction_matrix:
                int_df = pd.DataFrame(interaction_matrix)
                
                # Create pivot for heatmap
                if domain == 'mri' and len(int_df) > 20:
                    # Show top 20 for MRI
                    int_df = int_df.nlargest(20, 'Coefficient', keep='all')
                
                pivot_data = int_df.pivot_table(
                    index='Outcome', 
                    columns=['Sleep_Var', 'Diagnosis'], 
                    values='Coefficient', 
                    aggfunc='first'
                )
                
                # Plot heatmap
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                           ax=ax, cbar_kws={'label': 'Interaction Coefficient'})
                ax.set_title(f'{domain.title()} Sleep × Diagnosis Interactions', 
                           fontsize=13, fontweight='bold')
                ax.set_xlabel('Sleep Variable × Diagnosis')
                
                if domain == 'mri':
                    y_labels = [self.format_roi_name(label) for label in ax.get_yticklabels()]
                    ax.set_yticklabels(y_labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "longitudinal" / "interaction_heatmap.png")
        plt.close()
    
    def _create_model_comparison_plot(self):
        """Create model comparison visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # AIC comparison: Base vs Full
        self._plot_model_fit_comparison(ax1, 'AIC', 'base_model', 'full_model', 'AIC: Base vs Full Model')
        
        # BIC comparison: Base vs Full
        self._plot_model_fit_comparison(ax2, 'BIC', 'base_model', 'full_model', 'BIC: Base vs Full Model')
        
        # AIC comparison: Full vs Interaction
        self._plot_model_fit_comparison(ax3, 'AIC', 'full_model', 'interaction_model', 'AIC: Full vs Interaction Model')
        
        # Effect size comparison across models
        self._plot_effect_size_comparison(ax4)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "longitudinal" / "model_comparison.png")
        plt.close()
    
    def _plot_model_fit_comparison(self, ax, metric: str, model1: str, model2: str, title: str):
        """Plot model fit comparison."""
        comparison_data = []
        
        for domain in ['cognitive', 'mri']:
            if not self.results[model1].get(domain) or not self.results[model2].get(domain):
                continue
            
            df1 = pd.DataFrame(self.results[model1][domain])
            df2 = pd.DataFrame(self.results[model2][domain])
            
            # Focus on sleep main effects
            df1_sleep = df1[df1['Predictor'].isin(self.sleep_vars)]
            df2_sleep = df2[df2['Predictor'].isin(self.sleep_vars)]
            
            # Merge on outcome and sleep variable
            merged = df1_sleep.merge(
                df2_sleep, 
                on=['Outcome', 'Sleep_Variable'], 
                suffixes=('_1', '_2')
            )
            
            for _, row in merged.iterrows():
                comparison_data.append({
                    'Model1_Metric': row[f'{metric}_1'],
                    'Model2_Metric': row[f'{metric}_2'],
                    'Domain': domain.title()
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Create scatter plot
            colors = [RESEARCH_COLORS['cognitive'][0] if d == 'Cognitive' else RESEARCH_COLORS['mri'][0] 
                     for d in comp_df['Domain']]
            
            ax.scatter(comp_df['Model1_Metric'], comp_df['Model2_Metric'], 
                      c=colors, alpha=0.6, s=60)
            
            # Add diagonal line
            min_val = min(comp_df['Model1_Metric'].min(), comp_df['Model2_Metric'].min())
            max_val = max(comp_df['Model1_Metric'].max(), comp_df['Model2_Metric'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Formatting
            ax.set_xlabel(f'{model1.replace("_", " ").title()} {metric}')
            ax.set_ylabel(f'{model2.replace("_", " ").title()} {metric}')
            ax.set_title(title, fontweight='bold')
            
            # Add legend
            cognitive_patch = mpatches.Patch(color=RESEARCH_COLORS['cognitive'][0], label='Cognitive')
            mri_patch = mpatches.Patch(color=RESEARCH_COLORS['mri'][0], label='MRI')
            ax.legend(handles=[cognitive_patch, mri_patch])
        else:
            ax.text(0.5, 0.5, 'No group correlation data', ha='center', va='center')
            ax.set_title(title)
    
    def _plot_effect_size_by_group(self, ax, title: str):
        """Plot effect sizes (R²) by group."""
        group_effect_sizes = {}
        
        for group in self.results['by_group']:
            all_r2 = []
            for domain in ['cognitive', 'mri']:
                if domain in self.results['by_group'][group]:
                    r2_values = [r['R_Squared'] for r in self.results['by_group'][group][domain]
                               if not np.isnan(r['R_Squared'])]
                    all_r2.extend(r2_values)
            
            if all_r2:
                group_effect_sizes[group] = all_r2
        
        if group_effect_sizes:
            groups = list(group_effect_sizes.keys())
            data = list(group_effect_sizes.values())
            
            # Colors
            colors = [RESEARCH_COLORS['groups'].get(g, 'gray') for g in groups]
            
            # Box plot
            bp = ax.boxplot(data, labels=groups, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Effect Size (R²)')
            ax.set_title(title, fontweight='bold')
            
            # Add effect size interpretation lines
            ax.axhline(y=0.01, color='green', linestyle=':', alpha=0.5, label='Small (0.01)')
            ax.axhline(y=0.09, color='orange', linestyle=':', alpha=0.5, label='Medium (0.09)')
            ax.axhline(y=0.25, color='red', linestyle=':', alpha=0.5, label='Large (0.25)')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No effect size data', ha='center', va='center')
            ax.set_title(title)
    
    def _plot_significant_findings_by_group(self, ax, title: str):
        """Plot count of significant findings by group."""
        group_sig_counts = {}
        
        for group in self.results['by_group']:
            sig_count = 0
            total_count = 0
            
            for domain in ['cognitive', 'mri']:
                if domain in self.results['by_group'][group]:
                    domain_results = self.results['by_group'][group][domain]
                    total_count += len(domain_results)
                    sig_count += sum(1 for r in domain_results if r.get('Significant_FDR', False))
            
            if total_count > 0:
                group_sig_counts[group] = {
                    'significant': sig_count,
                    'total': total_count,
                    'percentage': (sig_count / total_count) * 100
                }
        
        if group_sig_counts:
            groups = list(group_sig_counts.keys())
            sig_counts = [group_sig_counts[g]['significant'] for g in groups]
            percentages = [group_sig_counts[g]['percentage'] for g in groups]
            
            # Colors
            colors = [RESEARCH_COLORS['groups'].get(g, 'gray') for g in groups]
            
            # Bar plot
            bars = ax.bar(groups, sig_counts, color=colors, alpha=0.7)
            
            # Add percentage labels
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                       f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Number of Significant Correlations')
            ax.set_title(title, fontweight='bold')
            
            # Add total counts as text
            for i, (group, bar) in enumerate(zip(groups, bars)):
                total = group_sig_counts[group]['total']
                ax.text(bar.get_x() + bar.get_width()/2, -max(sig_counts)*0.1,
                       f'n={total}', ha='center', va='top', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No significance data', ha='center', va='center')
            ax.set_title(title)
    
    def _create_correlation_scatterplots(self):
        """Create scatter plots for top correlation findings."""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        
        # Get top correlations from overall results
        top_correlations = []
        
        for domain in ['cognitive', 'mri']:
            if domain in self.results['overall']:
                domain_results = self.results['overall'][domain]
                # Get significant correlations sorted by effect size
                sig_results = [r for r in domain_results if r.get('Significant_FDR', False)]
                sig_results.sort(key=lambda x: abs(x['Correlation_Coefficient']), reverse=True)
                top_correlations.extend(sig_results[:3])  # Top 3 per domain
        
        # Sort all by effect size and take top 6
        top_correlations.sort(key=lambda x: abs(x['Correlation_Coefficient']), reverse=True)
        top_correlations = top_correlations[:6]
        
        for i, result in enumerate(top_correlations):
            if i >= len(axes):
                break
            
            ax = axes[i]
            self._plot_correlation_scatter(ax, result)
        
        # Hide unused axes
        for i in range(len(top_correlations), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "correlations" / "correlation_scatterplots.png")
        plt.close()
    
    def _plot_correlation_scatter(self, ax, correlation_result: Dict):
        """Plot individual correlation scatter plot."""
        sleep_var = correlation_result['Sleep_Variable']
        outcome = correlation_result['Outcome']
        domain = correlation_result['Domain']
        
        if sleep_var not in self.data.columns or outcome not in self.data.columns:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center')
            return
        
        # Clean data
        plot_data = self.data[[sleep_var, outcome]].dropna()
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No clean data', ha='center', va='center')
            return
        
        # Create scatter plot
        ax.scatter(plot_data[sleep_var], plot_data[outcome], 
                  alpha=0.6, s=50, 
                  color=RESEARCH_COLORS['cognitive'][0] if domain == 'Cognitive' else RESEARCH_COLORS['mri'][0])
        
        # Add regression line
        z = np.polyfit(plot_data[sleep_var], plot_data[outcome], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data[sleep_var].min(), plot_data[sleep_var].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Formatting
        ax.set_xlabel(sleep_var)
        
        if domain == 'MRI':
            formatted_outcome = self.format_roi_name(outcome)
        else:
            formatted_outcome = outcome
        ax.set_ylabel(formatted_outcome)
        
        # Title with statistics
        r = correlation_result['Correlation_Coefficient']
        p_val = correlation_result['P_Value']
        n = correlation_result['N']
        
        title = f'{formatted_outcome} vs {sleep_var}\nr = {r:.3f}, p = {p_val:.3e}, n = {n}'
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Add confidence interval if available
        if not np.isnan(correlation_result.get('CI_Lower', np.nan)):
            ci_text = f"95% CI: [{correlation_result['CI_Lower']:.3f}, {correlation_result['CI_Upper']:.3f}]"
            ax.text(0.02, 0.98, ci_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def generate_correlation_report(self):
        """Generate comprehensive correlation analysis report."""
        logger.info("Generating correlation analysis report...")
        
        # Save overall correlation results
        for domain in ['cognitive', 'mri']:
            if domain in self.results['overall'] and self.results['overall'][domain]:
                df = pd.DataFrame(self.results['overall'][domain])
                self.save_results_table(df, f"correlation_overall_{domain}_results.csv")
        
        # Save group-specific results
        for group in self.results['by_group']:
            for domain in ['cognitive', 'mri']:
                if domain in self.results['by_group'][group] and self.results['by_group'][group][domain]:
                    df = pd.DataFrame(self.results['by_group'][group][domain])
                    self.save_results_table(df, f"correlation_{group}_{domain}_results.csv")
        
        # Save group difference results
        for domain in ['cognitive', 'mri']:
            if domain in self.results['group_differences'] and self.results['group_differences'][domain]:
                df = pd.DataFrame(self.results['group_differences'][domain])
                self.save_results_table(df, f"correlation_group_differences_{domain}_results.csv")
        
        # Generate summary statistics
        self._generate_correlation_summary()
        
        return self
    
    def _generate_correlation_summary(self):
        """Generate correlation analysis summary statistics."""
        summary_stats = []
        
        # Overall correlation summary
        for domain in ['cognitive', 'mri']:
            if domain in self.results['overall'] and self.results['overall'][domain]:
                results = self.results['overall'][domain]
                
                correlations = [r['Correlation_Coefficient'] for r in results 
                              if not np.isnan(r['Correlation_Coefficient'])]
                effect_sizes = [r['R_Squared'] for r in results 
                              if not np.isnan(r['R_Squared'])]
                
            summary_stats.append({
                'Analysis_Type': 'Overall Correlations',
                'Domain': domain,
                'N_Correlations': len(results),
                'N_Significant_Raw': len([r for r in results if r['P_Value'] < 0.05]),
                'N_Significant_FDR': len([r for r in results if r.get('Significant_FDR', False)]),
                'Mean_Correlation': np.mean([abs(c) for c in correlations]) if correlations else np.nan,
                'Median_Correlation': np.median([abs(c) for c in correlations]) if correlations else np.nan,
                'Max_Correlation': max([abs(c) for c in correlations]) if correlations else np.nan,
                'Mean_Effect_Size': np.mean(effect_sizes) if effect_sizes else np.nan,
                'Median_Effect_Size': np.median(effect_sizes) if effect_sizes else np.nan,
                'Max_Effect_Size': max(effect_sizes) if effect_sizes else np.nan
            })
    
    def _plot_effect_size_comparison(self, ax):
        """Plot effect size comparison across models."""
        effect_data = []
        
        for model_type in ['base_model', 'full_model']:
            for domain in ['cognitive', 'mri']:
                if not self.results[model_type].get(domain):
                    continue
                
                df = pd.DataFrame(self.results[model_type][domain])
                sleep_effects = df[df['Predictor'].isin(self.sleep_vars)]
                
                for _, row in sleep_effects.iterrows():
                    effect_data.append({
                        'Model': model_type.replace('_', ' ').title(),
                        'Domain': domain.title(),
                        'Effect_Size': row['Effect_Size'],
                        'Significant': row.get('Significant_FDR', False)
                    })
        
        if effect_data:
            effect_df = pd.DataFrame(effect_data)
            
            # Box plot by model and domain
            positions = []
            box_data = []
            labels = []
            colors = []
            
            for i, model in enumerate(['Base Model', 'Full Model']):
                for j, domain in enumerate(['Cognitive', 'Mri']):
                    data = effect_df[(effect_df['Model'] == model) & 
                                   (effect_df['Domain'] == domain)]['Effect_Size'].dropna()
                    if len(data) > 0:
                        pos = i * 3 + j
                        positions.append(pos)
                        box_data.append(data)
                        labels.append(f'{model}\n{domain}')
                        colors.append(RESEARCH_COLORS['cognitive'][0] if domain == 'Cognitive' 
                                    else RESEARCH_COLORS['mri'][0])
            
            if box_data:
                bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
                
                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Effect Size (Partial η²)')
                ax.set_title('Effect Size Comparison Across Models', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No effect size data available', ha='center', va='center')
            ax.set_title('Effect Size Comparison')
    
    def generate_longitudinal_report(self):
        """Generate comprehensive longitudinal analysis report."""
        logger.info("Generating longitudinal analysis report...")
        
        # Save detailed results tables
        for model_type in self.results:
            for domain in ['cognitive', 'mri']:
                if self.results[model_type][domain]:
                    df = pd.DataFrame(self.results[model_type][domain])
                    filename = f"longitudinal_{model_type}_{domain}_results.csv"
                    self.save_results_table(df, filename)
        
        # Generate summary statistics
        self._generate_summary_stats()
        
        return self
    
    def _generate_summary_stats(self):
        """Generate summary statistics for longitudinal analysis."""
        summary_stats = []
        
        for model_type in self.results:
            for domain in ['cognitive', 'mri']:
                if not self.results[model_type][domain]:
                    continue
                
                df = pd.DataFrame(self.results[model_type][domain])
                sleep_effects = df[df['Predictor'].isin(self.sleep_vars)]
                
                if len(sleep_effects) > 0:
                    summary_stats.append({
                        'Model_Type': model_type,
                        'Domain': domain,
                        'Total_Tests': len(sleep_effects),
                        'Significant_Raw': len(sleep_effects[sleep_effects['P_Value'] < 0.05]),
                        'Significant_FDR': len(sleep_effects[sleep_effects.get('Significant_FDR', False)]),
                        'Mean_Effect_Size': sleep_effects['Effect_Size'].mean(),
                        'Median_Effect_Size': sleep_effects['Effect_Size'].median(),
                        'Max_Effect_Size': sleep_effects['Effect_Size'].max(),
                        'Mean_AIC': sleep_effects['AIC'].mean(),
                        'Mean_BIC': sleep_effects['BIC'].mean()
                    })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            self.save_results_table(summary_df, "longitudinal_summary_statistics.csv")


class CrossSectionalAnalyzer(BaseAnalyzer):
    """Cross-sectional analysis using LASSO regression and hypothesis testing."""
    
    def __init__(self, data: pd.DataFrame, output_dir: str):
        super().__init__(data, output_dir)
        self.results = {'lasso': {}, 'significance': {}, 'feature_importance': {}}
        self.scaler = StandardScaler()
    
    def run_lasso_analysis(self, alpha_range: Tuple[float, float] = (0.001, 1.0), cv_folds: int = 10):
        """
        Perform LASSO regression with cross-validation for variable selection.
        
        Parameters:
        -----------
        alpha_range : Tuple[float, float]
            Range of alpha values for LASSO regularization
        cv_folds : int
            Number of cross-validation folds
        """
        logger.info("Running LASSO analysis...")
        
        (self.output_dir / "cross_sectional").mkdir(exist_ok=True)
        
        # Prepare predictors
        predictor_cols = self.sociodemographic_vars + self.sleep_vars
        available_predictors = [col for col in predictor_cols if col in self.data.columns]
        
        if not available_predictors:
            logger.error("No predictor variables found")
            return self
        
        X = self.data[available_predictors].dropna()
        
        # Scale predictors
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=available_predictors, index=X.index)
        
        # Run LASSO for cognitive outcomes
        self._run_lasso_domain(X_scaled, self.cognitive_outcomes, 'cognitive', alpha_range, cv_folds)
        
        # Run LASSO for MRI outcomes
        self._run_lasso_domain(X_scaled, self.mri_outcomes, 'mri', alpha_range, cv_folds)
        
        return self
    
    def _run_lasso_domain(self, X: pd.DataFrame, outcomes: List[str], domain: str, 
                         alpha_range: Tuple[float, float], cv_folds: int):
        """Run LASSO analysis for a specific domain."""
        self.results['lasso'][domain] = []
        
        alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), 100)
        
        for outcome in outcomes:
            if outcome not in self.data.columns:
                continue
            
            try:
                # Align X and y indices
                y = self.data.loc[X.index, outcome].dropna()
                X_aligned = X.loc[y.index]
                
                if len(y) < 50:  # Minimum sample size
                    logger.warning(f"Insufficient data for {outcome} (n={len(y)})")
                    continue
                
                # Fit LASSO with cross-validation
                lasso_cv = LassoCV(alphas=alphas, cv=cv_folds, random_state=42, max_iter=2000)
                lasso_cv.fit(X_aligned, y)
                
                # Get coefficients
                coefficients = pd.Series(lasso_cv.coef_, index=X_aligned.columns)
                selected_features = coefficients[coefficients != 0]
                
                # Calculate metrics
                y_pred = lasso_cv.predict(X_aligned)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                self.results['lasso'][domain].append({
                    'Outcome': outcome,
                    'Alpha_Optimal': lasso_cv.alpha_,
                    'N_Features_Selected': len(selected_features),
                    'R2_Score': r2,
                    'MSE': mse,
                    'Selected_Features': selected_features.to_dict(),
                    'All_Coefficients': coefficients.to_dict()
                })
                
                logger.info(f"LASSO completed for {outcome}: {len(selected_features)} features selected, R² = {r2:.3f}")
                
            except Exception as e:
                logger.warning(f"LASSO failed for {outcome}: {e}")
    
    def run_significance_testing(self, alpha: float = 0.05):
        """
        Perform comprehensive significance testing with multiple comparison correction.
        
        Parameters:
        -----------
        alpha : float
            Significance threshold
        """
        logger.info("Running significance testing...")
        
        # Prepare data
        predictor_cols = self.sociodemographic_vars + self.sleep_vars
        available_predictors = [col for col in predictor_cols if col in self.data.columns]
        
        # Test cognitive outcomes
        self._test_domain_significance(available_predictors, self.cognitive_outcomes, 'cognitive', alpha)
        
        # Test MRI outcomes
        self._test_domain_significance(available_predictors, self.mri_outcomes, 'mri', alpha)
        
        return self
    
    def _test_domain_significance(self, predictors: List[str], outcomes: List[str], 
                                 domain: str, alpha: float):
        """Run significance testing for a domain."""
        self.results['significance'][domain] = []
        
        for outcome in outcomes:
            if outcome not in self.data.columns:
                continue
            
            for predictor in predictors:
                if predictor not in self.data.columns:
                    continue
                
                try:
                    # Get clean data
                    data_subset = self.data[[outcome, predictor] + self.base_predictors].dropna()
                    
                    if len(data_subset) < 30:
                        continue
                    
                    # Prepare regression formula
                    control_vars = [var for var in self.base_predictors if var in data_subset.columns]
                    formula = f"{outcome} ~ {predictor}"
                    if control_vars:
                        formula += f" + {' + '.join(control_vars)}"
                    
                    # Fit regression model
                    model = smf.ols(formula, data=data_subset).fit()
                    
                    # Extract results for the predictor of interest
                    if predictor in model.params.index:
                        result = {
                            'Outcome': outcome,
                            'Predictor': predictor,
                            'Coefficient': model.params[predictor],
                            'SE': model.bse[predictor],
                            'T_Statistic': model.tvalues[predictor],
                            'P_Value': model.pvalues[predictor],
                            'CI_Lower': model.conf_int().loc[predictor, 0],
                            'CI_Upper': model.conf_int().loc[predictor, 1],
                            'R2_Adj': model.rsquared_adj,
                            'N_Observations': len(data_subset),
                            'Domain': domain
                        }
                        
                        self.results['significance'][domain].append(result)
                
                except Exception as e:
                    logger.warning(f"Significance test failed: {outcome} ~ {predictor} ({e})")
        
        # Apply FDR correction
        if self.results['significance'][domain]:
            df = pd.DataFrame(self.results['significance'][domain])
            
            # Focus on sleep variables
            sleep_results = df[df['Predictor'].isin(self.sleep_vars)].copy()
            
            if len(sleep_results) > 0:
                # FDR correction
                fdr_rejected, fdr_pvals, _, _ = multipletests(
                    sleep_results['P_Value'], alpha=alpha, method='fdr_bh'
                )
                
                sleep_results['P_FDR'] = fdr_pvals
                sleep_results['Significant_FDR'] = fdr_rejected
                
                # Update results
                for i, result in enumerate(self.results['significance'][domain]):
                    if result['Predictor'] in self.sleep_vars:
                        idx = sleep_results[
                            (sleep_results['Outcome'] == result['Outcome']) & 
                            (sleep_results['Predictor'] == result['Predictor'])
                        ].index
                        if len(idx) > 0:
                            self.results['significance'][domain][i]['P_FDR'] = fdr_pvals[idx[0]]
                            self.results['significance'][domain][i]['Significant_FDR'] = fdr_rejected[idx[0]]
    
    def create_cross_sectional_figures(self):
        """Create publication-ready figures for cross-sectional analysis."""
        logger.info("Creating cross-sectional analysis figures...")
        
        fig_dir = self.output_dir / "figures" / "cross_sectional"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Figure 1: LASSO feature selection
        self._create_lasso_figure()
        
        # Figure 2: Significance testing results
        self._create_significance_figure()
        
        # Figure 3: Feature importance comparison
        self._create_feature_importance_figure()
        
        return self
    
    def _create_lasso_figure(self):
        """Create LASSO analysis visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # LASSO path for cognitive outcomes
        self._plot_lasso_path(ax1, 'cognitive', 'Cognitive Outcomes - LASSO Path')
        
        # LASSO path for MRI outcomes (top features)
        self._plot_lasso_path(ax2, 'mri', 'MRI Outcomes - LASSO Path', top_n=10)
        
        # Feature selection frequency
        self._plot_feature_selection_frequency(ax3, 'cognitive', 'Cognitive - Feature Selection Frequency')
        self._plot_feature_selection_frequency(ax4, 'mri', 'MRI - Feature Selection Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "cross_sectional" / "lasso_analysis.png")
        plt.close()
    
    def _plot_lasso_path(self, ax, domain: str, title: str, top_n: int = None):
        """Plot LASSO regularization path."""
        if domain not in self.results['lasso'] or not self.results['lasso'][domain]:
            ax.text(0.5, 0.5, 'No LASSO results available', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Aggregate feature importance across outcomes
        feature_importance = defaultdict(list)
        
        for result in self.results['lasso'][domain]:
            if top_n and len(feature_importance) >= top_n:
                break
            
            for feature, coef in result['Selected_Features'].items():
                if feature in self.sleep_vars:  # Focus on sleep variables
                    feature_importance[feature].append(abs(coef))
        
        # Calculate mean importance and plot
        mean_importance = {feat: np.mean(values) for feat, values in feature_importance.items()}
        
        if mean_importance:
            features = list(mean_importance.keys())
            importance = list(mean_importance.values())
            
            bars = ax.bar(features, importance, color=RESEARCH_COLORS['sleep'], alpha=0.7)
            ax.set_xlabel('Sleep Variables')
            ax.set_ylabel('Mean |Coefficient|')
            ax.set_title(title, fontweight='bold')
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, importance):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance)*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No sleep features selected', ha='center', va='center')
            ax.set_title(title)
    
    def _plot_feature_selection_frequency(self, ax, domain: str, title: str):
        """Plot frequency of feature selection across outcomes."""
        if domain not in self.results['lasso'] or not self.results['lasso'][domain]:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Count selection frequency
        selection_count = defaultdict(int)
        total_outcomes = len(self.results['lasso'][domain])
        
        for result in self.results['lasso'][domain]:
            for feature in result['Selected_Features'].keys():
                selection_count[feature] += 1
        
        # Focus on sleep variables and top sociodemographic variables
        relevant_features = {k: v for k, v in selection_count.items() 
                           if k in self.sleep_vars or v >= total_outcomes * 0.3}
        
        if relevant_features:
            features = list(relevant_features.keys())
            frequencies = [relevant_features[f] / total_outcomes * 100 for f in features]
            
            # Color coding
            colors = []
            for feat in features:
                if feat in self.sleep_vars:
                    colors.append(RESEARCH_COLORS['sleep'][self.sleep_vars.index(feat) % len(RESEARCH_COLORS['sleep'])])
                else:
                    colors.append('lightblue')
            
            bars = ax.bar(features, frequencies, color=colors, alpha=0.7)
            ax.set_xlabel('Variables')
            ax.set_ylabel('Selection Frequency (%)')
            ax.set_title(title, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Add frequency labels
            for bar, freq in zip(bars, frequencies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{freq:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No frequently selected features', ha='center', va='center')
            ax.set_title(title)
    
    def _create_significance_figure(self):
        """Create significance testing visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Volcano plots
        self._plot_volcano(ax1, 'cognitive', 'Cognitive Outcomes - Sleep Effects')
        self._plot_volcano(ax2, 'mri', 'MRI Outcomes - Sleep Effects')
        
        # Effect size distributions
        self._plot_effect_size_distribution(ax3, 'cognitive', 'Cognitive Effect Sizes')
        self._plot_effect_size_distribution(ax4, 'mri', 'MRI Effect Sizes')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "cross_sectional" / "significance_testing.png")
        plt.close()
    
    def _plot_volcano(self, ax, domain: str, title: str):
        """Create volcano plot for significance testing results."""
        if domain not in self.results['significance'] or not self.results['significance'][domain]:
            ax.text(0.5, 0.5, 'No significance results', ha='center', va='center')
            ax.set_title(title)
            return
        
        df = pd.DataFrame(self.results['significance'][domain])
        sleep_results = df[df['Predictor'].isin(self.sleep_vars)].copy()
        
        if len(sleep_results) == 0:
            ax.text(0.5, 0.5, 'No sleep variable results', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Calculate -log10(p-value)
        sleep_results['-log10_p'] = -np.log10(sleep_results['P_Value'].clip(lower=1e-10))
        
        # Color by significance and sleep variable
        colors = []
        for _, row in sleep_results.iterrows():
            if row.get('Significant_FDR', False):
                if row['Predictor'] == 'NPIK':
                    colors.append(RESEARCH_COLORS['sleep'][0])
                elif row['Predictor'] == 'NPIKSEV':
                    colors.append(RESEARCH_COLORS['sleep'][1])
                else:
                    colors.append(RESEARCH_COLORS['sleep'][2])
            else:
                colors.append('lightgray')
        
        # Create scatter plot
        ax.scatter(sleep_results['Coefficient'], sleep_results['-log10_p'], 
                  c=colors, alpha=0.6, s=60)
        
        # Add significance threshold line
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p = 0.05')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Coefficient')
        ax.set_ylabel('-log₁₀(p-value)')
        ax.set_title(title, fontweight='bold')
        ax.legend()
        
        # Add annotations for highly significant results
        highly_sig = sleep_results[sleep_results['-log10_p'] > 2]
        for _, row in highly_sig.iterrows():
            if domain == 'mri':
                label = self.format_roi_name(row['Outcome'])
            else:
                label = row['Outcome']
            ax.annotate(f"{label}\n({row['Predictor']})", 
                       (row['Coefficient'], row['-log10_p']),
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.8)
    
    def _plot_effect_size_distribution(self, ax, domain: str, title: str):
        """Plot distribution of effect sizes."""
        if domain not in self.results['significance'] or not self.results['significance'][domain]:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title)
            return
        
        df = pd.DataFrame(self.results['significance'][domain])
        sleep_results = df[df['Predictor'].isin(self.sleep_vars)].copy()
        
        if len(sleep_results) == 0:
            ax.text(0.5, 0.5, 'No sleep variable results', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Calculate effect sizes (Cohen's f²)
        sleep_results['Effect_Size'] = sleep_results['R2_Adj'].clip(lower=0)
        
        # Create violin plot by sleep variable
        sleep_vars_present = sleep_results['Predictor'].unique()
        data_by_var = []
        positions = []
        colors = []
        
        for i, var in enumerate(sleep_vars_present):
            data = sleep_results[sleep_results['Predictor'] == var]['Effect_Size'].dropna()
            if len(data) > 0:
                data_by_var.append(data)
                positions.append(i)
                colors.append(RESEARCH_COLORS['sleep'][i % len(RESEARCH_COLORS['sleep'])])
        
        if data_by_var:
            parts = ax.violinplot(data_by_var, positions=positions, widths=0.6, showmeans=True)
            
            # Color the violins
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(sleep_vars_present)
            ax.set_ylabel('Effect Size (R² adj)')
            ax.set_title(title, fontweight='bold')
            
            # Add horizontal lines for effect size interpretation
            ax.axhline(y=0.01, color='green', linestyle=':', alpha=0.5, label='Small (0.01)')
            ax.axhline(y=0.09, color='orange', linestyle=':', alpha=0.5, label='Medium (0.09)')
            ax.axhline(y=0.25, color='red', linestyle=':', alpha=0.5, label='Large (0.25)')
            ax.legend(loc='upper right', fontsize=8)
    
    def _create_feature_importance_figure(self):
        """Create comprehensive feature importance visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # LASSO vs significance comparison - cognitive
        self._plot_method_comparison(ax1, 'cognitive', 'Cognitive: LASSO vs Significance')
        
        # LASSO vs significance comparison - MRI
        self._plot_method_comparison(ax2, 'mri', 'MRI: LASSO vs Significance')
        
        # Sleep variable importance ranking
        self._plot_sleep_variable_ranking(ax3, 'Sleep Variable Importance - Cognitive')
        self._plot_sleep_variable_ranking(ax4, 'Sleep Variable Importance - MRI')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "cross_sectional" / "feature_importance.png")
        plt.close()
    
    def _plot_method_comparison(self, ax, domain: str, title: str):
        """Compare LASSO selection with significance testing."""
        # Get LASSO selected features
        lasso_features = set()
        if domain in self.results['lasso'] and self.results['lasso'][domain]:
            for result in self.results['lasso'][domain]:
                lasso_features.update(result['Selected_Features'].keys())
        
        # Get significantly associated features
        sig_features = set()
        if domain in self.results['significance'] and self.results['significance'][domain]:
            sig_df = pd.DataFrame(self.results['significance'][domain])
            sig_sleep = sig_df[(sig_df['Predictor'].isin(self.sleep_vars)) & 
                              (sig_df.get('Significant_FDR', False))]
            sig_features = set(sig_sleep['Predictor'].unique())
        
        # Create Venn diagram data
        lasso_only = lasso_features - sig_features
        sig_only = sig_features - lasso_features
        both = lasso_features & sig_features
        
        # Create bar plot showing overlap
        categories = ['LASSO Only', 'Significance Only', 'Both Methods']
        counts = [len(lasso_only), len(sig_only), len(both)]
        colors = [RESEARCH_COLORS['cognitive'][2], RESEARCH_COLORS['mri'][2], 'darkgreen']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Features')
        ax.set_title(title, fontweight='bold')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
        
        # Add feature names as text
        y_pos = max(counts) * 0.7
        if lasso_only:
            ax.text(0, y_pos, f'LASSO:\n{", ".join(list(lasso_only)[:3])}', 
                   ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        if sig_only:
            ax.text(1, y_pos, f'Significant:\n{", ".join(list(sig_only)[:3])}', 
                   ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        if both:
            ax.text(2, y_pos, f'Both:\n{", ".join(list(both)[:3])}', 
                   ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    def _plot_sleep_variable_ranking(self, ax, title: str):
        """Plot sleep variable importance ranking across methods."""
        domain = 'cognitive' if 'Cognitive' in title else 'mri'
        
        # Collect importance scores
        importance_scores = defaultdict(list)
        
        # LASSO importance
        if domain in self.results['lasso'] and self.results['lasso'][domain]:
            for result in self.results['lasso'][domain]:
                for var in self.sleep_vars:
                    if var in result['Selected_Features']:
                        importance_scores[var].append(abs(result['Selected_Features'][var]))
                    else:
                        importance_scores[var].append(0)
        
        # Significance importance (based on effect sizes)
        if domain in self.results['significance'] and self.results['significance'][domain]:
            sig_df = pd.DataFrame(self.results['significance'][domain])
            for var in self.sleep_vars:
                var_results = sig_df[sig_df['Predictor'] == var]
                if len(var_results) > 0:
                    mean_r2 = var_results['R2_Adj'].mean()
                    importance_scores[f"{var}_sig"].append(mean_r2)
        
        if importance_scores:
            # Calculate mean importance scores
            mean_scores = {}
            for var in self.sleep_vars:
                lasso_scores = importance_scores.get(var, [0])
                sig_scores = importance_scores.get(f"{var}_sig", [0])
                
                mean_scores[f"{var}_LASSO"] = np.mean(lasso_scores)
                mean_scores[f"{var}_Significance"] = np.mean(sig_scores)
            
            # Create grouped bar plot
            methods = ['LASSO', 'Significance']
            x = np.arange(len(self.sleep_vars))
            width = 0.35
            
            lasso_scores = [mean_scores[f"{var}_LASSO"] for var in self.sleep_vars]
            sig_scores = [mean_scores[f"{var}_Significance"] for var in self.sleep_vars]
            
            bars1 = ax.bar(x - width/2, lasso_scores, width, label='LASSO', 
                          color=RESEARCH_COLORS['cognitive'][0], alpha=0.7)
            bars2 = ax.bar(x + width/2, sig_scores, width, label='Significance', 
                          color=RESEARCH_COLORS['mri'][0], alpha=0.7)
            
            ax.set_xlabel('Sleep Variables')
            ax.set_ylabel('Mean Importance Score')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.sleep_vars)
            ax.legend()
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, height + max(max(lasso_scores), max(sig_scores))*0.02,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No importance data available', ha='center', va='center')
            ax.set_title(title)
    
    def generate_cross_sectional_report(self):
        """Generate comprehensive cross-sectional analysis report."""
        logger.info("Generating cross-sectional analysis report...")
        
        # Save LASSO results
        for domain in ['cognitive', 'mri']:
            if domain in self.results['lasso'] and self.results['lasso'][domain]:
                lasso_df = pd.DataFrame(self.results['lasso'][domain])
                self.save_results_table(lasso_df, f"cross_sectional_lasso_{domain}_results.csv")
        
        # Save significance testing results
        for domain in ['cognitive', 'mri']:
            if domain in self.results['significance'] and self.results['significance'][domain]:
                sig_df = pd.DataFrame(self.results['significance'][domain])
                self.save_results_table(sig_df, f"cross_sectional_significance_{domain}_results.csv")
        
        # Generate summary statistics
        self._generate_cross_sectional_summary()
        
        return self
    
    def _generate_cross_sectional_summary(self):
        """Generate summary statistics for cross-sectional analysis."""
        summary_stats = []
        
        # LASSO summary
        for domain in ['cognitive', 'mri']:
            if domain in self.results['lasso'] and self.results['lasso'][domain]:
                results = self.results['lasso'][domain]
                
                r2_scores = [r['R2_Score'] for r in results if not np.isnan(r['R2_Score'])]
                n_features = [r['N_Features_Selected'] for r in results]
                
                summary_stats.append({
                    'Analysis_Type': 'LASSO',
                    'Domain': domain,
                    'N_Outcomes_Analyzed': len(results),
                    'Mean_R2_Score': np.mean(r2_scores) if r2_scores else np.nan,
                    'Median_R2_Score': np.median(r2_scores) if r2_scores else np.nan,
                    'Mean_Features_Selected': np.mean(n_features),
                    'Total_Sleep_Selections': sum(1 for r in results 
                                                 for f in r['Selected_Features'] 
                                                 if f in self.sleep_vars)
                })
        
        # Significance testing summary
        for domain in ['cognitive', 'mri']:
            if domain in self.results['significance'] and self.results['significance'][domain]:
                df = pd.DataFrame(self.results['significance'][domain])
                sleep_results = df[df['Predictor'].isin(self.sleep_vars)]
                
                if len(sleep_results) > 0:
                    summary_stats.append({
                        'Analysis_Type': 'Significance Testing',
                        'Domain': domain,
                        'N_Tests_Conducted': len(sleep_results),
                        'N_Significant_Raw': len(sleep_results[sleep_results['P_Value'] < 0.05]),
                        'N_Significant_FDR': len(sleep_results[sleep_results.get('Significant_FDR', False)]),
                        'Mean_Effect_Size': sleep_results['R2_Adj'].mean(),
                        'Median_Effect_Size': sleep_results['R2_Adj'].median()
                    })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            self.save_results_table(summary_df, "cross_sectional_summary_statistics.csv")


class CorrelationAnalyzer(BaseAnalyzer):
    """Correlation analysis across diagnostic groups with effect size calculations."""
    
    def __init__(self, data: pd.DataFrame, output_dir: str):
        super().__init__(data, output_dir)
        self.results = {'overall': {}, 'by_group': {}, 'group_differences': {}}
    
    def compute_correlations(self, method: str = 'pearson', group_column: str = 'DX_bl'):
        """
        Compute correlations between sleep variables and outcomes.
        
        Parameters:
        -----------
        method : str
            Correlation method ('pearson', 'spearman')
        group_column : str
            Column name for diagnostic groups
        """
        logger.info(f"Computing {method} correlations...")
        
        (self.output_dir / "correlations").mkdir(exist_ok=True)
        
        # Overall correlations
        self._compute_overall_correlations(method)
        
        # Group-specific correlations
        if group_column in self.data.columns:
            self._compute_group_correlations(method, group_column)
            
            # Test for group differences in correlations
            self._test_correlation_differences(method, group_column)
        
        return self
    
    def _compute_overall_correlations(self, method: str):
        """Compute overall correlations across all subjects."""
        self.results['overall']['cognitive'] = []
        self.results['overall']['mri'] = []
        
        # Cognitive correlations
        for sleep_var in self.sleep_vars:
            if sleep_var not in self.data.columns:
                continue
                
            for outcome in self.cognitive_outcomes:
                if outcome not in self.data.columns:
                    continue
                
                correlation_result = self._calculate_correlation(
                    self.data[sleep_var], self.data[outcome], method
                )
                
                if correlation_result:
                    correlation_result.update({
                        'Sleep_Variable': sleep_var,
                        'Outcome': outcome,
                        'Domain': 'Cognitive'
                    })
                    self.results['overall']['cognitive'].append(correlation_result)
        
        # MRI correlations
        for sleep_var in self.sleep_vars:
            if sleep_var not in self.data.columns:
                continue
                
            for outcome in self.mri_outcomes:
                if outcome not in self.data.columns:
                    continue
                
                correlation_result = self._calculate_correlation(
                    self.data[sleep_var], self.data[outcome], method
                )
                
                if correlation_result:
                    correlation_result.update({
                        'Sleep_Variable': sleep_var,
                        'Outcome': outcome,
                        'Domain': 'MRI'
                    })
                    self.results['overall']['mri'].append(correlation_result)
        
        # Apply multiple comparison correction
        self._apply_correlation_corrections('overall')
    
    def _compute_group_correlations(self, method: str, group_column: str):
        """Compute correlations within diagnostic groups."""
        groups = self.data[group_column].unique()
        groups = [g for g in groups if not pd.isna(g)]
        
        for group in groups:
            group_data = self.data[self.data[group_column] == group]
            
            if len(group_data) < 10:  # Minimum sample size for correlations
                continue
            
            self.results['by_group'][group] = {'cognitive': [], 'mri': []}
            
            # Cognitive correlations within group
            for sleep_var in self.sleep_vars:
                if sleep_var not in group_data.columns:
                    continue
                    
                for outcome in self.cognitive_outcomes:
                    if outcome not in group_data.columns:
                        continue
                    
                    correlation_result = self._calculate_correlation(
                        group_data[sleep_var], group_data[outcome], method
                    )
                    
                    if correlation_result:
                        correlation_result.update({
                            'Sleep_Variable': sleep_var,
                            'Outcome': outcome,
                            'Domain': 'Cognitive',
                            'Group': group,
                            'N': len(group_data.dropna(subset=[sleep_var, outcome]))
                        })
                        self.results['by_group'][group]['cognitive'].append(correlation_result)
            
            # MRI correlations within group
            for sleep_var in self.sleep_vars:
                if sleep_var not in group_data.columns:
                    continue
                    
                for outcome in self.mri_outcomes:
                    if outcome not in group_data.columns:
                        continue
                    
                    correlation_result = self._calculate_correlation(
                        group_data[sleep_var], group_data[outcome], method
                    )
                    
                    if correlation_result:
                        correlation_result.update({
                            'Sleep_Variable': sleep_var,
                            'Outcome': outcome,
                            'Domain': 'MRI',
                            'Group': group,
                            'N': len(group_data.dropna(subset=[sleep_var, outcome]))
                        })
                        self.results['by_group'][group]['mri'].append(correlation_result)
        
        # Apply corrections within each group
        for group in self.results['by_group']:
            self._apply_correlation_corrections('by_group', group)
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series, method: str) -> Dict:
        """Calculate correlation with comprehensive statistics."""
        # Clean data
        clean_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(clean_data) < 10:
            return None
        
        x_clean = clean_data['x']
        y_clean = clean_data['y']
        
        try:
            if method == 'pearson':
                corr_coef, p_value = pearsonr(x_clean, y_clean)
            elif method == 'spearman':
                corr_coef, p_value = stats.spearmanr(x_clean, y_clean)
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            # Calculate confidence interval for Pearson correlation
            if method == 'pearson' and len(x_clean) > 3:
                # Fisher z-transformation
                z = np.arctanh(corr_coef)
                se = 1 / np.sqrt(len(x_clean) - 3)
                z_critical = stats.norm.ppf(0.975)  # 95% CI
                
                z_lower = z - z_critical * se
                z_upper = z + z_critical * se
                
                ci_lower = np.tanh(z_lower)
                ci_upper = np.tanh(z_upper)
            else:
                ci_lower, ci_upper = np.nan, np.nan
            
            # Effect size interpretation
            abs_corr = abs(corr_coef)
            if abs_corr < 0.1:
                effect_size = 'negligible'
            elif abs_corr < 0.3:
                effect_size = 'small'
            elif abs_corr < 0.5:
                effect_size = 'medium'
            else:
                effect_size = 'large'
            
            return {
                'Correlation_Coefficient': corr_coef,
                'P_Value': p_value,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'N': len(clean_data),
                'Method': method,
                'Effect_Size_Category': effect_size,
                'R_Squared': corr_coef ** 2
            }
            
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return None
    
    def _test_correlation_differences(self, method: str, group_column: str):
        """Test for significant differences in correlations between groups."""
        logger.info("Testing correlation differences between groups...")
        
        self.results['group_differences']['cognitive'] = []
        self.results['group_differences']['mri'] = []
        
        groups = list(self.results['by_group'].keys())
        
        if len(groups) < 2:
            logger.info("Need at least 2 groups for comparison")
            return
        
        # Compare pairs of groups
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1, group2 = groups[i], groups[j]
                
                # Cognitive comparisons
                self._compare_group_correlations(
                    group1, group2, 'cognitive', method
                )
                
                # MRI comparisons
                self._compare_group_correlations(
                    group1, group2, 'mri', method
                )
    
    def _compare_group_correlations(self, group1: str, group2: str, domain: str, method: str):
        """Compare correlations between two groups using Fisher's z-test."""
        if group1 not in self.results['by_group'] or group2 not in self.results['by_group']:
            return
        
        group1_results = {
            (r['Sleep_Variable'], r['Outcome']): r 
            for r in self.results['by_group'][group1][domain]
        }
        
        group2_results = {
            (r['Sleep_Variable'], r['Outcome']): r 
            for r in self.results['by_group'][group2][domain]
        }
        
        # Find common sleep-outcome pairs
        common_pairs = set(group1_results.keys()) & set(group2_results.keys())
        
        for sleep_var, outcome in common_pairs:
            r1_data = group1_results[(sleep_var, outcome)]
            r2_data = group2_results[(sleep_var, outcome)]
            
            # Fisher's z-test for correlation differences
            if method == 'pearson' and r1_data['N'] > 3 and r2_data['N'] > 3:
                r1, r2 = r1_data['Correlation_Coefficient'], r2_data['Correlation_Coefficient']
                n1, n2 = r1_data['N'], r2_data['N']
                
                # Fisher z-transformation
                z1 = np.arctanh(r1)
                z2 = np.arctanh(r2)
                
                # Standard error of difference
                se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
                
                # Z-statistic
                z_stat = (z1 - z2) / se_diff
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                self.results['group_differences'][domain].append({
                    'Sleep_Variable': sleep_var,
                    'Outcome': outcome,
                    'Group1': group1,
                    'Group2': group2,
                    'Correlation_Group1': r1,
                    'Correlation_Group2': r2,
                    'N_Group1': n1,
                    'N_Group2': n2,
                    'Z_Statistic': z_stat,
                    'P_Value': p_value,
                    'Correlation_Difference': r1 - r2,
                    'Domain': domain.title()
                })
        
        # Apply FDR correction to group differences
        if self.results['group_differences'][domain]:
            df = pd.DataFrame(self.results['group_differences'][domain])
            if len(df) > 0:
                fdr_rejected, fdr_pvals, _, _ = multipletests(
                    df['P_Value'], alpha=0.05, method='fdr_bh'
                )
                
                for i, result in enumerate(self.results['group_differences'][domain]):
                    result['P_FDR'] = fdr_pvals[i]
                    result['Significant_FDR'] = fdr_rejected[i]
    
    def _apply_correlation_corrections(self, result_type: str, group: str = None):
        """Apply FDR correction to correlation p-values."""
        if result_type == 'overall':
            for domain in ['cognitive', 'mri']:
                if self.results['overall'][domain]:
                    df = pd.DataFrame(self.results['overall'][domain])
                    self._apply_fdr_correction(df, self.results['overall'][domain])
        
        elif result_type == 'by_group' and group:
            for domain in ['cognitive', 'mri']:
                if self.results['by_group'][group][domain]:
                    df = pd.DataFrame(self.results['by_group'][group][domain])
                    self._apply_fdr_correction(df, self.results['by_group'][group][domain])
    
    def _apply_fdr_correction(self, df: pd.DataFrame, results_list: List[Dict]):
        """Apply FDR correction to a list of results."""
        if len(df) > 0:
            fdr_rejected, fdr_pvals, _, _ = multipletests(
                df['P_Value'], alpha=0.05, method='fdr_bh'
            )
            
            for i, result in enumerate(results_list):
                result['P_FDR'] = fdr_pvals[i]
                result['Significant_FDR'] = fdr_rejected[i]
    
    def create_correlation_figures(self):
        """Create publication-ready correlation analysis figures."""
        logger.info("Creating correlation analysis figures...")
        
        fig_dir = self.output_dir / "figures" / "correlations"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Figure 1: Overall correlation heatmaps
        self._create_correlation_heatmaps()
        
        # Figure 2: Group comparison plots
        self._create_group_comparison_plots()
        
        # Figure 3: Correlation scatter plots for top findings
        self._create_correlation_scatterplots()
        
        return self
    
    def _create_correlation_heatmaps(self):
        """Create correlation heatmaps for overall and group-specific results."""
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Overall cognitive correlations
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_correlation_heatmap(ax1, 'overall', 'cognitive', 'Overall Cognitive Correlations')
        
        # Overall MRI correlations (top 20)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_correlation_heatmap(ax2, 'overall', 'mri', 'Overall MRI Correlations (Top 20)', top_n=20)
        
        # Group differences heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_group_differences_heatmap(ax3, 'Group Correlation Differences')
        
        # Group-specific heatmaps
        groups = list(self.results['by_group'].keys())[:3]  # Show up to 3 groups
        for i, group in enumerate(groups):
            ax = fig.add_subplot(gs[1, i])
            self._plot_group_specific_heatmap(ax, group, f'{group} - Sleep-Cognition Correlations')
        
        plt.savefig(self.output_dir / "figures" / "correlations" / "correlation_heatmaps.png")
        plt.close()
    
    def _plot_correlation_heatmap(self, ax, result_type: str, domain: str, title: str, top_n: int = None):
        """Plot correlation heatmap for a specific domain."""
        if result_type == 'overall':
            data = self.results['overall'].get(domain, [])
        else:
            ax.text(0.5, 0.5, 'Group-specific data', ha='center', va='center')
            ax.set_title(title)
            return
        
        if not data:
            ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
            ax.set_title(title)
            return
        
        df = pd.DataFrame(data)
        
        # Select top correlations if specified
        if top_n and len(df) > top_n:
            df = df.nlargest(top_n, 'Correlation_Coefficient', keep='all')
        
        # Create pivot table
        pivot_data = df.pivot_table(
            index='Outcome', 
            columns='Sleep_Variable', 
            values='Correlation_Coefficient',
            aggfunc='first'
        )
        
        if pivot_data.empty:
            ax.text(0.5, 0.5, 'No pivot data available', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Create mask for non-significant correlations
        pivot_sig = df.pivot_table(
            index='Outcome', 
            columns='Sleep_Variable', 
            values='Significant_FDR',
            aggfunc='first'
        )
        
        # Plot heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Correlation Coefficient'}, ax=ax,
                   mask=~pivot_sig if not pivot_sig.empty else None)
        
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Format y-axis labels for MRI
        if domain == 'mri':
            y_labels = [self.format_roi_name(label.get_text()) for label in ax.get_yticklabels()]
            ax.set_yticklabels(y_labels, rotation=0)
    
    def _plot_group_differences_heatmap(self, ax, title: str):
        """Plot heatmap of correlation differences between groups."""
        all_diffs = []
        
        for domain in ['cognitive', 'mri']:
            if domain in self.results['group_differences']:
                all_diffs.extend(self.results['group_differences'][domain])
        
        if not all_diffs:
            ax.text(0.5, 0.5, 'No group differences', ha='center', va='center')
            ax.set_title(title)
            return
        
        df = pd.DataFrame(all_diffs)
        
        # Create comparison label
        df['Group_Comparison'] = df['Group1'] + ' vs ' + df['Group2']
        
        # Create pivot table for correlation differences
        pivot_data = df.pivot_table(
            index=['Domain', 'Outcome'], 
            columns=['Sleep_Variable', 'Group_Comparison'], 
            values='Correlation_Difference',
            aggfunc='first'
        )
        
        if not pivot_data.empty and len(pivot_data) <= 20:  # Limit size for readability
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       cbar_kws={'label': 'Correlation Difference'}, ax=ax)
            ax.set_title(title, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Too many comparisons\nfor heatmap display', ha='center', va='center')
            ax.set_title(title)
    
    def _plot_group_specific_heatmap(self, ax, group: str, title: str):
        """Plot correlation heatmap for a specific group."""
        if group not in self.results['by_group']:
            ax.text(0.5, 0.5, f'No data for {group}', ha='center', va='center')
            ax.set_title(title)
            return
        
        # Combine cognitive and MRI data for the group
        all_data = []
        all_data.extend(self.results['by_group'][group].get('cognitive', []))
        all_data.extend(self.results['by_group'][group].get('mri', []))
        
        if not all_data:
            ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
            ax.set_title(title)
            return
        
        df = pd.DataFrame(all_data)
        
        # Select top correlations for display
        if len(df) > 15:
            df = df.nlargest(15, 'Correlation_Coefficient', keep='all')
        
        # Create pivot table
        pivot_data = df.pivot_table(
            index='Outcome', 
            columns='Sleep_Variable', 
            values='Correlation_Coefficient',
            aggfunc='first'
        )
        
        if not pivot_data.empty:
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       cbar_kws={'label': 'Correlation'}, ax=ax)
            ax.set_title(title, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No pivot data', ha='center', va='center')
            ax.set_title(title)
    
    def _create_group_comparison_plots(self):
        """Create group comparison visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Correlation distribution by group - Cognitive
        self._plot_correlation_distribution_by_group(ax1, 'cognitive', 'Cognitive Correlations by Group')
        
        # Correlation distribution by group - MRI
        self._plot_correlation_distribution_by_group(ax2, 'mri', 'MRI Correlations by Group')
        
        # Effect size comparison
        self._plot_effect_size_by_group(ax3, 'Effect Size (R²) by Group')
        
        # Significant findings count
        self._plot_significant_findings_by_group(ax4, 'Significant Findings by Group')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "correlations" / "group_comparisons.png")
        plt.close()
    
    def _plot_correlation_distribution_by_group(self, ax, domain: str, title: str):
        """Plot distribution of correlations by group."""
        group_data = []
        group_labels = []
        colors = []
        
        for group in self.results['by_group']:
            if domain in self.results['by_group'][group]:
                correlations = [r['Correlation_Coefficient'] 
                              for r in self.results['by_group'][group][domain]
                              if not np.isnan(r['Correlation_Coefficient'])]
                
                if correlations:
                    group_data.append(correlations)
                    group_labels.append(group)
                    
                    # Color by group
                    if group in RESEARCH_COLORS['groups']:
                        colors.append(RESEARCH_COLORS['groups'][group])
                    else:
                        colors.append('gray')
        
        if group_data:
            # Create violin plot
            parts = ax.violinplot(group_data, positions=range(len(group_data)), 
                                 widths=0.7, showmeans=True, showmedians=True)
            
            # Color the violins
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(group_labels)))
            ax.set_xticklabels(group_labels)
            ax.set_ylabel('Correlation Coefficient')
            ax.set_title(title, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add sample size annotations
            for i, (data, label) in enumerate(zip(group_data, group_labels)):
                ax.text(i, max(data) + 0.1, f'n={len(data)}', ha='center', fontsize=9)
        else:
            ax.text(0.5, 0.5,