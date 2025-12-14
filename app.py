import os
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

current_dir = Path(__file__).parent.absolute()
logger.info(f" Current directory: {current_dir}")

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import datetime
import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import custom model classes
from model_classes import HybridRFSVM

def parse_boolean(value):
  
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ['true', '1', 'yes', 'y', 't', 'ya', 'benar']:
            return True
        elif value_lower in ['false', '0', 'no', 'n', 'f', 'tidak', 'tdk']:
            return False
        else:
            
            try:
                return int(value) != 0
            except:
                return False
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)

def parse_float_or_none(value):
   
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.lower() in ['null', 'none', 'undefined']:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


class FraminghamRiskScoreCalculator:
    def __init__(self):
      
        self.male_points = {
            'age': {
                20: -9, 25: -7, 30: -4, 35: 0, 40: 1, 45: 2, 
                50: 3, 55: 4, 60: 5, 65: 6, 70: 7, 75: 8
            },
            'total_chol': {
                '<160': 0, '160-199': 1, '200-239': 2, '240-279': 3, '280+': 4
            },
            'hdl': {
                '<35': 2, '35-44': 1, '45-49': 0, '50-59': -1, '60+': -2
            },
            'systolic_bp': {
                'untreated': {'<120': -2, '120-129': 0, '130-139': 1, '140-159': 2, '160+': 3},
                'treated': {'<120': 0, '120-129': 2, '130-139': 3, '140-159': 4, '160+': 5}
            },
            'smoker': {True: 2, False: 0},
            'diabetes': {True: 2, False: 0}
        }
        
        self.female_points = {
            'age': {
                20: -7, 25: -3, 30: 0, 35: 3, 40: 4, 45: 5, 
                50: 6, 55: 7, 60: 8, 65: 9, 70: 10, 75: 11
            },
            'total_chol': {
                '<160': 0, '160-199': 1, '200-239': 2, '240-279': 3, '280+': 4
            },
            'hdl': {
                '<35': 5, '35-44': 2, '45-49': 1, '50-59': 0, '60+': -3
            },
            'systolic_bp': {
                'untreated': {'<120': -3, '120-129': 0, '130-139': 1, '140-159': 2, '160+': 4},
                'treated': {'<120': -1, '120-129': 2, '130-139': 3, '140-159': 5, '160+': 7}
            },
            'smoker': {True: 3, False: 0},
            'diabetes': {True: 3, False: 0}
        }
        
       
        self.male_risk = {
            -20: 0.1, -19: 0.1, -18: 0.1, -17: 0.1, -16: 0.1, -15: 0.1, 
            -14: 0.1, -13: 0.1, -12: 0.1, -11: 0.1, -10: 0.1, -9: 0.1,
            -8: 0.1, -7: 0.1, -6: 0.1, -5: 0.1, -4: 0.1, -3: 0.1, 
            -2: 0.1, -1: 0.1, 0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 
            4: 0.1, 5: 0.1, 6: 0.2, 7: 0.3, 8: 0.4, 9: 0.5,
            10: 0.7, 11: 1.0, 12: 1.5, 13: 2.0, 14: 2.5, 
            15: 3.0, 16: 4.0, 17: 5.0, 18: 6.0, 19: 8.0,
            20: 10.0, 21: 12.0, 22: 14.0, 23: 16.0, 24: 18.0,
            25: 20.0, 26: 22.0, 27: 24.0, 28: 26.0, 29: 28.0,
            30: 30.0
        }
        
        self.female_risk = {
            -20: 0.1, -19: 0.1, -18: 0.1, -17: 0.1, -16: 0.1, -15: 0.1,
            -14: 0.1, -13: 0.1, -12: 0.1, -11: 0.1, -10: 0.1, -9: 0.1,
            -8: 0.1, -7: 0.1, -6: 0.1, -5: 0.1, -4: 0.1, -3: 0.1,
            -2: 0.1, -1: 0.1, 0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1,
            4: 0.1, 5: 0.1, 6: 0.2, 7: 0.3, 8: 0.4, 9: 0.5,
            10: 0.7, 11: 1.0, 12: 1.5, 13: 2.0, 14: 2.5, 
            15: 3.0, 16: 4.0, 17: 5.0, 18: 6.0, 19: 8.0,
            20: 10.0, 21: 12.0, 22: 14.0, 23: 16.0, 24: 18.0,
            25: 20.0, 26: 22.0, 27: 24.0, 28: 26.0, 29: 28.0,
            30: 30.0
        }
        
        self.risk_categories = {
            'LOW': (0, 10),
            'INTERMEDIATE': (11, 20),
            'HIGH': (21, 100)
        }
    
    def calculate_complete_frs(self, age, total_chol, hdl, systolic_bp, smoker, diabetic, gender='male', bp_treated=False):
        
        try:
          
            age = int(age)
            total_chol = int(total_chol)
            hdl = int(hdl)
            systolic_bp = int(systolic_bp)
            
            points = self.calculate_frs_points(age, total_chol, hdl, systolic_bp, smoker, diabetic, gender, bp_treated)
            risk_percentage = self.calculate_10_year_risk(points, gender)
            risk_category = self.get_risk_category(risk_percentage)
            
           
            if risk_category == 'LOW':
                color = '#10b981'  
                emoji = '🟢'
                recommendation = f'Risiko rendah ({risk_percentage}%). Pertahankan gaya hidup sehat.'
            elif risk_category == 'INTERMEDIATE':
                color = '#f59e0b'  
                emoji = '🟡'
                recommendation = f'Risiko sedang ({risk_percentage}%). Perbaiki gaya hidup dan konsultasi dokter.'
            else: 
                color = '#ef4444'  
                emoji = '🔴'
                recommendation = f'RISIKO TINGGI ({risk_percentage}%)! Evaluasi medis segera diperlukan!'
            
           
            components = {
                'age': age,
                'total_cholesterol': total_chol,
                'hdl_cholesterol': hdl,
                'systolic_bp': systolic_bp,
                'smoker': smoker,
                'diabetic': diabetic,
                'bp_treated': bp_treated,
                'gender': gender
            }
            
           
            dominant_factors = []
            if total_chol >= 240:
                dominant_factors.append(f"Kolesterol tinggi ({total_chol} mg/dL)")
            if hdl < 40:
                dominant_factors.append(f"HDL rendah ({hdl} mg/dL)")
            if systolic_bp >= 140:
                dominant_factors.append(f"Hipertensi ({systolic_bp} mmHg)")
            if smoker:
                dominant_factors.append("Merokok")
            if diabetic:
                dominant_factors.append("Diabetes")
            
            return {
                'points': points,
                'risk_percentage': round(risk_percentage, 1),
                'risk_category': risk_category,
                'color': color,
                'emoji': emoji,
                'recommendation': recommendation,
                'components': components,
                'dominant_factors': dominant_factors,
                'has_lab_data': True,
                'note': f'Framingham 10-Year Risk Score ({gender})',
                'formula_used': 'Original Framingham with corrected scoring'
            }
            
        except Exception as e:
            logger.error(f"Error in FRS calculation: {e}")
            return {
                'has_lab_data': False,
                'error': f"Kesalahan perhitungan: {str(e)}",
                'recommendation': 'Periksa data input Anda'
            }
    
    def calculate_frs_points(self, age, total_chol, hdl, systolic_bp, smoker, diabetic, gender='male', bp_treated=False):
        
        points = 0
        table = self.male_points if gender.lower() == 'male' else self.female_points
        
      
        if age < 35:
            points += -5
        elif age < 45:
            points += 0
        elif age < 55:
            points += 5
        elif age < 65:
            points += 8
        else:
            points += 10
        
        
        if total_chol < 160:
            points += 0
        elif total_chol < 200:
            points += 2
        elif total_chol < 240:
            points += 4
        elif total_chol < 280:
            points += 6
        else:  
            points += 8  
        
        
        if hdl < 40:
            points += 4  
        elif hdl < 50:
            points += 2
        elif hdl < 60:
            points += 0
        else:  
            points += -2
        
        
        bp_status = 'treated' if bp_treated else 'untreated'
        if systolic_bp < 120:
            points += -1
        elif systolic_bp < 130:
            points += 0
        elif systolic_bp < 140:
            points += 1
        elif systolic_bp < 160:
            points += 3
        else: 
            points += 5
        
       
        if smoker:
            points += 4  
        
        
        if diabetic:
            points += 3  
        
        return points
    
    def calculate_10_year_risk(self, points, gender='male'):
        
        risk_table = self.male_risk if gender.lower() == 'male' else self.female_risk
        
      
        if points <= 0:
            return 0.1
        
        
        if points in risk_table:
            return risk_table[points]
        
      
        sorted_points = sorted(risk_table.keys())
        
        if points < sorted_points[0]:
            return risk_table[sorted_points[0]]
        elif points > sorted_points[-1]:
            return risk_table[sorted_points[-1]]
        else:
            for i in range(len(sorted_points)-1):
                if sorted_points[i] <= points <= sorted_points[i+1]:
                    p1, r1 = sorted_points[i], risk_table[sorted_points[i]]
                    p2, r2 = sorted_points[i+1], risk_table[sorted_points[i+1]]
                    risk = r1 + (r2 - r1) * (points - p1) / (p2 - p1)
                    return risk
        
        return 0.1
    
    def get_risk_category(self, risk_percentage):
        
        for category, (min_val, max_val) in self.risk_categories.items():
            if min_val <= risk_percentage <= max_val:
                return category
        return 'LOW'


class OptimizedRiskScoreCalculator:
   
    def __init__(self):
        self.risk_thresholds = {
            'SANGAT_RENDAH': (0, 20),
            'RENDAH': (21, 40),
            'SEDANG': (41, 55),
            'SEDANG_TINGGI': (56, 70),
            'TINGGI': (71, 85),
            'SANGAT_TINGGI': (86, 100)
        }
        
        self.weights = {
            'blood_pressure': 0.30,
            'heart_rate': 0.20,
            'temperature': 0.15,
            'age': 0.10,
            'bmi': 0.05,
            'lifestyle': 0.03,
            'family_history': 0.10,
            'diabetes': 0.07
        }
    
    def calculate_component_scores(self, systole, diastole, heart_rate, temperature, age, bmi, smoking, active,
                                  diabetic=False, family_history=False, family_history_score=0, family_history_type='none'):
        
        scores = {}
        
        
        if systole >= 180 or diastole >= 120:
            bp_score = 35
        elif systole >= 160 or diastole >= 100:
            bp_score = 30
        elif systole >= 140 or diastole >= 90:
            bp_score = 25
        elif systole >= 130 or diastole >= 85:
            bp_score = 18
        elif systole >= 120:
            bp_score = 10
        elif systole < 90 or diastole < 60:
            bp_score = 8
        else:
            bp_score = 0
        scores['blood_pressure'] = bp_score
        
      
        if heart_rate >= 150:
            hr_score = 25
        elif heart_rate >= 130:
            hr_score = 20
        elif heart_rate >= 120:
            hr_score = 16
        elif heart_rate >= 100:
            hr_score = 12
        elif heart_rate <= 40:
            hr_score = 25
        elif heart_rate <= 50:
            hr_score = 18
        elif heart_rate <= 59:
            hr_score = 10
        else:
            hr_score = 0
        scores['heart_rate'] = hr_score
        
        
        if temperature >= 40.0:
            temp_score = 20
        elif temperature >= 39.5:
            temp_score = 18
        elif temperature >= 39.0:
            temp_score = 16
        elif temperature >= 38.5:
            temp_score = 14
        elif temperature >= 38.0:
            temp_score = 12
        elif temperature >= 37.5:
            temp_score = 8
        elif temperature <= 35.0:
            temp_score = 15
        elif temperature <= 36.0:
            temp_score = 8
        else:
            temp_score = 0
        scores['temperature'] = temp_score
        
       
        if age >= 75:
            age_score = 12
        elif age >= 65:
            age_score = 10
        elif age >= 55:
            age_score = 8
        elif age >= 45:
            age_score = 6
        elif age >= 35:
            age_score = 4
        elif age < 18:
            age_score = 2
        else:
            age_score = 0
        scores['age'] = age_score
        
       
        if bmi >= 35:
            bmi_score = 5
        elif bmi >= 30:
            bmi_score = 4
        elif bmi >= 27.5:
            bmi_score = 3
        elif bmi >= 25:
            bmi_score = 2
        elif bmi < 18.5:
            bmi_score = 3
        else:
            bmi_score = 0
        scores['bmi'] = bmi_score
        
       
        lifestyle_score = 0
        if smoking == 'current':
            lifestyle_score += 2
        elif smoking == 'former':
            lifestyle_score += 1
        if not active:
            lifestyle_score += 1
        scores['lifestyle'] = min(3, lifestyle_score)
        
      
        scores['diabetes'] = 7 if diabetic else 0
        
        
        family_history_val = 0
        if family_history:
            base_points = 3
            
            if family_history_score >= 7:
                base_points += 4
            elif family_history_score >= 4:
                base_points += 3
            elif family_history_score >= 1:
                base_points += 2
            
            if family_history_type == 'premature':
                base_points += 3
            elif family_history_type == 'non-premature':
                base_points += 1
            
            family_history_val = min(10, base_points)
        
        scores['family_history'] = family_history_val
        
        return scores
    
    def calculate_family_history_impact(self, family_history, family_history_score, family_history_type, age, condition_scores):
        
        impact_multiplier = 1.0
        impact_description = ""
        
        if family_history:
            impact_multiplier = 1.2
            
            if family_history_score >= 7:
                impact_multiplier = 1.4
                impact_description = "RIWAYAT KELUARGA KUAT: Multiple affected relatives, risiko sangat meningkat"
            elif family_history_score >= 4:
                impact_multiplier = 1.3
                impact_description = "RIWAYAT KELUARGA SIGNIFIKAN: Risiko meningkat secara signifikan"
            
            if family_history_type == 'premature':
                impact_multiplier *= 1.15
                if not impact_description:
                    impact_description = "RIWAYAT KELUARGA PREMATURE: Onset dini meningkatkan risiko"
                else:
                    impact_description += ", onset dini"
            
            if age < 50 and family_history:
                impact_multiplier *= 1.1
        
        return impact_multiplier, impact_description
    
    def calculate_fever_impact(self, temperature, condition_scores):
       
        impact_multiplier = 1.0
        impact_description = ""
        
        if temperature >= 39.0:
            impact_multiplier = 1.4
            impact_description = "DEMAM TINGGI: Meningkatkan beban jantung secara signifikan"
        elif temperature >= 38.5:
            impact_multiplier = 1.3
            impact_description = "DEMAM: Meningkatkan risiko komplikasi kardiovaskular"
        elif temperature >= 38.0:
            impact_multiplier = 1.2
            impact_description = "DEMAM: Meningkatkan heart rate dan tekanan darah"
        elif temperature >= 37.5:
            impact_multiplier = 1.1
            impact_description = "DEMAM RINGAN: Sedikit meningkatkan beban kardiovaskular"
        elif temperature <= 36.0:
            impact_multiplier = 1.15
            impact_description = "HIPOTERMIA: Meningkatkan risiko aritmia"
        
        if condition_scores['heart_rate'] > 0 or condition_scores['blood_pressure'] > 0:
            impact_multiplier *= 1.1
        
        return impact_multiplier, impact_description
    
    def calculate_optimized_risk_score(self, systole, diastole, heart_rate, temperature, age, bmi, 
                                      smoking_status, active_lifestyle, diabetic=False, family_history=False,
                                      family_history_score=0, family_history_type='none'):
       
        component_scores = self.calculate_component_scores(
            systole, diastole, heart_rate, temperature, age, bmi, 
            smoking_status, active_lifestyle, diabetic, family_history, 
            family_history_score, family_history_type
        )
        
        fever_multiplier, fever_description = self.calculate_fever_impact(temperature, component_scores)
        family_multiplier, family_description = self.calculate_family_history_impact(
            family_history, family_history_score, family_history_type, age, component_scores
        )
        
        base_score = 0
        max_possible_score = 0
        
        component_max = {
            'blood_pressure': 35,
            'heart_rate': 25,
            'temperature': 20,
            'age': 12,
            'bmi': 5,
            'lifestyle': 3,
            'diabetes': 7,
            'family_history': 10
        }
        
        for component, score in component_scores.items():
            base_score += (score * self.weights[component])
            max_possible_score += (component_max[component] * self.weights[component])
        
        normalized_score = (base_score / max_possible_score * 100) if max_possible_score > 0 else 0
        adjusted_score = normalized_score * fever_multiplier * family_multiplier
        
        abnormal_count = sum(1 for k in ['blood_pressure', 'heart_rate', 'temperature'] if component_scores[k] > 0)
        
        if abnormal_count >= 3:
            adjusted_score *= 1.2
        elif abnormal_count == 2:
            adjusted_score *= 1.1
        
        final_score = min(100, max(0, round(adjusted_score)))
        combined_description = f"{fever_description} {' ' + family_description if family_description else ''}"
        
        return final_score, component_scores, combined_description.strip()
    
    def determine_optimized_risk_level(self, risk_score, combined_description=""):
        """Menentukan level risiko"""
        if risk_score >= 86:
            return 'SANGAT_TINGGI', '#FF0000', '🔴', f'Segera konsultasi dokter! {combined_description}'
        elif risk_score >= 71:
            return 'TINGGI', '#FF6B00', '🟠', f'Konsultasi dokter dalam 1-2 hari. {combined_description}'
        elif risk_score >= 56:
            return 'SEDANG_TINGGI', '#FFD700', '🟡', f'Konsultasi dokter dalam 1 minggu. {combined_description}'
        elif risk_score >= 36:
            return 'SEDANG', '#87CEEB', '🔵', f'Perbaiki gaya hidup. {combined_description}'
        elif risk_score >= 16:
            return 'RENDAH', '#90EE90', '🟢', f'Pertahankan gaya hidup sehat. {combined_description}'
        else:
            return 'SANGAT_RENDAH', '#008000', '✅', f'Kondisi sangat baik. {combined_description}'


class CardiovascularPredictor:
    def __init__(self, pipeline, label_encoder, risk_calculator, frs_calculator, feature_names):
        self.pipeline = pipeline
        self.label_encoder = label_encoder
        self.risk_calculator = risk_calculator
        self.frs_calculator = frs_calculator
        self.feature_names = feature_names or []
        
       
        self._analyze_features()
        
        
        self.condition_descriptions = {
            'NORMAL': 'Semua parameter vital dalam batas normal. Kondisi kesehatan baik.',
            'BORDERLINE_HYPERTENSION': 'Tekanan darah normal-tinggi (pre-hipertensi). Perlu modifikasi gaya hidup.',
            'HYPERTENSION_STAGE1': 'Hipertensi stage 1. Konsultasi dokter dan modifikasi gaya hidup diperlukan.',
            'HYPERTENSION_STAGE2': 'Hipertensi stage 2. Evaluasi medis dan terapi diperlukan.',
            'HYPERTENSION_CRISIS': 'Krisis hipertensi. PERLU PENANGANAN DARURAT!',
            'TACHYCARDIA': 'Denyut jantung meningkat (tachycardia). Evaluasi penyebab diperlukan.',
            'BRADYCARDIA': 'Denyut jantung rendah (bradycardia). Pemeriksaan lebih lanjut diperlukan.'
        }
        
        
        self.risk_level_descriptions = {
            'SANGAT_RENDAH': 'Risiko sangat rendah. Pertahankan gaya hidup sehat.',
            'RENDAH': 'Risiko rendah. Lanjutkan gaya hidup sehat.',
            'SEDANG': 'Risiko sedang. Perlu perbaikan gaya hidup.',
            'SEDANG_TINGGI': 'Risiko sedang-tinggi. Konsultasi dokter dalam 1 minggu.',
            'TINGGI': 'Risiko tinggi. Konsultasi dokter dalam 1-2 hari.',
            'SANGAT_TINGGI': 'Risiko sangat tinggi. Segera konsultasi dokter!'
        }

    def _analyze_features(self):
        
        self.numeric_features = []
        self.categorical_mappings = {}
        self.binary_features = []
        
        for feat in self.feature_names:
           
            if '_' in feat and any(x in feat for x in ['smoking_status', 'fever_severity', 
                                                       'gender', 'family_history_type', 
                                                       'inheritance_pattern']):
                parts = feat.split('_')
                if len(parts) >= 2:
                    base_feature = '_'.join(parts[:-1])
                    value = parts[-1]
                    if base_feature not in self.categorical_mappings:
                        self.categorical_mappings[base_feature] = []
                    self.categorical_mappings[base_feature].append(value)
            
          
            elif feat in ['family_history_cvd', 'diabetes', 'has_fever']:
                self.binary_features.append(feat)
            
           
            else:
                self.numeric_features.append(feat)
        
        logger.info(f"🔍 Feature analysis:")
        logger.info(f"   Numeric features: {len(self.numeric_features)}")
        logger.info(f"   Binary features: {self.binary_features}")
        logger.info(f"   Categorical mappings: {self.categorical_mappings}")

    def calculate_derived_features(self, systole, diastole, heart_rate, temperature):
       
        map_value = diastole + (systole - diastole) / 3
        
        
        pulse_pressure = systole - diastole
        
        
        if heart_rate < 60:
            hr_variability = np.random.normal(25, 8)
        elif heart_rate > 100:
            hr_variability = np.random.normal(10, 5)
        else:
            hr_variability = np.random.normal(15, 6)
        
        if temperature >= 38.0:
            hr_variability *= 1.5
        
        hr_variability = max(5, min(50, hr_variability))
        
        return {
            'map': round(map_value, 1),
            'pulse_pressure': round(pulse_pressure, 1),
            'hr_variability': round(hr_variability, 1)
        }

    def prepare_input_data(self, systole, diastole, heart_rate, temperature,
                      age=None, bmi=None, smoking_status=None, active_lifestyle=None,
                      diabetes=None, family_history_cvd=None, family_history_score=None,
                      family_history_type=None, gender=None, inheritance_pattern=None):
       
        age = age if age is not None else 45
        bmi = bmi if bmi is not None else 24.0
        smoking_status = smoking_status if smoking_status is not None else 'never'
        active_lifestyle = active_lifestyle if active_lifestyle is not None else True
        diabetes = diabetes if diabetes is not None else False
        family_history_cvd = family_history_cvd if family_history_cvd is not None else False
        family_history_score = family_history_score if family_history_score is not None else 0
        family_history_type = family_history_type if family_history_type is not None else 'none'
        gender = gender if gender is not None else 'male'
        inheritance_pattern = inheritance_pattern if inheritance_pattern is not None else 'none'

        
        try:
            systole = float(systole)
            diastole = float(diastole)
            heart_rate = float(heart_rate)
            temperature = float(temperature)
            age = int(age)
            bmi = float(bmi)
            family_history_score = int(family_history_score)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input type: {e}")
        
       
        if systole <= diastole:
            raise ValueError(f"Systole ({systole}) harus lebih besar dari Diastole ({diastole})")
        
      
        derived = self.calculate_derived_features(systole, diastole, heart_rate, temperature)
        
       
        if temperature >= 39.0:
            fever_severity = 'high'
        elif temperature >= 38.0:
            fever_severity = 'moderate'
        elif temperature >= 37.5:
            fever_severity = 'mild'
        else:
            fever_severity = 'none'
        
        has_fever = temperature >= 37.5
        
       
        input_data = {}
        
       
        for feature in self.feature_names:
            input_data[feature] = 0.0
        
        
        numeric_map = {
            'systole': systole,
            'diastole': diastole,
            'heart_rate': heart_rate,
            'temperature': temperature,
            'age': age,
            'bmi': bmi,
            'map': derived['map'],
            'pulse_pressure': derived['pulse_pressure'],
            'hr_variability': derived['hr_variability'],
            'family_history_score': family_history_score
        }
        
        for feat, value in numeric_map.items():
            if feat in self.feature_names:
                input_data[feat] = value
        
        
        smoking_col = f"smoking_status_{smoking_status}"
        if smoking_col in self.feature_names:
            input_data[smoking_col] = 1.0
        
        
        active_str = "true" if active_lifestyle else "false"
        active_col = f"active_lifestyle_{active_str}"
        if active_col in self.feature_names:
            input_data[active_col] = 1.0
        
       
        fever_col = f"fever_severity_{fever_severity}"
        if fever_col in self.feature_names:
            input_data[fever_col] = 1.0
        
        
        family_type_col = f"family_history_type_{family_history_type}"
        if family_type_col in self.feature_names:
            input_data[family_type_col] = 1.0
        
        
        gender_col = f"gender_{gender}"
        if gender_col in self.feature_names:
            input_data[gender_col] = 1.0
        
       
        inherit_col = f"inheritance_pattern_{inheritance_pattern}"
        if inherit_col in self.feature_names:
            input_data[inherit_col] = 1.0
        
       
        binary_map = {
            'family_history_cvd': 1.0 if family_history_cvd else 0.0,
            'diabetes': 1.0 if diabetes else 0.0,
            'has_fever': 1.0 if has_fever else 0.0
        }
        
        for feat, value in binary_map.items():
            if feat in self.feature_names:
                input_data[feat] = value
        
        
        input_df = pd.DataFrame([input_data])
        
        
        if self.feature_names:
            
            missing_in_df = [f for f in self.feature_names if f not in input_df.columns]
            extra_in_df = [f for f in input_df.columns if f not in self.feature_names]
            
            if missing_in_df:
                logger.warning(f" Missing features in input_df: {missing_in_df}")
               
                for feat in missing_in_df:
                    input_df[feat] = 0.0
            
            if extra_in_df:
                logger.warning(f" Extra features in input_df: {extra_in_df}")
               
                input_df = input_df.drop(columns=extra_in_df)
            
           
            input_df = input_df[self.feature_names]
        
        
        logger.info(f" Input shape: {input_df.shape}")
        logger.info(f" Feature count: {len(self.feature_names)}")
        
        return input_df

    def predict_with_risk_scoring(self, input_df, frs_params=None, 
                                 systole=None, diastole=None, heart_rate=None, 
                                 temperature=None, age=None, bmi=None,
                                 smoking_status=None, active_lifestyle=None,
                                 diabetes=None, family_history_cvd=None,
                                 family_history_score=None, family_history_type=None):
       
        try:
            if input_df.empty:
                raise ValueError("Input data kosong")
            
            if self.pipeline is None:
                raise ValueError("Model pipeline tidak tersedia")
            
          
            predicted_condition = self.pipeline.predict(input_df)[0]
            proba = self.pipeline.predict_proba(input_df)[0]
            confidence = np.max(proba) * 100
            
           
            probabilities_dict = {}
            probabilities_percent = {}
            
            if self.label_encoder:
                for i, class_name in enumerate(self.label_encoder.classes_):
                    probabilities_dict[class_name] = float(proba[i])
                    probabilities_percent[class_name] = float(proba[i] * 100)
            
            
            systole = systole if systole is not None else 120.0
            diastole = diastole if diastole is not None else 80.0
            heart_rate = heart_rate if heart_rate is not None else 72.0
            temperature = temperature if temperature is not None else 36.5
            age = age if age is not None else 45
            bmi = bmi if bmi is not None else 24.0
            smoking_status = smoking_status if smoking_status is not None else 'never'
            active_lifestyle = active_lifestyle if active_lifestyle is not None else True
            diabetes = diabetes if diabetes is not None else False
            family_history_cvd = family_history_cvd if family_history_cvd is not None else False
            family_history_score = family_history_score if family_history_score is not None else 0
            family_history_type = family_history_type if family_history_type is not None else 'none'
            
           
            custom_risk_score, components, combined_description = self.risk_calculator.calculate_optimized_risk_score(
                systole, diastole,
                heart_rate, temperature,
                age, bmi,
                smoking_status, active_lifestyle,
                diabetes,
                family_history_cvd,
                family_history_score,
                family_history_type
            )
            
            custom_risk_level, custom_risk_color, custom_risk_emoji, custom_risk_recommendation = self.risk_calculator.determine_optimized_risk_level(
                custom_risk_score, combined_description
            )
            
            
            frs_result = None
            has_lab_data = False
            
            try:
                if frs_params and frs_params.get('total_cholesterol') is not None and frs_params.get('hdl_cholesterol') is not None:
                    has_lab_data = True
                    smoker_for_frs = frs_params.get('smoking_status', 'never') == 'current'
                    
                    frs_result = self.frs_calculator.calculate_complete_frs(
                        age=frs_params.get('age', 45),
                        total_chol=frs_params.get('total_cholesterol'),
                        hdl=frs_params.get('hdl_cholesterol'),
                        systolic_bp=frs_params.get('systole', 120),
                        smoker=smoker_for_frs,
                        diabetic=frs_params.get('diabetes', False),
                        gender=frs_params.get('gender', 'male'),
                        bp_treated=frs_params.get('bp_treated', False)
                    )
                    frs_result['has_lab_data'] = True
                    frs_result['note'] = 'Framingham 10-Year Risk Score (memerlukan data kolesterol)'
                else:
                    frs_result = {
                        'has_lab_data': False,
                        'note': 'Framingham Risk Score tidak dapat dihitung',
                        'reason': 'Data kolesterol tidak tersedia',
                        'recommendation': 'Tambahkan data kolesterol total dan HDL untuk prediksi risiko 10 tahun',
                        'requires_parameters': ['total_cholesterol', 'hdl_cholesterol']
                    }
                    
            except Exception as e:
                logger.error(f"Error calculating FRS: {e}")
                frs_result = {
                    'has_lab_data': False,
                    'note': 'Error menghitung Framingham Risk Score',
                    'error': str(e)
                }
            
            
            personalized_recommendations = self.generate_recommendations(
                predicted_condition, bmi, diabetes, smoking_status,
                active_lifestyle, family_history_cvd, family_history_type,
                custom_risk_level, age, heart_rate, systole, diastole
            )
            
            response = {
                'success': True,
                'condition': predicted_condition,
                'condition_description': self.condition_descriptions.get(predicted_condition, ''),
                'confidence_percent': round(confidence, 1),
                'probabilities': probabilities_dict,
                'probabilities_percent': probabilities_percent,
                'custom_risk_score': custom_risk_score,
                'custom_risk_level': custom_risk_level,
                'custom_risk_color': custom_risk_color,
                'custom_risk_emoji': custom_risk_emoji,
                'custom_risk_recommendation': custom_risk_recommendation,
                'combined_description': combined_description,
                'top_predictions': [
                    {'condition': cond, 'probability': round(prob * 100, 1)}
                    for cond, prob in zip(self.label_encoder.classes_, proba)
                ][:3],
                'risk_components': components,
                'framingham_risk_score': frs_result,
                'lab_data_available': has_lab_data,
                'input_parameters': {
                    'vitals': {
                        'systole': systole,
                        'diastole': diastole,
                        'heart_rate': heart_rate,
                        'temperature': temperature,
                        'has_fever': temperature >= 37.5,
                        'fever_severity': 'high' if temperature >= 39.0 else 
                                        'moderate' if temperature >= 38.0 else 
                                        'mild' if temperature >= 37.5 else 'none'
                    },
                    'demographics': {
                        'age': age,
                        'bmi': bmi
                    },
                    'lifestyle': {
                        'smoking_status': smoking_status,
                        'active_lifestyle': active_lifestyle,
                        'diabetes': diabetes
                    },
                    'family_history': {
                        'has_history': family_history_cvd,
                        'score': family_history_score,
                        'type': family_history_type
                    }
                },
                'personalized_recommendations': personalized_recommendations
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'condition': 'ERROR',
                'confidence': 0.0
            }

    def generate_recommendations(self, condition, bmi, diabetes, smoking_status, 
                                 active_lifestyle, family_history_cvd, family_history_type,
                                 risk_level, age, heart_rate, systole, diastole):
       
        recommendations = []
        
       
        if condition == 'NORMAL':
            recommendations.append("Pertahankan gaya hidup sehat dengan pola makan seimbang")
            recommendations.append("Olahraga rutin 30 menit sehari, 5 kali seminggu")
            recommendations.append("Cek tekanan darah rutin setiap 6 bulan")
        
        elif condition == 'BORDERLINE_HYPERTENSION':
            recommendations.append("Mulai diet rendah garam (<5g/hari)")
            recommendations.append("Kurangi konsumsi makanan olahan dan tinggi sodium")
            recommendations.append("Mulai program olahraga ringan seperti jalan cepat 30 menit/hari")
            recommendations.append("Monitor tekanan darah mingguan")
        
        elif condition == 'HYPERTENSION_STAGE1':
            recommendations.append("Segera konsultasi dokter untuk evaluasi medis")
            recommendations.append("Diet DASH (Dietary Approaches to Stop Hypertension)")
            recommendations.append("Olahraga aerobik 150 menit/minggu")
            recommendations.append("Pantau tekanan darah harian, catat dalam log")
        
        elif condition == 'HYPERTENSION_STAGE2':
            recommendations.append("Konsultasi dokter dalam 1-2 minggu untuk terapi obat")
            recommendations.append("Diet ketat rendah garam dan rendah lemak jenuh")
            recommendations.append("Olahraga terpantau sesuai kondisi")
            recommendations.append("Cek darah rutin (kolesterol, gula darah)")
        
        elif condition == 'HYPERTENSION_CRISIS':
            recommendations.append("SEGERA KE RUMAH SAKIT atau UGD TERDEKAT")
            recommendations.append("Jangan menunda pengobatan")
            recommendations.append("Duduk tenang sambil menunggu bantuan medis")
            recommendations.append("Catat gejala yang dirasakan untuk informasi dokter")
        
        elif condition == 'TACHYCARDIA':
            recommendations.append("Konsultasi kardiologis untuk evaluasi jantung")
            recommendations.append("Hindari kafein dan stimulan")
            recommendations.append("Latihan pernapasan untuk mengatur denyut jantung")
            recommendations.append("Monitor denyut jantung harian")
        
        elif condition == 'BRADYCARDIA':
            recommendations.append("Konsultasi kardiologis untuk evaluasi pacemaker jantung")
            recommendations.append("Hindari obat-obatan yang memperlambat denyut jantung")
            recommendations.append("Monitor gejala pusing atau pingsan")
            recommendations.append("Cek EKG untuk menilai ritme jantung")
        
       
        if bmi >= 27.5:
            recommendations.append("Program penurunan berat badan: target turun 5-10% dari berat saat ini")
            recommendations.append("Konsultasi gizi untuk diet rendah kalori seimbang")
            recommendations.append("Olahraga kombinasi kardio dan strength training")
            recommendations.append("Pantau lingkar perut (pria <90cm, wanita <80cm)")
        
        elif bmi >= 25:
            recommendations.append("Pertahankan berat badan ideal untuk mencegah obesitas")
            recommendations.append("Diet rendah gula dan lemak")
            recommendations.append("Olahraga rutin 150 menit/minggu")
        
        elif bmi < 18.5:
            recommendations.append("Tingkatkan asupan kalori sehat untuk mencapai berat normal")
            recommendations.append("Konsultasi gizi untuk peningkatan berat badan sehat")
            recommendations.append("Latihan beban untuk membangun massa otot")
        
      
        if diabetes:
            recommendations.append("Kontrol gula darah ketat (HbA1c <7%)")
            recommendations.append("Diet diabetes: rendah GI, tinggi serat")
            recommendations.append("Monitor gula darah rutin")
            recommendations.append("Cek kesehatan kaki dan mata secara berkala")
        
       
        if smoking_status == 'current':
            recommendations.append("PROGRAM BERHENTI MEROKOK segera")
            recommendations.append("Konsultasi dokter untuk terapi pengganti nikotin")
            recommendations.append("Hindari lingkungan yang memicu keinginan merokok")
            recommendations.append("Gabung support group untuk berhenti merokok")
        
       
        if not active_lifestyle:
            recommendations.append("Mulai aktivitas fisik dari yang ringan, seperti jalan kaki 10 menit/hari")
            recommendations.append("Gunakan pedometer untuk target 7.000-10.000 langkah/hari")
            recommendations.append("Kurangi waktu duduk, berdiri setiap 30 menit")
        
       
        if family_history_cvd:
            recommendations.append("Screening kardiovaskular lebih dini karena riwayat keluarga")
            if family_history_type == 'premature':
                recommendations.append("Pemeriksaan jantung lengkap karena riwayat premature")
            recommendations.append("Kontrol faktor risiko ekstra ketat")
            recommendations.append("Edukasi keluarga tentang pencegahan penyakit jantung")
        
        
        if age >= 50:
            recommendations.append("Screening kesehatan tahunan (kolesterol, gula darah, EKG)")
            recommendations.append("Diet Mediterania untuk kesehatan jantung")
            recommendations.append("Aktivitas fisik sesuai kemampuan, hindari high-impact")
        
       
        if systole >= 140 or diastole >= 90:
            recommendations.append("Monitor tekanan darah 2x sehari (pagi dan malam)")
            recommendations.append("Catat dalam log tekanan darah")
            recommendations.append("Kurangi stres dengan teknik relaksasi")
        
       
        if heart_rate > 100:
            recommendations.append("Hindari stimulan (kafein, energi drink)")
            recommendations.append("Latihan relaksasi dan pernapasan dalam")
            recommendations.append("Cukup tidur 7-8 jam/hari")
        
       
        if risk_level in ['TINGGI', 'SANGAT_TINGGI']:
            recommendations.insert(0, "SEGERA KONSULTASI DOKTER atau ke IGD")
        elif risk_level == 'SEDANG_TINGGI':
            recommendations.insert(0, "Konsultasi dokter dalam 1 minggu")
        elif risk_level == 'SEDANG':
            recommendations.insert(0, "Konsultasi dokter dalam 1-2 bulan")
        
        return recommendations[:10]  


MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "api_model.joblib")


app = Flask(__name__)


logger.info(f"\n Loading model from: {MODEL_PATH}")

use_model = False
model_data = None
pipeline = None
label_encoder = None
risk_calculator = None
frs_calculator = None
feature_names = None
model_info = None
predictor = None

if os.path.exists(MODEL_PATH):
    try:
        logger.info(f" Model file exists, loading...")
        
        # Inject HybridRFSVM ke __main__ untuk backward compatibility
        import __main__
        __main__.HybridRFSVM = HybridRFSVM
        
        model_data = joblib.load(MODEL_PATH)
        
      
        pipeline = model_data.get("pipeline")
        label_encoder = model_data.get("label_encoder")
        feature_names = model_data.get("feature_names")
        model_info = model_data.get("model_info", {})
        
        
        risk_calculator = OptimizedRiskScoreCalculator()
        frs_calculator = FraminghamRiskScoreCalculator()
        
      
        logger.info(f" Model verification:")
        logger.info(f"   Pipeline type: {type(pipeline).__name__ if pipeline else 'None'}")
        logger.info(f"   Label encoder: {'Loaded' if label_encoder else 'Not loaded'}")
        logger.info(f"   Feature names count: {len(feature_names) if feature_names else 0}")
        
        if pipeline is not None and label_encoder is not None and feature_names:
            predictor = CardiovascularPredictor(
                pipeline=pipeline,
                label_encoder=label_encoder,
                risk_calculator=risk_calculator,
                frs_calculator=frs_calculator,
                feature_names=feature_names
            )
            use_model = True
            logger.info(" Model loaded successfully!")
            logger.info(f"   Model: {model_info.get('name', 'Cardiovascular Risk Predictor')}")
            logger.info(f"   Version: {model_info.get('version', 'v6.0')}")
            logger.info(f"   Features: {len(feature_names)}")
            
            if label_encoder:
                classes = list(label_encoder.classes_)
                logger.info(f"   Classes ({len(classes)}): {classes}")
        else:
            logger.error(" Model components missing:")
            logger.error(f"   Pipeline: {'OK' if pipeline else 'MISSING'}")
            logger.error(f"   Label encoder: {'OK' if label_encoder else 'MISSING'}")
            logger.error(f"   Feature names: {'OK' if feature_names else 'MISSING'}")
            use_model = False
        
    except Exception as e:
        logger.error(f" Error loading model: {e}", exc_info=True)
        use_model = False
else:
    logger.error(f" Model not found at {MODEL_PATH}")
    logger.error("   Please run train.py first to generate model")


@app.route("/")
def index():
   
    return render_template('index.html')

@app.route("/model-info")
def get_model_info():
    
    if not use_model:
        return jsonify({
            "error": "Model tidak tersedia",
            "status": "model_not_loaded"
        }), 503
    
    classes = []
    if label_encoder:
        classes = list(label_encoder.classes_)
    
    return jsonify({
        "status": "loaded",
        "model_name": model_info.get("name", "Cardiovascular Risk Predictor v6.0"),
        "version": model_info.get("version", "6.0-api-ready"),
        "description": "Hybrid RF+SVM model for cardiovascular condition prediction",
        "features_included": {
            "family_history": True,
            "framingham_score": True,
            "fever_as_risk_factor": True,
            "feature_count": len(feature_names) if feature_names else 0
        },
        "classes": classes,
        "class_count": len(classes),
        "created_date": model_info.get("created_date", ""),
        "required_parameters": {
            "mandatory": ["systole", "diastole", "heart_rate", "temperature"],
            "recommended": ["age", "bmi", "smoking_status", "gender"],
            "optional": ["family_history_cvd", "family_history_score", "family_history_type", 
                        "diabetes", "active_lifestyle", "inheritance_pattern"],
            "lab_optional": ["total_cholesterol", "hdl_cholesterol", "bp_treated"]
        },
        "model_features_sample": feature_names[:10] if feature_names else []
    })

@app.route("/predict", methods=["POST"])
def predict():
    
    if not use_model or predictor is None:
        return jsonify({
            "success": False,
            "error": "Model tidak tersedia",
            "status": "model_not_loaded"
        }), 503
    
    try:
        data = request.get_json(force=True)
        logger.info(f" Received prediction request")
        
        
        required_params = ['systole', 'diastole', 'heart_rate', 'temperature']
        missing_params = [param for param in required_params if param not in data]
        
        if missing_params:
            return jsonify({
                "success": False,
                "error": f"Parameter berikut diperlukan: {', '.join(missing_params)}",
                "status": "missing_parameter"
            }), 400
        
        
        systole = float(data.get("systole", 120))
        diastole = float(data.get("diastole", 80))
        heart_rate = float(data.get("heart_rate", 72))
        temperature = float(data.get("temperature", 36.5))
        
        
        if systole <= diastole:
            return jsonify({
                "success": False,
                "error": f"Systole ({systole}) harus lebih besar dari Diastole ({diastole})",
                "status": "invalid_parameter"
            }), 400
        
       
        age = int(data.get("age", 45))
        bmi = float(data.get("bmi", 24.0))
        smoking_status = data.get("smoking_status", "never")
        active_lifestyle = parse_boolean(data.get("active_lifestyle", True))
        
        
        diabetes = parse_boolean(data.get("diabetes", False))
        
        gender = data.get("gender", "male")
        
       
        family_history_cvd = parse_boolean(data.get("family_history_cvd", False))
        family_history_score = int(data.get("family_history_score", 0))
        family_history_type = data.get("family_history_type", "none")
        inheritance_pattern = data.get("inheritance_pattern", "none")
        
       
        total_cholesterol = parse_float_or_none(data.get("total_cholesterol"))
        hdl_cholesterol = parse_float_or_none(data.get("hdl_cholesterol"))
        bp_treated = parse_boolean(data.get("bp_treated", False))
        
       
        has_lab_data = total_cholesterol is not None and hdl_cholesterol is not None
        
        logger.info(f" Data status: Lab data {'available' if has_lab_data else 'NOT available'}")
        if has_lab_data:
            logger.info(f"   Total Cholesterol: {total_cholesterol}, HDL: {hdl_cholesterol}")
        
       
        logger.info(f" Preparing input data...")
        input_df = predictor.prepare_input_data(
            systole=systole,
            diastole=diastole,
            heart_rate=heart_rate,
            temperature=temperature,
            age=age,
            bmi=bmi,
            smoking_status=smoking_status,
            active_lifestyle=active_lifestyle,
            diabetes=diabetes,  
            family_history_cvd=family_history_cvd,
            family_history_score=family_history_score,
            family_history_type=family_history_type,
            gender=gender,
            inheritance_pattern=inheritance_pattern
        )
        
        logger.info(f" Input data prepared. Shape: {input_df.shape}")
        
       
        frs_params = {
            'age': age,
            'diabetes': diabetes,  
            'gender': gender,
            'bp_treated': bp_treated,
            'systole': systole,
            'smoking_status': smoking_status
        }
        
        
        if has_lab_data:
            frs_params['total_cholesterol'] = total_cholesterol
            frs_params['hdl_cholesterol'] = hdl_cholesterol
        else:
            frs_params['total_cholesterol'] = None
            frs_params['hdl_cholesterol'] = None
        
      
        logger.info(f" Making prediction...")
        prediction_result = predictor.predict_with_risk_scoring(
            input_df=input_df,
            frs_params=frs_params,
            systole=systole,
            diastole=diastole,
            heart_rate=heart_rate,
            temperature=temperature,
            age=age,
            bmi=bmi,
            smoking_status=smoking_status,
            active_lifestyle=active_lifestyle,
            diabetes=diabetes,
            family_history_cvd=family_history_cvd,
            family_history_score=family_history_score,
            family_history_type=family_history_type
        )
        
        if not prediction_result['success']:
            return jsonify({
                "success": False,
                "error": prediction_result['error'],
                "condition": "ERROR"
            }), 500
        
        
        response = {
            "success": True,
            "status": "success",
            "model_version": model_info.get("version", "v6.0"),
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction": {
                "condition": prediction_result['condition'],
                "condition_description": prediction_result['condition_description'],
                "confidence_percent": prediction_result['confidence_percent'],
                "probabilities_percent": prediction_result['probabilities_percent'],
                "top_predictions": prediction_result['top_predictions']
            },
            "risk_assessment": {
                "level": prediction_result['custom_risk_level'],
                "score": prediction_result['custom_risk_score'],
                "recommendation": prediction_result['custom_risk_recommendation'],
                "description": prediction_result.get('combined_description', ''),
                "color": prediction_result['custom_risk_color'],
                "emoji": prediction_result['custom_risk_emoji'],
                "components": prediction_result.get('risk_components', {})
            },
           
            "framingham_risk_score": prediction_result.get('framingham_risk_score'),
            "lab_data_available": prediction_result.get('lab_data_available', False),
            
            "family_history_impact": {
                "has_history": family_history_cvd,
                "score": family_history_score,
                "type": family_history_type,
                "inheritance_pattern": inheritance_pattern
            },
            "personalized_recommendations": prediction_result.get('personalized_recommendations', []),
            "input_summary": {
                "vitals": {
                    "systole": systole,
                    "diastole": diastole,
                    "heart_rate": heart_rate,
                    "temperature": temperature,
                    "has_fever": temperature >= 37.5
                },
                "demographics": {
                    "age": age,
                    "bmi": bmi,
                    "gender": gender
                },
                "lifestyle": {
                    "smoking_status": smoking_status,
                    "active_lifestyle": active_lifestyle,
                    "diabetes": diabetes
                },
                "family_history": {
                    "has_history": family_history_cvd,
                    "score": family_history_score,
                    "type": family_history_type,
                    "inheritance_pattern": inheritance_pattern
                },
                "lab_data": {
                    "available": has_lab_data,
                    "total_cholesterol": total_cholesterol,
                    "hdl_cholesterol": hdl_cholesterol,
                    "bp_treated": bp_treated
                }
            },
            "medical_disclaimer": "HASIL INI HANYA SKRINING AWAL - KONSULTASIKAN DENGAN TENAGA MEDIS PROFESIONAL"
        }
        
        logger.info(f" Prediction successful: {prediction_result['condition']}")
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "condition": "INPUT_ERROR"
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Terjadi kesalahan sistem: {str(e)}",
            "condition": "SYSTEM_ERROR"
        }), 500

@app.route("/health")
def health_check():
   
    status_details = {
        "status": "healthy" if use_model else "model_not_loaded",
        "model_loaded": use_model,
        "pipeline_available": pipeline is not None,
        "label_encoder_available": label_encoder is not None,
        "features_available": feature_names is not None,
        "features_count": len(feature_names) if feature_names else 0,
        "conditions_available": list(label_encoder.classes_) if label_encoder else [],
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return jsonify(status_details)

@app.route("/debug-features", methods=["GET"])
def debug_features():
   
    if not use_model:
        return jsonify({"error": "Model not loaded"}), 503
    
    features_info = {
        "total_features": len(feature_names) if feature_names else 0,
        "features": feature_names if feature_names else [],
        "feature_sample": feature_names[:20] if feature_names and len(feature_names) > 20 else feature_names,
        "conditions": list(label_encoder.classes_) if label_encoder else []
    }
    
    return jsonify(features_info)

@app.route("/sample-inputs", methods=["GET"])
def get_sample_inputs():
   
    samples = [
        {
            "description": " SKENARIO 1: Data lengkap dengan lab (FRS AKTIF)",
            "scenario": "with_lab_data",
            "input": {
                "systole": 145,
                "diastole": 92,
                "heart_rate": 85,
                "temperature": 37.0,
                "gender": "male",
                "age": 48,
                "bmi": 28.5,
                "smoking_status": "former",
                "active_lifestyle": False,
                "diabetes": True,
                "family_history_cvd": True,
                "family_history_score": 7,
                "family_history_type": "non-premature",
                "inheritance_pattern": "paternal",
                "total_cholesterol": 240,
                "hdl_cholesterol": 38,
                "bp_treated": False
            }
        },
        {
            "description": " SKENARIO 2: Tanpa data lab (FRS TIDAK AKTIF)",
            "scenario": "without_lab_data",
            "input": {
                "systole": 112,
                "diastole": 74,
                "heart_rate": 72,
                "temperature": 36.5,
                "gender": "female",
                "age": 28,
                "bmi": 21.5,
                "smoking_status": "never",
                "active_lifestyle": True,
                "diabetes": False,
                "family_history_cvd": False,
                "family_history_score": 0,
                "family_history_type": "none",
                "inheritance_pattern": "none"
                
            }
        },
        {
            "description": " SKENARIO 3: Demam tinggi dengan lab data",
            "scenario": "fever_with_lab",
            "input": {
                "systole": 125,
                "diastole": 80,
                "heart_rate": 125,
                "temperature": 38.5,
                "gender": "female",
                "age": 30,
                "bmi": 24.0,
                "smoking_status": "never",
                "active_lifestyle": True,
                "diabetes": False,
                "family_history_cvd": False,
                "family_history_score": 0,
                "family_history_type": "none",
                "inheritance_pattern": "none",
                "total_cholesterol": 190,
                "hdl_cholesterol": 55,
                "bp_treated": False
            }
        },
        {
            "description": " SKENARIO 4: Hipertensi tanpa riwayat keluarga",
            "scenario": "hypertension_no_family",
            "input": {
                "systole": 160,
                "diastole": 95,
                "heart_rate": 78,
                "temperature": 36.8,
                "gender": "male",
                "age": 55,
                "bmi": 26.0,
                "smoking_status": "current",
                "active_lifestyle": False,
                "diabetes": False,
                "family_history_cvd": False,
                "family_history_score": 0,
                "family_history_type": "none",
                "inheritance_pattern": "none",
                "total_cholesterol": 210,
                "hdl_cholesterol": 45,
                "bp_treated": True
            }
        }
    ]
    
    return jsonify({
        "status": "success",
        "samples": samples,
        "note": "Gunakan SKENARIO 1 untuk Framingham Risk Score, SKENARIO 2 tanpa lab data",
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint tidak ditemukan"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method tidak diizinkan"
    }), 405

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": "Terjadi kesalahan internal server"
    }), 500


if __name__ == "__main__":
    print("=" * 80)
    print("CARDIOVASCULAR RISK PREDICTION API v6.0 - DUAL SCORING SYSTEM")
    print("COMPATIBLE WITH train.py v6.0")
    print("=" * 80)
    print(f"Model Status: {' LOADED' if use_model else ' NOT AVAILABLE'}")
    
    if use_model:
        print(f"\n MODEL INFORMATION:")
        print(f"   Name: {model_info.get('name', 'Cardiovascular Risk Predictor')}")
        print(f"   Version: {model_info.get('version', 'v6.0')}")
        print(f"   Features: {len(feature_names) if feature_names else 'Unknown'}")
        
        if label_encoder:
            classes = list(label_encoder.classes_)
            print(f"   Conditions ({len(classes)}): {', '.join(classes)}")
    
    print(f"\n DUAL SCORING SYSTEM:")
    print(f"    Clinical Risk Score: Selalu dihitung (tanpa data lab)")
    print(f"    Framingham Risk Score: Hanya jika ada data kolesterol")
    
    print(f"\n API ENDPOINTS:")
    print(f"   GET  /                 - Home")
    print(f"   GET  /model-info       - Model information")
    print(f"   GET  /health           - Health check")
    print(f"   GET  /debug-features   - Debug features")
    print(f"   GET  /sample-inputs    - Sample inputs (4 skenario)")
    print(f"   POST /predict          - Make prediction")
    
    print(f"\n SAMPLE CURL COMMANDS:")
    print(f"  1. Dengan data lab (FRS AKTIF):")
    print(f'     curl -X POST http://localhost:5000/predict \\')
    print(f'          -H "Content-Type: application/json" \\')
    print(f'          -d \'{{"systole": 120, "diastole": 80, "heart_rate": 72, "temperature": 36.5, "age": 45, "diabetes": false, "total_cholesterol": 200, "hdl_cholesterol": 50}}\'')
    
    print(f"\n  2. Tanpa data lab (FRS TIDAK AKTIF):")
    print(f'     curl -X POST http://localhost:5000/predict \\')
    print(f'          -H "Content-Type: application/json" \\')
    print(f'          -d \'{{"systole": 120, "diastole": 80, "heart_rate": 72, "temperature": 36.5, "age": 45, "diabetes": false}}\'')
    
    print(f"\n Starting server on http://localhost:5000")
    print("=" * 80)
    

    app.run(host="0.0.0.0", port=5000, debug=True)

