import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class InclusionExclusionTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms systematic review data into binary features based on expert inclusion/exclusion criteria.
    
    Each feature directly corresponds to a specific criterion in the expert-defined protocol,
    facilitating transparent feature importance analysis and model interpretation.
    """
    def __init__(self):
        # Population criteria
        self.under18_re = re.compile(r'(\bage range\b|\bage\b).+?(\d{1,2})[–\-\u2013]\d{1,2}', re.I)
        self.pediatric_re = re.compile(r'\b(pediatric|child|children|infant|neonate|newborn)\b', re.I)
        self.pregnancy_re = re.compile(r'\b(pregnan(t|cy)|expectant|gestation|fetus|uterus)\b', re.I)
        
        # Malformation types
        self.dural_fistula_re = re.compile(r'\bdural arteriovenous fistula\b', re.I)
        self.pial_fistula_re = re.compile(r'\bpial arteriovenous fistula\b', re.I)
        self.vein_galen_re = re.compile(r'\bvein of galen malformation\b', re.I)
        self.cavernous_re = re.compile(r'\bcavernous malformation\b', re.I)
        
        # Study design criteria
        self.trial_re = re.compile(r'\b(randomized controlled trial|clinical trial|cohort study|case series|systematic review)\b', re.I)
        self.meta_re = re.compile(r'\b(meta-?analysis|literature review)\b', re.I)
        self.case_study_re = re.compile(r'\bcase (study|report)\b', re.I)
        
        # Outcome reporting
        self.occl_re = re.compile(r'\b(occlusion rate|occlusion rates)\b', re.I)
        
        # Intervention exposure
        self.expo_re = re.compile(r'\b(gamma knife|cyberknife|novalis|linear accelerator.*?radiosurgery)\b', re.I)
        
        # Exclusion keywords
        self.auto_excl = re.compile(r'\b(hypofractionated|proton beam|fractionated stereotactic|tomotherapy)\b', re.I)
        
        # Patient minimum
        self.patnum_re = re.compile(r'\b(n ?= ?)(\d+)\b|\b(\d+)\s+patients?\b', re.I)
        
    def fit(self, X, y=None):
        """No fitting required for this transformer."""
        return self

    def transform(self, X):
        if hasattr(X, 'columns'):
            required_columns = ['title', 'abstract']
            missing_columns = [col for col in required_columns if col not in X.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            titles = np.asarray(X['title'].fillna(''))
            abstracts = np.asarray(X['abstract'].fillna(''))
            n = len(titles)
        else:
            combined_texts = X
            titles = np.full(len(combined_texts), '')
            abstracts = np.asarray(combined_texts)
            n = len(abstracts)
        
        feature_names = self.get_feature_names_out()
        n_features = len(feature_names)
        feats = np.zeros((n, n_features), dtype=int)

        for i, (t, a) in enumerate(zip(titles, abstracts)):
            txt = (t or "") + " " + (a or "")
            
            # Feature extraction logic
            m = self.under18_re.search(txt)
            feats[i, 0] = int(bool(m) and int(m.group(2)) < 18)
            
            feats[i, 1] = int(bool(self.pediatric_re.search(txt)))
            feats[i, 2] = int(bool(self.pregnancy_re.search(txt)))
            feats[i, 3] = int(bool(self.dural_fistula_re.search(txt) or 
                                self.pial_fistula_re.search(txt) or
                                self.vein_galen_re.search(txt) or
                                self.cavernous_re.search(txt)))
            feats[i, 4] = int(bool(self.trial_re.search(txt)))
            feats[i, 5] = int(bool(self.meta_re.search(txt)))
            feats[i, 6] = int(bool(self.occl_re.search(txt)))
            feats[i, 7] = int(bool(self.expo_re.search(txt)))
            feats[i, 8] = int(bool(self.auto_excl.search(txt)))
            
            m2 = self.patnum_re.search(txt)
            if m2:
                num = int(m2.group(2) or m2.group(3))
                feats[i, 9] = int(num < 30)
            else:
                feats[i, 9] = 0
            
            case_study = bool(self.case_study_re.search(txt))
            small_n = False
            if m2:
                small_n = int(m2.group(2) or m2.group(3)) < 10
            feats[i, 10] = int(case_study and (small_n or not m2))
            
            feats[i, 11] = int(feats[i, 0] or feats[i, 1] or feats[i, 2] or feats[i, 3])
        
        return feats
        
    def get_feature_names_out(self):
        """Return feature names for output features."""
        return np.array([
            'under18_mention',
            'pediatric_population',
            'pregnancy_mention',
            'excluded_malformation',
            'included_study_design',
            'excluded_study_design',
            'occlusion_reporting',
            'supported_intervention',
            'exclusion_keywords',
            'patients_under_30',
            'case_study_few_patients',
            'combined_population_exclusion'
        ])