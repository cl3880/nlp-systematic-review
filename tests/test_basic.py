"""
Basic tests for core functionality.
These tests verify imports work and basic operations function correctly.
"""

import pytest
import pandas as pd
import numpy as np


class TestBasicFunctionality:
    """Basic tests for core imports and simple operations."""
    
    def test_imports_work(self):
        """Test that core modules can be imported."""
        from src.utils.hard_filters import extract_publication_year
        from src.models.model_factory import create_model
        
        # If we get here, imports worked
        assert True
    
    def test_extract_publication_year(self):
        """Test basic year extraction functionality."""
        from src.utils.hard_filters import extract_publication_year
        
        # Test valid cases
        assert extract_publication_year("2020-01-15") == 2020
        assert extract_publication_year("2019") == 2019
        
        # Test invalid cases
        assert extract_publication_year(None) is None
        assert extract_publication_year("no year here") is None
    
    def test_pandas_operations(self):
        """Test basic pandas operations work."""
        df = pd.DataFrame({
            'title': ['Test 1', 'Test 2'],
            'year': [2020, 2021]
        })
        
        assert len(df) == 2
        assert df['year'].sum() == 4041
    
    def test_numpy_operations(self):
        """Test basic numpy operations work."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.mean() == 3.0
    
    def test_model_creation_minimal(self):
        """Test minimal model creation without fitting."""
        from src.models.model_factory import create_model
        
        # Test basic model creation
        model = create_model(
            model_type='logreg',
            min_df=1,
            max_features=10
        )
        
        # Check basic structure
        assert hasattr(model, 'named_steps')
        assert 'tfidf' in model.named_steps
        assert 'clf' in model.named_steps 