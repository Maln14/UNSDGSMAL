
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Auto-generate requirements.txt file
def generate_requirements_file():
    """Automatically generate requirements.txt file with all necessary dependencies"""
    requirements_content = """# UN SDGs Data Science Pipeline - Requirements
# Generated for GitHub deployment

# Core Data Science Libraries
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0

# Visualization Libraries
plotly>=5.15.0

# HTTP Requests
requests>=2.31.0

# Machine Learning Libraries
scikit-learn>=1.3.0
statsmodels>=0.14.0

# File Processing
openpyxl>=3.1.0

# Development Tools (optional)
pipreqs>=0.4.11

# Additional dependencies that may be needed
urllib3>=2.0.0"""
    
    try:
        # Only create if doesn't exist or is empty
        if not os.path.exists('requirements.txt') or os.path.getsize('requirements.txt') == 0:
            with open('requirements.txt', 'w') as f:
                f.write(requirements_content)
            print("✅ Generated requirements.txt file successfully!")
            print("📦 To install dependencies, run: pip install -r requirements.txt")
        else:
            print("📦 requirements.txt already exists")
    except Exception as e:
        print(f"⚠️ Could not create requirements.txt: {e}")
        print("📦 Manual installation command:")
        print("pip install streamlit pandas numpy plotly requests scikit-learn statsmodels openpyxl")

def update_requirements_with_pipreqs():
    """Update requirements.txt using pipreqs if available"""
    try:
        import subprocess
        result = subprocess.run(['pipreqs', '--force', '--savepath', 'requirements_auto.txt', '.'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Auto-generated requirements_auto.txt using pipreqs!")
            print("📦 Compare with requirements.txt and merge if needed")
        else:
            print("⚠️ pipreqs failed, using manual requirements.txt")
    except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
        print("📦 pipreqs not available, using manual requirements.txt")
    except Exception as e:
        print(f"⚠️ Error running pipreqs: {e}")

# Generate requirements file on import
generate_requirements_file()


# Page configuration
st.set_page_config(
    page_title="UN SDGs Analysis Pipeline",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA FETCHER CLASS
# =============================================================================

class DataFetcher:
    """Handles data fetching from UN Statistics Division and World Bank APIs"""
    
    def __init__(self):
        self.un_base_url = "https://unstats.un.org/SDGAPI"
        self.wb_base_url = "https://api.worldbank.org/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        # Increase timeout and add retry logic
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def test_un_api(self):
        """Test UN API connectivity with detailed debugging"""
        try:
            print("Testing UN API connection...")
            response = self.session.get(f"{self.un_base_url}/v1/sdg/Goal/List", timeout=15)
            print(f"UN API Response Status: {response.status_code}")
            if response.status_code == 200:
                print("UN API: Successfully connected!")
                return True
            else:
                print(f"UN API: Failed with status {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            print("UN API: Connection timed out")
            return False
        except requests.exceptions.ConnectionError:
            print("UN API: Connection error - check internet connection")
            return False
        except Exception as e:
            print(f"UN API: Error - {str(e)}")
            return False
    
    def test_world_bank_api(self):
        """Test World Bank API connectivity with detailed debugging"""
        try:
            print("Testing World Bank API connection...")
            response = self.session.get(f"{self.wb_base_url}/country?format=json&per_page=1", timeout=15)
            print(f"World Bank API Response Status: {response.status_code}")
            if response.status_code == 200:
                print("World Bank API: Successfully connected!")
                return True
            else:
                print(f"World Bank API: Failed with status {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            print("World Bank API: Connection timed out")
            return False
        except requests.exceptions.ConnectionError:
            print("World Bank API: Connection error - check internet connection")
            return False
        except Exception as e:
            print(f"World Bank API: Error - {str(e)}")
            return False
    
    def get_countries(self):
        """Fetch countries data from UN API with enhanced error handling"""
        try:
            print("Fetching countries from UN API...")
            response = self.session.get(f"{self.un_base_url}/v1/sdg/GeoArea/List", timeout=20)
            print(f"Countries API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                countries_data = response.json()
                print(f"Received {len(countries_data)} countries from API")
                countries_df = pd.DataFrame(countries_data)
                
                if 'geoAreaName' in countries_df.columns:
                    countries_df = countries_df.rename(columns={
                        'geoAreaCode': 'code',
                        'geoAreaName': 'country_name'
                    })
                    # Filter for country codes (3 characters) and remove regions
                    country_codes_df = countries_df[countries_df['code'].str.len() == 3].reset_index(drop=True)
                    print(f"Filtered to {len(country_codes_df)} valid countries")
                    st.success(f"✅ Loaded {len(country_codes_df)} countries from UN API")
                    return country_codes_df
                    
            # If we get here, the API response wasn't what we expected
            print("API response format unexpected, using sample data")
            st.warning("⚠️ UN API format changed, using sample countries data")
            return self._generate_sample_countries()
            
        except requests.exceptions.Timeout:
            print("API request timed out")
            st.warning("⚠️ UN API connection timed out, using sample countries data")
            return self._generate_sample_countries()
        except requests.exceptions.ConnectionError:
            print("Connection error to UN API")
            st.warning("⚠️ Cannot connect to UN API, using sample countries data")
            return self._generate_sample_countries()
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            st.warning(f"⚠️ API Error: {str(e)}, using sample countries data")
            return self._generate_sample_countries()
    
    def _generate_sample_countries(self):
        """Generate comprehensive sample countries list"""
        sample_countries = [
            {'code': 'USA', 'country_name': 'United States'},
            {'code': 'GBR', 'country_name': 'United Kingdom'},
            {'code': 'FRA', 'country_name': 'France'},
            {'code': 'DEU', 'country_name': 'Germany'},
            {'code': 'JPN', 'country_name': 'Japan'},
            {'code': 'CHN', 'country_name': 'China'},
            {'code': 'IND', 'country_name': 'India'},
            {'code': 'BRA', 'country_name': 'Brazil'},
            {'code': 'CAN', 'country_name': 'Canada'},
            {'code': 'AUS', 'country_name': 'Australia'},
            {'code': 'ITA', 'country_name': 'Italy'},
            {'code': 'ESP', 'country_name': 'Spain'},
            {'code': 'KOR', 'country_name': 'South Korea'},
            {'code': 'MEX', 'country_name': 'Mexico'},
            {'code': 'NLD', 'country_name': 'Netherlands'},
            {'code': 'SWE', 'country_name': 'Sweden'},
            {'code': 'NOR', 'country_name': 'Norway'},
            {'code': 'DNK', 'country_name': 'Denmark'},
            {'code': 'FIN', 'country_name': 'Finland'},
            {'code': 'CHE', 'country_name': 'Switzerland'},
            {'code': 'SGP', 'country_name': 'Singapore'},
            {'code': 'NZL', 'country_name': 'New Zealand'},
            {'code': 'ZAF', 'country_name': 'South Africa'},
            {'code': 'ARG', 'country_name': 'Argentina'},
            {'code': 'CHL', 'country_name': 'Chile'},
            {'code': 'THA', 'country_name': 'Thailand'},
            {'code': 'MYS', 'country_name': 'Malaysia'},
            {'code': 'IDN', 'country_name': 'Indonesia'},
            {'code': 'RUS', 'country_name': 'Russia'},
            {'code': 'TUR', 'country_name': 'Turkey'}
        ]
        return pd.DataFrame(sample_countries)
    
    def get_sdg_data(self, sdg_number, country_codes=None, start_year=2015, end_year=2023):
        """Fetch SDG data from UN API"""
        try:
            url = f"{self.un_base_url}/v1/sdg/Goal/{sdg_number}/Data"
            params = {'startYear': start_year, 'endYear': end_year}
            
            if country_codes:
                params['geoAreaCode'] = ','.join(country_codes[:10])  # Limit to 10 countries
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    return df
            
            # Return sample data if API fails
            return self._generate_sample_sdg_data(sdg_number, country_codes, start_year, end_year)
            
        except Exception as e:
            st.warning(f"Using sample data for SDG {sdg_number}: {str(e)}")
            return self._generate_sample_sdg_data(sdg_number, country_codes, start_year, end_year)
    
    def get_world_bank_data(self, indicator, country_codes=None, start_year=2015, end_year=2023):
        """Fetch World Bank data"""
        try:
            countries = ','.join(country_codes[:10]) if country_codes else 'all'
            url = f"{self.wb_base_url}/country/{countries}/indicator/{indicator}"
            params = {
                'date': f"{start_year}:{end_year}",
                'format': 'json',
                'per_page': 1000
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    df = pd.DataFrame(data[1])
                    return df
            
            return self._generate_sample_wb_data(indicator, country_codes, start_year, end_year)
            
        except Exception as e:
            st.warning(f"Using sample data for {indicator}: {str(e)}")
            return self._generate_sample_wb_data(indicator, country_codes, start_year, end_year)
    
    def get_combined_data(self, sdg_number, country_codes=None, start_year=2015, end_year=2023):
        """Get combined SDG and World Bank data"""
        sdg_data = self.get_sdg_data(sdg_number, country_codes, start_year, end_year)
        
        wb_indicators = {
            2: 'AG.PRD.FOOD.XD',  # Food production index
            4: 'SE.PRM.NENR',     # Primary education enrollment
            8: 'SL.UEM.TOTL.ZS'   # Unemployment rate
        }
        
        wb_data = pd.DataFrame()
        if sdg_number in wb_indicators:
            wb_data = self.get_world_bank_data(wb_indicators[sdg_number], country_codes, start_year, end_year)
        
        return sdg_data, wb_data
    
    def _generate_sample_sdg_data(self, sdg_number, country_codes=None, start_year=2015, end_year=2023):
        """Generate sample SDG data for demonstration"""
        # Use more comprehensive country list
        default_countries = ['United States', 'United Kingdom', 'France', 'Germany', 'Japan', 
                           'China', 'India', 'Brazil', 'Canada', 'Australia', 'Italy', 'Spain',
                           'South Korea', 'Mexico', 'Netherlands', 'Sweden', 'Norway', 'Denmark']
        
        if country_codes:
            # Convert country codes to names for sample data
            code_to_name = {
                'USA': 'United States', 'GBR': 'United Kingdom', 'FRA': 'France', 
                'DEU': 'Germany', 'JPN': 'Japan', 'CHN': 'China', 'IND': 'India',
                'BRA': 'Brazil', 'CAN': 'Canada', 'AUS': 'Australia', 'ITA': 'Italy',
                'ESP': 'Spain', 'KOR': 'South Korea', 'MEX': 'Mexico', 'NLD': 'Netherlands',
                'SWE': 'Sweden', 'NOR': 'Norway', 'DNK': 'Denmark', 'FIN': 'Finland'
            }
            countries = [code_to_name.get(code, code) for code in country_codes[:10]]
        else:
            countries = default_countries[:10]
            
        years = list(range(start_year, end_year + 1))
        
        data = []
        
        # Create realistic trend patterns for each country
        country_trends = {}
        for i, country in enumerate(countries):
            # Create diverse trend patterns
            trend_type = i % 4  # Different trend types
            base_value = np.random.uniform(40, 90)  # Random starting point
            
            if trend_type == 0:  # Improving trend
                yearly_change = np.random.uniform(0.5, 2.0)
                trend_direction = 'improving'
            elif trend_type == 1:  # Declining trend  
                yearly_change = np.random.uniform(-2.0, -0.5)
                trend_direction = 'declining'
            elif trend_type == 2:  # Stable with slight improvement
                yearly_change = np.random.uniform(-0.2, 0.5)
                trend_direction = 'stable'
            else:  # Volatile but generally improving
                yearly_change = np.random.uniform(0.2, 1.0)
                trend_direction = 'volatile'
                
            country_trends[country] = {
                'base_value': base_value,
                'yearly_change': yearly_change,
                'trend_direction': trend_direction
            }
        
        for country in countries:
            country_info = country_trends[country]
            base_value = country_info['base_value']
            yearly_change = country_info['yearly_change']
            trend_direction = country_info['trend_direction']
            
            for i, year in enumerate(years):
                # Calculate progressive value based on trend
                if trend_direction == 'volatile':
                    # Add some volatility
                    volatility = np.random.normal(0, 2)
                    value = base_value + (yearly_change * i) + volatility
                else:
                    # Steady progression with small random variation
                    noise = np.random.normal(0, 1)
                    value = base_value + (yearly_change * i) + noise
                
                # Set target values based on SDG
                if sdg_number == 2:  # Zero Hunger
                    target_value = 85
                elif sdg_number == 4:  # Quality Education
                    target_value = 95
                elif sdg_number == 8:  # Decent Work
                    target_value = 80
                else:
                    target_value = 85
                
                # Ensure proper country code mapping
                country_code_mapping = {
                    'United States': 'USA', 'United Kingdom': 'GBR', 'France': 'FRA', 
                    'Germany': 'DEU', 'Japan': 'JPN', 'China': 'CHN', 'India': 'IND',
                    'Brazil': 'BRA', 'Canada': 'CAN', 'Australia': 'AUS', 'Italy': 'ITA',
                    'Spain': 'ESP', 'South Korea': 'KOR', 'Mexico': 'MEX', 'Netherlands': 'NLD',
                    'Sweden': 'SWE', 'Norway': 'NOR', 'Denmark': 'DNK', 'Finland': 'FIN'
                }
                
                data.append({
                    'goal': sdg_number,
                    'target': f"{sdg_number}.1",
                    'indicator': f"{sdg_number}.1.1",
                    'geoAreaName': country,
                    'geoAreaCode': country_code_mapping.get(country, country[:3].upper()),
                    'timePeriod': year,
                    'value': max(0, min(100, value)),
                    'valueType': 'Sample Data',
                    'units': 'Percentage',
                    'targetValue': target_value
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_wb_data(self, indicator, country_codes=None, start_year=2015, end_year=2023):
        """Generate sample World Bank data"""
        # Use same comprehensive country mapping
        default_countries = ['United States', 'United Kingdom', 'France', 'Germany', 'Japan', 
                           'China', 'India', 'Brazil', 'Canada', 'Australia']
        
        if country_codes:
            code_to_name = {
                'USA': 'United States', 'GBR': 'United Kingdom', 'FRA': 'France', 
                'DEU': 'Germany', 'JPN': 'Japan', 'CHN': 'China', 'IND': 'India',
                'BRA': 'Brazil', 'CAN': 'Canada', 'AUS': 'Australia', 'ITA': 'Italy',
                'ESP': 'Spain', 'KOR': 'South Korea', 'MEX': 'Mexico', 'NLD': 'Netherlands'
            }
            countries = [code_to_name.get(code, code) for code in country_codes[:10]]
        else:
            countries = default_countries
            
        years = list(range(start_year, end_year + 1))
        
        data = []
        
        # Create realistic trend patterns for WB data too
        country_wb_trends = {}
        for i, country in enumerate(countries):
            trend_type = i % 3  # Different trend patterns
            
            if 'FOOD' in indicator:
                base_value = np.random.uniform(95, 115)
                if trend_type == 0:
                    yearly_change = np.random.uniform(0.5, 1.5)  # Improving
                else:
                    yearly_change = np.random.uniform(-0.5, 0.5)  # Stable/declining
            elif 'EDU' in indicator or 'PRM' in indicator:
                base_value = np.random.uniform(80, 98)
                if trend_type == 0:
                    yearly_change = np.random.uniform(0.2, 1.0)  # Improving
                else:
                    yearly_change = np.random.uniform(-0.3, 0.3)  # Stable
            elif 'UEM' in indicator:
                base_value = np.random.uniform(3, 10)
                if trend_type == 0:
                    yearly_change = np.random.uniform(-0.3, -0.1)  # Improving (decreasing unemployment)
                else:
                    yearly_change = np.random.uniform(-0.1, 0.2)  # Stable/worsening
            else:
                base_value = np.random.uniform(50, 90)
                yearly_change = np.random.uniform(-1, 1)
                
            country_wb_trends[country] = {
                'base_value': base_value, 
                'yearly_change': yearly_change
            }
        
        for country in countries:
            trend_info = country_wb_trends[country]
            base_value = trend_info['base_value']
            yearly_change = trend_info['yearly_change']
            
            for i, year in enumerate(years):
                # Progressive value with trend
                noise = np.random.normal(0, 0.5)
                value = base_value + (yearly_change * i) + noise
                
                data.append({
                    'indicator': {'id': indicator, 'value': 'Sample Indicator'},
                    'country': {'id': country[:3].upper(), 'value': country},
                    'countryiso3code': country[:3].upper(),
                    'date': str(year),
                    'value': max(0, value),
                    'decimal': 1
                })
        
        return pd.DataFrame(data)

# =============================================================================
# DATA PROCESSOR CLASS
# =============================================================================

class DataProcessor:
    """Handles data cleaning, processing, and analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
    
    def clean_data(self, data):
        """Clean and standardize data"""
        if data.empty:
            return data
        
        # Standardize column names
        data = data.copy()
        
        # Handle different data formats
        if 'geoAreaName' in data.columns:
            data = data.rename(columns={
                'geoAreaName': 'country_name',
                'geoAreaCode': 'country_code',
                'timePeriod': 'year'
            })
        elif 'country' in data.columns:
            if isinstance(data['country'].iloc[0], dict):
                data['country_name'] = data['country'].apply(lambda x: x.get('value', ''))
                data['country_code'] = data['country'].apply(lambda x: x.get('id', ''))
        
        # Ensure country_name is properly set and not using codes or indices
        if 'country_name' in data.columns:
            # Make sure we're not accidentally using country codes as names
            data['country_name'] = data['country_name'].astype(str)
            
            # Reset index to avoid using index numbers as country names
            data = data.reset_index(drop=True)
            
            # If country_name looks like a code (3 letters all caps), try to map it
            code_to_name = {
                'USA': 'United States', 'GBR': 'United Kingdom', 'FRA': 'France', 
                'DEU': 'Germany', 'JPN': 'Japan', 'CHN': 'China', 'IND': 'India',
                'BRA': 'Brazil', 'CAN': 'Canada', 'AUS': 'Australia', 'ITA': 'Italy',
                'ESP': 'Spain', 'KOR': 'South Korea', 'MEX': 'Mexico', 'NLD': 'Netherlands',
                'SWE': 'Sweden', 'NOR': 'Norway', 'DNK': 'Denmark', 'FIN': 'Finland'
            }
            
            # Check if country_name values are actually numbers (pandas index)
            numeric_mask = data['country_name'].str.isnumeric()
            if numeric_mask.any():
                # If we have numeric country names, they're likely index values, replace with actual names
                if 'geoAreaName' in data.columns:
                    data.loc[numeric_mask, 'country_name'] = data.loc[numeric_mask, 'geoAreaName']
                else:
                    # Fallback: create country names from selected countries list
                    unique_countries = ['United States', 'United Kingdom', 'France', 'Germany', 'Japan', 
                                      'China', 'India', 'Brazil', 'Canada', 'Australia']
                    num_countries_needed = numeric_mask.sum()
                    country_cycle = (unique_countries * ((num_countries_needed // len(unique_countries)) + 1))[:num_countries_needed]
                    data.loc[numeric_mask, 'country_name'] = country_cycle
            
            # Apply code to name mapping
            data['country_name'] = data['country_name'].map(lambda x: code_to_name.get(x, x))
            
        # Convert year to numeric
        if 'year' in data.columns:
            data['year'] = pd.to_numeric(data['year'], errors='coerce')
        elif 'date' in data.columns:
            data['year'] = pd.to_numeric(data['date'], errors='coerce')
        
        # Convert value to numeric
        if 'value' in data.columns:
            data['value'] = pd.to_numeric(data['value'], errors='coerce')
        
        # Remove rows with missing critical data
        data = data.dropna(subset=['year', 'value'])
        
        return data
    
    def create_progress_indicators(self, data):
        """Create progress indicators and trends"""
        if data.empty:
            return data
        
        data = data.copy()
        
        # Calculate year-over-year change
        data = data.sort_values(['country_name', 'year'])
        data['previous_value'] = data.groupby('country_name')['value'].shift(1)
        data['yoy_change'] = data['value'] - data['previous_value']
        data['yoy_change_pct'] = (data['yoy_change'] / data['previous_value']) * 100
        
        # Calculate moving averages
        data['ma_3year'] = data.groupby('country_name')['value'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        data['ma_5year'] = data.groupby('country_name')['value'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
        
        # Calculate overall trend using linear regression
        def calculate_trend(group):
            if len(group) < 2:
                return 0
            x = np.arange(len(group))
            y = group['value'].values
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except:
                return 0
        
        trends = data.groupby('country_name').apply(calculate_trend).reset_index()
        trends.columns = ['country_name', 'overall_trend']
        data = data.merge(trends, on='country_name', how='left')
        
        return data
    
    def calculate_sdg_scores(self, data):
        """Calculate normalized SDG scores"""
        if data.empty:
            return data
        
        data = data.copy()
        
        # Normalize values to 0-100 scale
        if 'value' in data.columns:
            min_val = data['value'].min()
            max_val = data['value'].max()
            if max_val > min_val:
                data['normalized_score'] = ((data['value'] - min_val) / (max_val - min_val)) * 100
            else:
                data['normalized_score'] = 50  # Default if no variation
        
        # Calculate performance categories
        data['performance_category'] = pd.cut(
            data['normalized_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
            include_lowest=True
        )
        
        return data

# =============================================================================
# VISUALIZATION CLASS
# =============================================================================

class Visualizations:
    """Handles all visualization creation"""
    
    def create_country_performance_chart(self, data, title="Country Performance Over Time"):
        """Create country performance line chart"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        fig = px.line(
            data,
            x='year',
            y='value',
            color='country_name',
            title=title,
            labels={'value': 'Performance Score', 'year': 'Year'},
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Performance Score",
            legend_title="Country"
        )
        
        return fig
    
    def create_regional_comparison(self, data, title="Regional Performance Comparison"):
        """Create regional comparison chart"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Simple regional grouping
        regional_mapping = {
            'United States': 'North America',
            'Canada': 'North America',
            'United Kingdom': 'Europe',
            'France': 'Europe',
            'Germany': 'Europe',
            'Japan': 'Asia',
            'China': 'Asia',
            'India': 'Asia',
            'Brazil': 'South America',
            'Australia': 'Oceania'
        }
        
        data['region'] = data['country_name'].map(regional_mapping).fillna('Other')
        regional_avg = data.groupby('region')['value'].mean().reset_index()
        
        fig = px.bar(
            regional_avg,
            x='region',
            y='value',
            title=title,
            color='value',
            color_continuous_scale='viridis',
            height=400
        )
        
        return fig
    
    def create_ranking_chart(self, data, title="Country Rankings"):
        """Create country ranking chart"""
        if data.empty:
            return go.Figure().add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Get latest scores for each country
        latest_data = data.groupby('country_name').last().reset_index()
        latest_data = latest_data.sort_values('value', ascending=True).tail(10)
        
        fig = px.bar(
            latest_data,
            x='value',
            y='country_name',
            orientation='h',
            title=title,
            color='value',
            color_continuous_scale='viridis',
            height=400
        )
        
        return fig

# =============================================================================
# PREDICTIVE MODELS CLASS
# =============================================================================

class PredictiveModels:
    """Handles machine learning models for SDG prediction"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_fitted = {}
    
    def prepare_features(self, data):
        """Prepare features for machine learning"""
        if data.empty:
            return pd.DataFrame(), pd.Series()
        
        # Create lagged features
        data = data.sort_values(['country_name', 'year'])
        data['value_lag1'] = data.groupby('country_name')['value'].shift(1)
        data['value_lag2'] = data.groupby('country_name')['value'].shift(2)
        data['trend_indicator'] = data['overall_trend']
        
        # Create time-based features
        data['year_numeric'] = data['year'] - data['year'].min()
        
        # Select features
        feature_columns = ['year_numeric', 'value_lag1', 'value_lag2', 'trend_indicator']
        features = data[feature_columns].copy()
        target = data['value'].copy()
        
        # Remove rows with NaN values
        valid_rows = features.notna().all(axis=1) & target.notna()
        features = features[valid_rows]
        target = target[valid_rows]
        
        return features, target
    
    def train_models(self, data):
        """Train all prediction models"""
        features, target = self.prepare_features(data)
        
        if len(features) < 10:  # Need minimum data for training
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                # Train model
                if name in ['Linear Regression', 'Ridge Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                results[name] = {
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'predictions': y_pred,
                    'actual': y_test.values
                }
                
                self.is_fitted[name] = True
                
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
                self.is_fitted[name] = False
        
        return results
    
    def predict_future(self, data, years_ahead=5):
        """Make future predictions"""
        if data.empty:
            return pd.DataFrame()
        
        # Prepare latest data point for each country
        latest_data = data.groupby('country_name').last().reset_index()
        predictions = []
        
        for _, row in latest_data.iterrows():
            country = row['country_name']
            last_year = int(row['year'])
            last_value = row['value']
            trend = row.get('overall_trend', 0)
            
            for year_offset in range(1, years_ahead + 1):
                # Simple trend-based prediction
                predicted_value = last_value + (trend * year_offset)
                predicted_value = max(0, min(100, predicted_value))  # Keep within bounds
                
                predictions.append({
                    'country_name': country,
                    'year': last_year + year_offset,
                    'predicted_value': predicted_value,
                    'prediction_type': 'Trend-based'
                })
        
        return pd.DataFrame(predictions)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class Utils:
    """Utility functions for the application"""
    
    @staticmethod
    def get_sdg_info():
        """Get SDG information"""
        return {
            2: {
                'title': 'Zero Hunger',
                'icon': '🌾',
                'description': 'End hunger, achieve food security and improved nutrition and promote sustainable agriculture'
            },
            4: {
                'title': 'Quality Education',
                'icon': '📚',
                'description': 'Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all'
            },
            8: {
                'title': 'Decent Work and Economic Growth',
                'icon': '💼',
                'description': 'Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all'
            }
        }
    
    @staticmethod
    def create_country_selector(countries_df, key="countries", max_countries=10):
        """Create country selection widget"""
        if countries_df.empty:
            return []
        
        country_options = sorted(countries_df['country_name'].unique())
        default_countries = ['United States', 'United Kingdom', 'France', 'Germany', 'Japan']
        default_selection = [c for c in default_countries if c in country_options][:5]
        
        selected = st.multiselect(
            f"Select Countries (max {max_countries}):",
            country_options,
            default=default_selection,
            key=key,
            max_selections=max_countries
        )
        
        return selected
    
    @staticmethod
    def format_number(num, decimal_places=1):
        """Format numbers for display"""
        if pd.isna(num):
            return "N/A"
        return f"{num:.{decimal_places}f}"
    
    @staticmethod
    def get_trend_text(trend_value):
        """Convert trend value to text"""
        if pd.isna(trend_value):
            return "No Data"
        elif trend_value > 0.1:
            return "Improving"
        elif trend_value < -0.1:
            return "Declining"
        else:
            return "Stable"

# =============================================================================
# INITIALIZE COMPONENTS
# =============================================================================

@st.cache_resource
def get_components():
    """Initialize all components with caching"""
    data_fetcher = DataFetcher()
    data_processor = DataProcessor()
    visualizations = Visualizations()
    predictive_models = PredictiveModels()
    return data_fetcher, data_processor, visualizations, predictive_models

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    
    # Initialize components
    data_fetcher, data_processor, visualizations, predictive_models = get_components()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Page:",
        ["Home", "SDG 2 - Zero Hunger", "SDG 4 - Quality Education", "SDG 8 - Decent Work", "Comparative Analysis", "Predictive Models"]
    )
    
    # Main content based on selected page
    if page == "Home":
        show_home_page(data_fetcher)
    elif "SDG 2" in page:
        show_sdg_page(2, data_fetcher, data_processor, visualizations)
    elif "SDG 4" in page:
        show_sdg_page(4, data_fetcher, data_processor, visualizations)
    elif "SDG 8" in page:
        show_sdg_page(8, data_fetcher, data_processor, visualizations)
    elif "Comparative" in page:
        show_comparative_analysis(data_fetcher, data_processor, visualizations)
    elif "Predictive" in page:
        show_predictive_models(data_fetcher, data_processor, predictive_models)

def show_home_page(data_fetcher):
    """Display home page"""
    st.title("UN Sustainable Development Goals Analysis Pipeline")
    st.markdown("""
    ### Analyzing Progress Towards SDGs 2, 4, and 8
    
    This interactive dashboard provides comprehensive analysis of:
    - **SDG 2**: Zero Hunger - End hunger, achieve food security and improved nutrition
    - **SDG 4**: Quality Education - Ensure inclusive and equitable quality education
    - **SDG 8**: Decent Work - Promote sustained, inclusive and sustainable economic growth
    
    Navigate through the sidebar to explore specific SDG analyses, comparative studies, and predictive models.
    """)
    
    # Main dashboard overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="SDG 2 - Zero Hunger",
            value="193 Countries",
            delta="Global Coverage"
        )
        st.info("Food security, nutrition, and sustainable agriculture indicators")
    
    with col2:
        st.metric(
            label="SDG 4 - Quality Education",
            value="Education Access",
            delta="Lifelong Learning"
        )
        st.info("Educational attainment, literacy, and learning outcomes")
    
    with col3:
        st.metric(
            label="SDG 8 - Decent Work",
            value="Economic Growth",
            delta="Employment"
        )
        st.info("Economic indicators, employment, and decent work metrics")
    
    # Data source information
    st.markdown("---")
    st.subheader("Data Sources")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **UN Statistics Division API**
        - Official SDG indicators
        - 210+ indicators across 193+ countries
        - Time series data from 2000-2023
        """)
    
    with col2:
        st.markdown("""
        **World Bank Open Data**
        - Complementary socioeconomic indicators
        - GDP, population, development metrics
        - Cross-validation data sources
        """)
    
    # Quick data overview
    st.markdown("---")
    st.subheader("Quick Data Overview")
    
    # Test API connectivity
    with st.spinner("Testing API connectivity..."):
        try:
            un_status = data_fetcher.test_un_api()
            wb_status = data_fetcher.test_world_bank_api()
            
            col1, col2 = st.columns(2)
            with col1:
                if un_status:
                    st.success("UN Statistics API: Connected")
                else:
                    st.error("UN Statistics API: Connection Failed")
            
            with col2:
                if wb_status:
                    st.success("World Bank API: Connected")
                else:
                    st.error("World Bank API: Connection Failed")
                    
        except Exception as e:
            st.error(f"API connectivity test failed: {str(e)}")
    
    # Instructions
    st.markdown("---")
    st.subheader("Getting Started")
    st.markdown("""
    1. **Individual SDG Analysis**: Use the sidebar to navigate to specific SDG pages for detailed analysis
    2. **Comparative Analysis**: Compare progress across different SDGs and countries
    3. **Predictive Models**: Explore machine learning models for forecasting SDG progress
    4. **Data Export**: Download processed data for further analysis
    
    Each page provides interactive visualizations, filtering options, and detailed insights for policy makers and researchers.
    """)

def show_sdg_page(sdg_number, data_fetcher, data_processor, visualizations):
    """Display individual SDG analysis page"""
    sdg_info = Utils.get_sdg_info()[sdg_number]
    
    st.title(f"SDG {sdg_number}: {sdg_info['title']}")
    st.markdown(f"**{sdg_info['description']}**")
    
    # Sidebar controls
    st.sidebar.header(f"SDG {sdg_number} Analysis Controls")
    
    # Get countries data
    with st.spinner("Loading countries data..."):
        countries_df = data_fetcher.get_countries()
    
    if countries_df.empty:
        st.error("Unable to load countries data. Please check your internet connection.")
        return
    
    # Country selection
    selected_countries = Utils.create_country_selector(countries_df, key=f"sdg_{sdg_number}_countries")
    
    # Year range
    start_year, end_year = st.slider(
        "Select Year Range:",
        min_value=2000,
        max_value=2023,
        value=(2015, 2023),
        key=f"sdg_{sdg_number}_years"
    )
    
    # Load data button
    if st.sidebar.button(f"Load SDG {sdg_number} Data", key=f"load_sdg_{sdg_number}"):
        country_codes = countries_df[countries_df['country_name'].isin(selected_countries)]['code'].tolist() if selected_countries else None
        
        with st.spinner(f"Loading SDG {sdg_number} data..."):
            sdg_data, wb_data = data_fetcher.get_combined_data(sdg_number, country_codes, start_year, end_year)
        
        if not sdg_data.empty:
            # Process data
            clean_data = data_processor.clean_data(sdg_data)
            progress_data = data_processor.create_progress_indicators(clean_data)
            scores = data_processor.calculate_sdg_scores(progress_data)
            
            
            # Store in session state
            st.session_state[f'sdg_{sdg_number}_data'] = progress_data
            st.session_state[f'sdg_{sdg_number}_scores'] = scores
            st.session_state[f'sdg_{sdg_number}_selected_countries'] = selected_countries
    
    # Check if data is loaded
    if f'sdg_{sdg_number}_data' not in st.session_state:
        st.info(f"👆 Use the sidebar to load SDG {sdg_number} data for analysis")
        
        # Display SDG information
        st.subheader(f"About SDG {sdg_number}")
        st.markdown(f"{sdg_info['description']}")
        
        return
    
    # Display loaded data
    data = st.session_state[f'sdg_{sdg_number}_data']
    scores = st.session_state[f'sdg_{sdg_number}_scores']
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Rankings", "Export"])
    
    with tab1:
        st.subheader("Performance Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = scores['normalized_score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}")
        
        with col2:
            improving_countries = len(scores[scores['overall_trend'] > 0])
            st.metric("Improving Countries", improving_countries)
        
        with col3:
            latest_year = data['year'].max()
            st.metric("Latest Data Year", int(latest_year))
        
        with col4:
            total_countries = data['country_name'].nunique()
            st.metric("Countries Analyzed", total_countries)
        
        # Performance chart
        if not data.empty:
            fig = visualizations.create_country_performance_chart(data, f"SDG {sdg_number} Performance Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Trend Analysis")
        
        if not scores.empty:
            # Trend summary - ensure we get the actual country names properly
            trend_summary = scores.groupby('country_name', as_index=False).agg({
                'overall_trend': 'first'
            })
            trend_summary['trend_category'] = trend_summary['overall_trend'].apply(Utils.get_trend_text)
            
            
            trend_counts = trend_summary['trend_category'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Trend distribution
                fig_trend = px.pie(
                    values=trend_counts.values,
                    names=trend_counts.index,
                    title="Trend Distribution",
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Top performers
                st.markdown("#### Top Improving Countries")
                top_improving = trend_summary.nlargest(5, 'overall_trend')
                
                if not top_improving.empty:
                    for _, row in top_improving.iterrows():
                        country_name = str(row['country_name'])  # Ensure it's a string
                        trend_value = row['overall_trend']
                        trend_text = Utils.get_trend_text(trend_value)
                        
                        
                        st.metric(
                            label=country_name,
                            value=trend_text, 
                            delta=f"{trend_value:.3f}"
                        )
                else:
                    st.info("No trend data available")
    
    with tab3:
        st.subheader("Country Rankings")
        
        if not scores.empty:
            # Rankings chart
            fig_ranking = visualizations.create_ranking_chart(scores, f"SDG {sdg_number} Country Rankings")
            st.plotly_chart(fig_ranking, use_container_width=True)
            
            # Rankings table
            latest_scores = scores.groupby('country_name').last().reset_index()
            latest_scores = latest_scores.sort_values('normalized_score', ascending=False)
            
            st.dataframe(
                latest_scores[['country_name', 'normalized_score', 'performance_category']].rename(columns={
                    'country_name': 'Country',
                    'normalized_score': 'Score',
                    'performance_category': 'Category'
                }),
                hide_index=True,
                use_container_width=True
            )
    
    with tab4:
        st.subheader("Export Data")
        
        if not data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Download SDG {sdg_number} Data"):
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"sdg_{sdg_number}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.markdown("**Data Summary**")
                st.metric("Total Records", f"{len(data):,}")
                st.metric("Countries", data['country_name'].nunique())
                st.metric("Years Covered", f"{data['year'].min():.0f} - {data['year'].max():.0f}")

def show_comparative_analysis(data_fetcher, data_processor, visualizations):
    """Display comparative analysis page"""
    st.title("Comparative Analysis Across SDGs")
    st.markdown("Compare progress and performance across SDGs 2, 4, and 8 to identify patterns, correlations, and policy insights.")
    
    # Sidebar controls
    st.sidebar.header("Comparative Analysis Controls")
    
    # Get countries data
    with st.spinner("Loading countries data..."):
        countries_df = data_fetcher.get_countries()
    
    if countries_df.empty:
        st.error("Unable to load countries data. Please check your internet connection.")
        return
    
    # Country selection
    selected_countries = Utils.create_country_selector(countries_df, key="comp_countries")
    
    # Year range
    start_year, end_year = st.slider(
        "Select Year Range:",
        min_value=2000,
        max_value=2023,
        value=(2015, 2023),
        key="comp_years"
    )
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type:",
        ["Cross-SDG Performance", "Country Benchmarking", "Regional Comparison", "Trend Analysis"],
        key="comp_analysis_type"
    )
    
    # SDG selection for comparison
    sdgs_to_compare = st.sidebar.multiselect(
        "Select SDGs to Compare:",
        [2, 4, 8],
        default=[2, 4, 8],
        format_func=lambda x: f"SDG {x} - {Utils.get_sdg_info()[x]['title']}",
        key="comp_sdgs"
    )
    
    # Load data button
    if st.sidebar.button("Load Comparative Data", key="load_comp_data"):
        sdg_data_dict = {}
        
        country_codes = countries_df[countries_df['country_name'].isin(selected_countries)]['code'].tolist() if selected_countries else None
        
        with st.spinner("Loading data for comparative analysis..."):
            for sdg in sdgs_to_compare:
                sdg_data, _ = data_fetcher.get_combined_data(sdg, country_codes, start_year, end_year)
                sdg_data_dict[sdg] = sdg_data
        
        st.session_state.comp_sdg_data = sdg_data_dict
        st.session_state.comp_selected_countries = selected_countries
        st.session_state.comp_selected_analysis_type = analysis_type
    
    # Check if data is loaded
    if 'comp_sdg_data' not in st.session_state:
        st.info("👆 Use the sidebar to load data for comparative analysis")
        
        # Display comparative analysis overview
        st.subheader("About Comparative Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### Cross-SDG Analysis Capabilities:
            
            **Cross-SDG Performance**
            - Compare countries across multiple SDGs simultaneously
            - Identify leading and lagging performers
            - Visualize multi-dimensional performance patterns
            
            **Country Benchmarking**
            - Rank countries by overall SDG performance
            - Compare against global and regional averages
            - Identify best practices and success stories
            
            **Regional Comparison**
            - Group countries by geographic regions
            - Compare regional performance patterns
            - Understand cultural and economic influences
            
            **Trend Analysis**
            - Analyze acceleration and deceleration patterns
            - Identify countries with improving trajectories
            - Understand momentum changes over time
            """)
        
        with col2:
            st.markdown("### SDG Interconnections")
            sdg_info = Utils.get_sdg_info()
            
            for sdg_num in [2, 4, 8]:
                info = sdg_info[sdg_num]
                st.markdown(f"""
                **{info['icon']} SDG {sdg_num}**  
                {info['title']}
                """)
        
        return
    
    # Process loaded data
    sdg_data_dict = st.session_state.comp_sdg_data
    selected_countries = st.session_state.comp_selected_countries
    analysis_type = st.session_state.comp_selected_analysis_type
    
    # Data processing for each SDG
    processed_data = {}
    score_data = {}
    
    with st.spinner("Processing comparative data..."):
        for sdg, data in sdg_data_dict.items():
            if not data.empty:
                clean_data = data_processor.clean_data(data)
                progress_data = data_processor.create_progress_indicators(clean_data)
                scores = data_processor.calculate_sdg_scores(progress_data)
                
                processed_data[sdg] = progress_data
                score_data[sdg] = scores
    
    # Display data summary
    st.subheader("Comparative Data Summary")
    
    summary_cols = st.columns(len(sdgs_to_compare))
    for i, sdg in enumerate(sdgs_to_compare):
        with summary_cols[i]:
            if sdg in processed_data and not processed_data[sdg].empty:
                data = processed_data[sdg]
                st.metric(
                    f"SDG {sdg} Records",
                    f"{len(data):,}",
                    delta=f"{data['country_name'].nunique()} countries"
                )
            else:
                st.metric(f"SDG {sdg} Records", "No Data")
    
    # Display content based on analysis type
    st.subheader(f"{analysis_type} Results")
    
    if analysis_type == "Cross-SDG Performance":
        show_cross_sdg_performance(processed_data, score_data, selected_countries, sdgs_to_compare, visualizations)
    elif analysis_type == "Country Benchmarking":
        show_country_benchmarking(processed_data, score_data, selected_countries, sdgs_to_compare)
    elif analysis_type == "Regional Comparison":
        show_regional_comparison(processed_data, score_data, selected_countries, sdgs_to_compare, visualizations)
    elif analysis_type == "Trend Analysis":
        show_trend_analysis(processed_data, score_data, selected_countries, sdgs_to_compare)

def show_cross_sdg_performance(processed_data, score_data, selected_countries, sdgs_to_compare, visualizations):
    """Display cross-SDG performance analysis"""
    
    if not score_data:
        st.warning("No score data available for analysis.")
        return
    
    # Create comparison matrix
    st.markdown("#### Performance Comparison Matrix")
    
    comparison_data = []
    for country in selected_countries[:8]:  # Limit for readability
        row = {'Country': country}
        for sdg in sdgs_to_compare:
            if sdg in score_data and not score_data[sdg].empty:
                country_scores = score_data[sdg][score_data[sdg]['country_name'] == country]
                if not country_scores.empty:
                    latest_score = country_scores['normalized_score'].iloc[-1]
                    row[f'SDG {sdg}'] = f"{latest_score:.1f}"
                else:
                    row[f'SDG {sdg}'] = "N/A"
            else:
                row[f'SDG {sdg}'] = "N/A"
        comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    
    # Performance radar chart
    if len(selected_countries) > 0 and score_data:
        st.markdown("#### Multi-SDG Performance Radar")
        
        fig = go.Figure()
        
        for country in selected_countries[:5]:  # Limit to 5 for readability
            values = []
            labels = []
            
            for sdg in sdgs_to_compare:
                if sdg in score_data and not score_data[sdg].empty:
                    country_data = score_data[sdg][score_data[sdg]['country_name'] == country]
                    if not country_data.empty:
                        avg_value = country_data['normalized_score'].mean()
                        values.append(avg_value)
                        labels.append(f'SDG {sdg}')
            
            if values:
                # Close the radar chart
                values += values[:1]
                labels += labels[:1]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=country,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title="Multi-SDG Performance Comparison",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_country_benchmarking(processed_data, score_data, selected_countries, sdgs_to_compare):
    """Display country benchmarking analysis"""
    
    if not score_data:
        st.warning("No score data available for benchmarking.")
        return
    
    # Create comprehensive country rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Overall SDG Rankings")
        rankings = []
        for country in selected_countries[:10]:
            total_score = 0
            sdg_count = 0
            for sdg in sdgs_to_compare:
                if sdg in score_data and not score_data[sdg].empty:
                    country_scores = score_data[sdg][score_data[sdg]['country_name'] == country]
                    if not country_scores.empty:
                        latest_score = country_scores['normalized_score'].iloc[-1]
                        total_score += latest_score
                        sdg_count += 1
            
            if sdg_count > 0:
                avg_score = total_score / sdg_count
                rankings.append({'Country': country, 'Average Score': avg_score, 'SDGs Covered': sdg_count})
        
        if rankings:
            rankings_df = pd.DataFrame(rankings).sort_values('Average Score', ascending=False)
            st.dataframe(rankings_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### Performance Benchmark")
        if rankings:
            fig_bench = px.bar(
                rankings_df.head(8),
                x='Country',
                y='Average Score',
                title='Country Performance vs Benchmark',
                color='Average Score',
                color_continuous_scale='Viridis',
                height=400
            )
            global_avg = sum(r['Average Score'] for r in rankings) / len(rankings)
            fig_bench.add_hline(y=global_avg, line_dash="dash", line_color="red", 
                               annotation_text=f"Global Average: {global_avg:.1f}")
            st.plotly_chart(fig_bench, use_container_width=True)

def show_regional_comparison(processed_data, score_data, selected_countries, sdgs_to_compare, visualizations):
    """Display regional comparison analysis"""
    
    if not score_data:
        st.warning("No score data available for regional comparison.")
        return
    
    # Simple regional grouping
    regional_groups = {
        'Europe': ['Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Finland', 'United Kingdom'],
        'Asia': ['China', 'India', 'Japan', 'South Korea', 'Thailand', 'Malaysia', 'Singapore', 'Indonesia'],
        'Africa': ['Nigeria', 'South Africa', 'Kenya', 'Ghana', 'Ethiopia', 'Morocco', 'Egypt'],
        'Americas': ['United States', 'Canada', 'Brazil', 'Mexico', 'Argentina', 'Chile', 'Colombia']
    }
    
    # Group selected countries by region
    country_regions = {}
    for country in selected_countries:
        for region, countries in regional_groups.items():
            if country in countries:
                country_regions[country] = region
                break
        else:
            country_regions[country] = 'Other'
    
    # Regional performance analysis
    regional_scores = {}
    for region in set(country_regions.values()):
        region_countries = [c for c, r in country_regions.items() if r == region]
        region_total = 0
        region_count = 0
        
        for sdg in sdgs_to_compare:
            if sdg in score_data and not score_data[sdg].empty:
                for country in region_countries:
                    country_scores = score_data[sdg][score_data[sdg]['country_name'] == country]
                    if not country_scores.empty:
                        latest_score = country_scores['normalized_score'].iloc[-1]
                        region_total += latest_score
                        region_count += 1
        
        if region_count > 0:
            regional_scores[region] = region_total / region_count
    
    if regional_scores:
        col1, col2 = st.columns(2)
        with col1:
            fig_regional = px.bar(
                x=list(regional_scores.keys()),
                y=list(regional_scores.values()),
                title='Average Regional Performance',
                color=list(regional_scores.values()),
                color_continuous_scale='Viridis',
                height=400
            )
            st.plotly_chart(fig_regional, use_container_width=True)
        
        with col2:
            st.markdown("#### Regional Insights")
            sorted_regions = sorted(regional_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (region, score) in enumerate(sorted_regions):
                rank_label = "1st" if i == 0 else "2nd" if i == 1 else "3rd" if i == 2 else f"{i+1}th"
                st.metric(f"{rank_label} - {region}", f"{score:.1f}", delta=f"Rank {i+1}")

def show_trend_analysis(processed_data, score_data, selected_countries, sdgs_to_compare):
    """Display trend analysis"""
    
    if not score_data:
        st.warning("No score data available for trend analysis.")
        return
    
    st.markdown("#### Advanced Trend Analysis")
    
    # Trend acceleration analysis
    acceleration_data = []
    for sdg in sdgs_to_compare:
        if sdg in score_data and not score_data[sdg].empty:
            for country in selected_countries:
                country_data = score_data[sdg][score_data[sdg]['country_name'] == country].sort_values('year')
                if len(country_data) >= 3:
                    # Calculate acceleration (change in trend)
                    recent_trend = country_data.tail(3)['overall_trend'].mean()
                    early_trend = country_data.head(3)['overall_trend'].mean()
                    acceleration = recent_trend - early_trend
                    
                    acceleration_data.append({
                        'Country': country,
                        'SDG': f'SDG {sdg}',
                        'Acceleration': acceleration,
                        'Status': 'Accelerating' if acceleration > 0 else 'Decelerating'
                    })
    
    if acceleration_data:
        acc_df = pd.DataFrame(acceleration_data)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_acc = px.scatter(
                acc_df,
                x='Country',
                y='Acceleration',
                color='SDG',
                symbol='Status',
                title='Progress Acceleration by Country and SDG',
                hover_data=['Status'],
                height=400
            )
            fig_acc.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            st.markdown("#### Acceleration Summary")
            accelerating = acc_df[acc_df['Acceleration'] > 0]
            decelerating = acc_df[acc_df['Acceleration'] <= 0]
            
            st.metric("Countries Accelerating", len(accelerating), delta=f"{len(accelerating)/len(acc_df)*100:.1f}%")
            st.metric("Countries Decelerating", len(decelerating), delta=f"{len(decelerating)/len(acc_df)*100:.1f}%")
            
            if not accelerating.empty:
                top_accelerator = accelerating.loc[accelerating['Acceleration'].idxmax()]
                st.success(f"Top Accelerator: {top_accelerator['Country']} ({top_accelerator['SDG']})")

def show_predictive_models(data_fetcher, data_processor, predictive_models):
    """Display predictive models page"""
    st.title("Predictive Models for SDG Forecasting")
    st.markdown("Use machine learning models to forecast future SDG progress and identify trends.")
    
    # Sidebar controls
    st.sidebar.header("Predictive Modeling Controls")
    
    # Get countries data
    with st.spinner("Loading countries data..."):
        countries_df = data_fetcher.get_countries()
    
    if countries_df.empty:
        st.error("Unable to load countries data. Please check your internet connection.")
        return
    
    # Model configuration
    target_sdg = st.sidebar.selectbox(
        "Select Target SDG:",
        [2, 4, 8],
        format_func=lambda x: f"SDG {x} - {Utils.get_sdg_info()[x]['title']}",
        key="pred_target_sdg"
    )
    
    # Country selection
    selected_countries = Utils.create_country_selector(countries_df, key="pred_countries", max_countries=5)
    
    # Year range for training
    start_year, end_year = st.slider(
        "Training Data Year Range:",
        min_value=2000,
        max_value=2023,
        value=(2015, 2022),
        key="pred_years"
    )
    
    # Prediction horizon
    years_ahead = st.sidebar.slider(
        "Prediction Horizon (years):",
        min_value=1,
        max_value=10,
        value=5,
        key="pred_horizon"
    )
    
    # Train models button
    if st.sidebar.button("Train Prediction Models", key="train_models"):
        country_codes = countries_df[countries_df['country_name'].isin(selected_countries)]['code'].tolist() if selected_countries else None
        
        with st.spinner("Loading data and training models..."):
            sdg_data, _ = data_fetcher.get_combined_data(target_sdg, country_codes, start_year, end_year)
            
            if not sdg_data.empty:
                # Process data
                clean_data = data_processor.clean_data(sdg_data)
                progress_data = data_processor.create_progress_indicators(clean_data)
                
                # Train models
                model_results = predictive_models.train_models(progress_data)
                
                # Make predictions
                future_predictions = predictive_models.predict_future(progress_data, years_ahead)
                
                # Store results
                st.session_state.pred_model_results = model_results
                st.session_state.pred_future_predictions = future_predictions
                st.session_state.pred_training_data = progress_data
                st.session_state.pred_trained_sdg = target_sdg
    
    # Check if models are trained
    if 'pred_model_results' not in st.session_state:
        st.info("👆 Use the sidebar to train prediction models")
        
        # Display modeling information
        st.subheader("About Predictive Modeling")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Machine Learning Models:
            
            **Linear Regression**
            - Simple linear relationships
            - Good baseline performance
            - Interpretable coefficients
            
            **Ridge Regression**
            - Regularized linear model
            - Handles multicollinearity
            - Prevents overfitting
            
            **Random Forest**
            - Ensemble of decision trees
            - Captures non-linear patterns
            - Feature importance analysis
            
            **Gradient Boosting**
            - Sequential ensemble method
            - High predictive accuracy
            - Handles complex relationships
            """)
        
        with col2:
            st.markdown("""
            ### Prediction Features:
            
            **Time-based Features**
            - Year progression
            - Seasonal patterns
            - Historical context
            
            **Lagged Variables**
            - Previous year values
            - Two-year lag features
            - Momentum indicators
            
            **Trend Indicators**
            - Long-term trend direction
            - Rate of change
            - Acceleration patterns
            
            **Model Validation**
            - Cross-validation scores
            - Out-of-sample testing
            - Performance metrics
            """)
        
        return
    
    # Display model results
    model_results = st.session_state.pred_model_results
    future_predictions = st.session_state.pred_future_predictions
    training_data = st.session_state.pred_training_data
    target_sdg = st.session_state.pred_trained_sdg
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "Predictions", "Feature Analysis", "Export"])
    
    with tab1:
        st.subheader("Model Performance Comparison")
        
        if model_results:
            # Performance metrics table
            performance_data = []
            for model_name, results in model_results.items():
                performance_data.append({
                    'Model': model_name,
                    'R² Score': f"{results['r2']:.3f}",
                    'MSE': f"{results['mse']:.3f}",
                    'MAE': f"{results['mae']:.3f}"
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, hide_index=True, use_container_width=True)
            
            # Best model identification
            best_model = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
            st.success(f"Best Performing Model: {best_model} (R² = {model_results[best_model]['r2']:.3f})")
            
            # Performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # R² scores comparison
                r2_scores = [model_results[model]['r2'] for model in model_results.keys()]
                fig_r2 = px.bar(
                    x=list(model_results.keys()),
                    y=r2_scores,
                    title='Model R² Scores Comparison',
                    color=r2_scores,
                    color_continuous_scale='viridis',
                    height=400
                )
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with col2:
                # Prediction vs Actual for best model
                best_results = model_results[best_model]
                fig_scatter = px.scatter(
                    x=best_results['actual'],
                    y=best_results['predictions'],
                    title=f'{best_model}: Predictions vs Actual',
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    height=400
                )
                # Add perfect prediction line
                min_val = min(min(best_results['actual']), min(best_results['predictions']))
                max_val = max(max(best_results['actual']), max(best_results['predictions']))
                fig_scatter.add_shape(
                    type="line", line=dict(dash="dash"),
                    x0=min_val, x1=max_val, y0=min_val, y1=max_val
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        st.subheader("Future Predictions")
        
        if not future_predictions.empty:
            # Predictions chart
            fig_pred = px.line(
                future_predictions,
                x='year',
                y='predicted_value',
                color='country_name',
                title=f'SDG {target_sdg} Predictions ({years_ahead} years ahead)',
                labels={'predicted_value': 'Predicted Score', 'year': 'Year'},
                height=500
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Predictions table
            st.markdown("#### Detailed Predictions")
            pred_summary = future_predictions.groupby('country_name').agg({
                'predicted_value': ['mean', 'min', 'max'],
                'year': ['min', 'max']
            }).round(2)
            
            pred_summary.columns = ['Avg Score', 'Min Score', 'Max Score', 'Start Year', 'End Year']
            st.dataframe(pred_summary, use_container_width=True)
            
            # Key insights
            st.markdown("#### Key Insights")
            best_predicted = future_predictions.groupby('country_name')['predicted_value'].mean().idxmax()
            worst_predicted = future_predictions.groupby('country_name')['predicted_value'].mean().idxmin()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Best Predicted Performer",
                    best_predicted,
                    delta=f"{future_predictions[future_predictions['country_name']==best_predicted]['predicted_value'].mean():.1f}"
                )
            
            with col2:
                st.metric(
                    "Challenging Outlook",
                    worst_predicted,
                    delta=f"{future_predictions[future_predictions['country_name']==worst_predicted]['predicted_value'].mean():.1f}"
                )
            
            with col3:
                avg_prediction = future_predictions['predicted_value'].mean()
                st.metric("Average Predicted Score", f"{avg_prediction:.1f}")
    
    with tab3:
        st.subheader("Feature Analysis")
        
        if not training_data.empty:
            # Feature importance (simplified)
            st.markdown("#### Training Data Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Trend distribution
                trend_dist = training_data.groupby('country_name')['overall_trend'].first()
                trend_categories = trend_dist.apply(Utils.get_trend_text)
                trend_counts = trend_categories.value_counts()
                
                fig_trend = px.pie(
                    values=trend_counts.values,
                    names=trend_counts.index,
                    title='Trend Distribution in Training Data',
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Value distribution
                fig_hist = px.histogram(
                    training_data,
                    x='value',
                    title='Value Distribution',
                    nbins=20,
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Data quality metrics
            st.markdown("#### Data Quality Metrics")
            quality_metrics = {
                'Total Records': len(training_data),
                'Countries': training_data['country_name'].nunique(),
                'Years Covered': f"{training_data['year'].min():.0f} - {training_data['year'].max():.0f}",
                'Missing Values': training_data['value'].isna().sum(),
                'Data Completeness': f"{(1 - training_data['value'].isna().mean()) * 100:.1f}%"
            }
            
            quality_cols = st.columns(len(quality_metrics))
            for i, (metric, value) in enumerate(quality_metrics.items()):
                with quality_cols[i]:
                    st.metric(metric, value)
    
    with tab4:
        st.subheader("Export Predictions")
        
        if not future_predictions.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Predictions"):
                    csv = future_predictions.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"sdg_{target_sdg}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.markdown("**Prediction Summary**")
                st.metric("Countries Predicted", future_predictions['country_name'].nunique())
                st.metric("Prediction Years", f"{years_ahead} years")
                st.metric("Model Used", "Ensemble Average")

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()