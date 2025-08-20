# HRV Study Analysis

## Overview

This repository contains the analysis code for the HRV (Heat Recovery Ventilation) study, examining the effects of positive pressure ventilation units on indoor environmental conditions in 30 houses monitored over two winter seasons (2008 and 2009) in West Auckland, New Zealand.

## Study Design

- **30 houses total**: 20 active homes (with ventilation units) and 10 control homes
- **Monitoring periods**: 
  - 2008: July 28 - October 24 (winter/spring)
  - 2009: June 15 - September 11 (full winter)
- **Measurements**: Temperature, relative humidity, and pollutants in living rooms, bedrooms, and roof cavities
- **Control houses**: 5, 7, 12, 16, 17, 21, 23, 25, 28, 30
- **Active houses**: All others (1-30 excluding control houses)
  - Week 1 of 2008: No ventilation units (baseline)
  - All other weeks: Ventilation units installed

## Repository Structure

```
hrv-study-analysis/
├── src/                    # Core analysis modules
│   ├── utils.py           # Data loading and processing utilities
│   ├── config.py          # Configuration settings
│   └── statistical_tests.py # Statistical testing functions
├── scripts/               # Analysis scripts for research questions
│   ├── question_01.py    # Outside temperature comparison
│   ├── question_02.py    # Full day temperature comparison
│   ├── question_03.py    # Part of day temperature comparison
│   ├── question_04.py    # Temperature exposure analysis
│   ├── question_05.py    # Roof cavity 18°C analysis
│   ├── question_06.py    # Outside conditions for 18°C
│   ├── question_07.py    # Full day RH comparison
│   ├── question_08.py    # Part of day RH comparison
│   ├── question_09.py    # Comfort zone analysis
│   └── question_10.py    # Roof cavity air quality
├── notebooks/            # Jupyter notebooks for exploration
├── results/             # Output directory
│   ├── figures/        # Generated plots
│   ├── tables/         # Generated tables
│   └── reports/        # Summary reports
└── tests/              # Unit tests
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd hrv-study-analysis
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n hrv-analysis python=3.9
conda activate hrv-analysis
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Data Path

The data path can be configured in three ways (in order of precedence):

#### Option 1: Environment Variable (Recommended)

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and set your data path:

```bash
HRV_DATA_PATH=/path/to/your/data
```

#### Option 2: System Environment Variable

Set the environment variable in your shell:

```bash
# Linux/Mac
export HRV_DATA_PATH=/path/to/your/data

# Windows Command Prompt
set HRV_DATA_PATH=C:\path\to\your\data

# Windows PowerShell
$env:HRV_DATA_PATH="C:\path\to\your\data"
```

#### Option 3: Programmatically

Set the path in your Python code:

```python
from src.config import set_data_path
set_data_path('/path/to/your/data')
```

#### Option 4: Default Location

If no path is configured, the system will look for data in the `data/` subdirectory of the project.

### 5. Set Up Data Directory Structure

Your data directory should have the following structure:

```bash
your_data_path/
└── house_data/
    └── preprocessed/
        ├── 18_01.csv  # Week 1, Year 2008, House 01
        ├── 28_01.csv  # Week 2, Year 2008, House 01
        ├── 19_01.csv  # Week 1, Year 2009, House 01
        ├── 29_01.csv  # Week 2, Year 2009, House 01
        └── ...        # Files for all 30 houses
```

### 6. Verify Installation and Configuration

```bash
# Check configuration
python -c "from src.config import print_config; print_config()"

# Test data loading
python -c "from src.utils import load_house_data; df = load_house_data(1, 8, 1); print(f'Loaded {len(df)} records')"
```

## Key Methodological Notes

### Day/Night Definition

This analysis uses **solar irradiance** to define daytime and nighttime periods, rather than fixed clock times:

- **Daytime**: Solar irradiance ≥ 0.005 W/m²
- **Nighttime**: Solar irradiance < 0.005 W/m²

This provides a more accurate representation of actual daylight conditions than fixed time periods (e.g., 9am-5pm), especially important for:
- Seasonal variations in daylight hours
- Weather-dependent solar availability
- Roof cavity heating analysis

The system will automatically:
1. Use solar irradiance data when available (column `ext__SR`)
2. Fall back to fixed times (9am-5pm) if solar data is missing
3. Log statistics about actual daylight hours for each analysis

### Data Format

Each CSV file should contain:
- `timestamps`: DateTime column in format 'yyyy-mm-dd hh:mm:ss'
- Temperature columns: `T-LWY`, `T-BWY`, `T-RWY` (Living, Bedroom, Roof)
- Relative humidity columns: `RH-LWY`, `RH-BWY`, `RH-RWY`
- External data: `ext__T`, `ext__SR`, `ext__RH`, `ext__WD`, `ext__WS`

Where:
- W = Week number (1 or 2)
- Y = Year (8 for 2008, 9 for 2009)

## Usage

### Using the Utilities

```python
from src.utils import load_house_data, get_group_statistics
from src.statistical_tests import independent_t_test

# Load data for House 1, Week 1, Year 2008
df = load_house_data(week=1, year=8, house_num=1)

# Get statistics for active homes
stats = get_group_statistics(week=1, year=8, variable='T', 
                            location='L', group='active')

# Perform t-test
result = independent_t_test(group1_data, group2_data)
```

## Research Questions

1. **Q1**: Was outdoor temperature different between active and control home locations?
2. **Q2**: Were active homes warmer than control homes for the whole day?
3. **Q3**: Were active homes warmer than control homes for part of the day?
4. **Q4**: Difference in exposure to extreme temperatures and WHO recommendations?
5. **Q5**: Is it possible to reach 18°C in roof cavity during winter?
6. **Q6**: What outdoor conditions are needed for 18°C in roof cavity?
7. **Q7**: Was relative humidity lower in active homes for the whole day?
8. **Q8**: Was relative humidity lower in active homes for part of the day?
9. **Q9**: Were active home occupants in comfort zone longer?
10. **Q10**: How often was roof cavity air warmer and drier than living areas?

## Statistical Methods

- **Independent samples t-tests**: Compare active vs control homes
- **Paired t-tests**: Compare before/after ventilation installation
- **ANOVA**: Compare multiple groups or time periods
- **Confidence intervals**: 95% CI for all estimates
- **Effect sizes**: Cohen's d for practical significance
- **Time series analysis**: For temporal patterns
- **Non-parametric tests**: When normality assumptions are violated

## Output

Results are saved in the `results/` directory:
- **Figures**: PNG and PDF formats for all plots
- **Tables**: CSV and Excel formats for statistical results
- **Reports**: HTML and PDF summary reports



For questions about the analysis or data, please contact the research team.

## Citation

If you use this code or data, please cite:
```
[Citation information to be added]
```
