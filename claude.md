# Claude Code Development Guidelines

> **Audience**: Claude Code AI agent working on data analysis projects

This repository contains independent data analysis projects showcasing business intelligence through statistical analysis and machine learning. Each project is self-contained with its own data, analysis notebooks, and business deliverables.

## Required Development Workflow

**CRITICAL**: Always follow this sequence when working on any project:
```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Run code formatting and linting
black .
flake8 .

# 3. Verify notebooks execute without errors
jupyter nbconvert --to notebook --execute notebooks/*.ipynb

# 4. Run any unit tests (if present)
pytest tests/ -v
```

**All steps must pass before considering work complete.**

## Repository Structure
```
data-analytics-portfolio/
│
├── project-name/
│   ├── data/
│   │   ├── raw/                    # Original dataset or link
│   │   ├── processed/              # Cleaned data
│   │   └── data_dictionary.md      # Variable definitions
│   │
│   ├── case_brief.md               # Business problem and objectives
│   │
│   ├── notebooks/
│   │   ├── 01_exploratory_analysis.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   ├── 03_modeling.ipynb
│   │   └── 04_evaluation.ipynb
│   │
│   ├── src/                        # Python helper functions (optional)
│   │
│   ├── results/
│   │   ├── figures/                # Visualizations
│   │   └── models/                 # Saved models
│   │
│   ├── business_memo.pdf           # Executive summary
│   │
│   └── README.md                   # Project overview
│
├── requirements.txt                # Project dependencies
└── README.md                       # Portfolio overview
```

## Core Principles

### 1. **Project Independence**
- Each project folder is completely self-contained
- **NEVER create shared utilities across projects** - code duplication is acceptable and preferred
- Projects should not depend on each other's code, data, or results
- If similar functionality is needed, copy and adapt rather than import

**Rationale**: This ensures each analysis can be understood, run, and maintained independently without worrying about breaking dependencies.

### 2. **Reproducibility**
- All analysis must be fully reproducible from raw data
- Document data sources with direct download links when possible
- Pin package versions in requirements.txt
- Include random seeds for stochastic operations
- Clear execution order in numbered notebooks

### 3. **Clean Code Standards**
- Follow PEP 8 style guidelines
- Use meaningful variable names (avoid single letters except in loops/lambda)
- Add docstrings to all functions
- Type hints for function signatures (Python 3.10+)
- Keep functions focused and under 50 lines when possible

## Development Rules

### Code Quality

**Python Standards:**
- Python ≥ 3.8 with type hints where appropriate
- Use `black` for formatting (line length: 88)
- Use `flake8` for linting
- Prefer explicit over implicit (clarity > brevity)

**Notebook Standards:**
- Clear markdown explanations before each code block
- One logical step per cell
- Show intermediate results with print statements or visualizations
- Include cell execution numbers in final version
- Restart kernel and run all cells before saving

**Anti-Patterns to Avoid:**
```python
# ❌ Don't do this - unclear variable names
df2 = df1[df1['x'] > 5]
m = LinearRegression()

# ✅ Do this instead - descriptive names
filtered_customers = raw_customers[raw_customers['age'] > 5]
revenue_model = LinearRegression()

# ❌ Don't do this - magic numbers
threshold = 0.73

# ✅ Do this instead - named constants with context
CHURN_PROBABILITY_THRESHOLD = 0.73  # Based on ROC curve analysis

# ❌ Don't do this - no explanation
model.fit(X_train, y_train)

# ✅ Do this instead - explain what's happening
# Train logistic regression model on customer features
# to predict likelihood of churn within 90 days
model.fit(X_train, y_train)
```

### File Organization

**Notebook Naming:**
- Use numbered prefixes: `01_`, `02_`, `03_`, `04_`
- Descriptive names: `exploratory_analysis`, not `eda`
- Lowercase with underscores

**Data Files:**
- Keep raw data immutable in `data/raw/`
- Save processed data to `data/processed/`
- Document transformations in notebooks
- Use appropriate formats: CSV for tables, PNG for images, pickle for models

**Code Structure:**
```python
# Standard imports first
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Then project-specific imports
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration and constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
FIGURE_DPI = 300

# Set global options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
np.random.seed(RANDOM_SEED)
```

### Documentation Requirements

**case_brief.md Structure:**
```markdown
# Project Title

## Business Context
[1-2 paragraphs on the business situation]

## Objectives
[Clear list of what needs to be achieved]

## Key Questions
1. [Specific question 1]
2. [Specific question 2]
3. [Specific question 3]

## Success Metrics
[How will we measure if the analysis was successful?]

## Data Source
[Link and description of dataset]

## Deliverables
- Exploratory analysis insights
- Predictive model (if applicable)
- Business recommendations
- Executive memo
```

**Notebook Documentation:**
- Start with markdown cell explaining notebook purpose
- Section headers (markdown) before major analysis steps
- Inline comments for complex logic only
- Markdown explanations for insights and findings
- Summary cell at the end with key takeaways

**Business Memo Requirements:**
- Maximum 5 pages
- Executive summary (maximum 8-10 sentences)
- Key findings (bullet points)
- Recommendations with rationale
- Implementation considerations
- Visual: 3-4 key charts only

## Critical Patterns

### Data Loading
```python
# Always include error handling and path validation
from pathlib import Path

def load_data(filename: str) -> pd.DataFrame:
    """
    Load dataset from raw data directory.
    
    Args:
        filename: Name of the CSV file to load
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    data_path = Path(__file__).parent / "data" / "raw" / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    return df
```

### Model Training
```python
# Always split data with fixed random seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,  # Fixed for reproducibility
    stratify=y  # Maintain class distribution
)

# Document model parameters clearly
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Train with clear logging
print("Training Random Forest model...")
model.fit(X_train, y_train)
print(f"Training complete. Model score: {model.score(X_test, y_test):.3f}")
```

### Visualization Standards
```python
# Use consistent styling and informative labels
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(x, y, alpha=0.6, s=50)
ax.set_xlabel('Customer Lifetime Value ($)', fontsize=12)
ax.set_ylabel('Churn Probability', fontsize=12)
ax.set_title('Customer Segmentation Analysis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/customer_segmentation.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Error Handling
```python
# Be specific with exceptions - never use bare except
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: Could not find data file at {data_path}")
    raise
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty")
    raise
except Exception as e:
    print(f"Unexpected error loading data: {e}")
    raise
```

## Working with Claude Code

### When Starting a New Project

1. **Create project structure first:**
```bash
   mkdir -p project-name/{data/{raw,processed},notebooks,src,results/{figures,models}}
   touch project-name/{case_brief.md,README.md}
```

2. **Write case_brief.md before any code**
   - Define the business problem clearly
   - List specific questions to answer
   - Identify success metrics

3. **Create notebooks in sequence:**
   - Start with exploratory analysis
   - Don't skip ahead to modeling
   - Each notebook should build on previous insights

4. **Document as you go:**
   - Explain findings in markdown cells
   - Don't save all documentation for the end

### Best Practices for Analysis

**Exploratory Analysis (Notebook 01):**
- Load data and examine structure
- Check for missing values and data quality issues
- Generate summary statistics
- Create visualizations of key distributions
- Identify potential features for modeling
- Document initial hypotheses

**Feature Engineering (Notebook 02):**
- Transform variables based on EDA insights
- Create derived features
- Handle missing values appropriately
- Encode categorical variables
- Scale/normalize as needed
- Document transformation logic

**Modeling (Notebook 03):**
- Start with simple baseline model
- Try multiple algorithms
- Use cross-validation for evaluation
- Tune hyperparameters systematically
- Compare models objectively
- Select final model with justification

**Evaluation (Notebook 04):**
- Test final model on held-out data
- Generate confusion matrix / error metrics
- Feature importance analysis
- Business impact calculation
- Identify limitations and risks
- Provide implementation recommendations

## Common Pitfalls to Avoid

❌ **Don't:**
- Import code from other project folders
- Use relative imports between projects
- Create a shared `utils/` folder for the repository
- Leave unexplained magic numbers in code
- Skip data validation steps
- Use inconsistent random seeds
- Forget to document data sources
- Create visualizations without clear labels
- Train models without evaluating properly

✅ **Do:**
- Copy and adapt code between projects when needed
- Document why you chose specific approaches
- Validate data quality before analysis
- Use meaningful variable names throughout
- Create reproducible analyses with fixed seeds
- Include business context in technical work
- Write for an audience (future you or MBA peers)
- Test that notebooks run from top to bottom

## Project Completion Checklist

Before considering a project complete:

- [ ] `case_brief.md` clearly defines business problem
- [ ] All notebooks execute without errors (Restart & Run All)
- [ ] Data dictionary documents all variables
- [ ] Visualizations have clear titles and labels
- [ ] Key findings are documented in markdown cells
- [ ] `business_memo.pdf` provides executive summary
- [ ] Project `README.md` summarizes the analysis
- [ ] Main repository `README.md` table is updated
- [ ] Code follows PEP 8 and passes linting
- [ ] No hardcoded file paths (use pathlib)
- [ ] All data sources are documented with links
- [ ] Random seeds are set for reproducibility

## Remember

- **Clarity over cleverness** - Code should be obvious to read
- **Each project stands alone** - No cross-project dependencies
- **Business value first** - Always connect analysis to decisions
- **Document thoroughly** - Future readers (including you) will thank you
- **Reproducibility matters** - Others should be able to replicate your work

---

*This is a learning portfolio focused on demonstrating analytical capabilities and business acumen. Code quality and clear communication are more important than optimization or advanced techniques.*