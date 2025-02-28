# Portfolio Optimization Project

## Setup and Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run main portfolio scoring
python integration.py

# Run location clustering analysis
python location_clustering.py
```

## Code Standards
- **Imports**: Standard imports first (os, datetime), then third-party packages (pandas, numpy, sklearn), then local modules
- **Formatting**: Follow PEP 8 guidelines with 4-space indentation
- **Error Handling**: Use try/except blocks with specific error logging
- **Logging**: Use Python's logging module with appropriate levels (INFO, WARNING, ERROR)
- **Documentation**: Use docstrings with Parameters and Returns sections
- **Variable Naming**: Use snake_case for variables and functions, CamelCase for classes
- **Types**: Annotate function parameters and return types when feasible
- **Data Loading**: Use relative paths for CSV files, create directories with os.makedirs(path, exist_ok=True)

## Project Structure
- **Scorers**: Category scoring modules (A, B, C, D) for portfolio analysis
- **Integration**: Main pipeline in integration.py
- **Analysis**: Location clustering in location_clustering.py
- **Visualization**: Plots stored in results/visualizations/
- **Results**: Output saved to results/ directory with timestamps