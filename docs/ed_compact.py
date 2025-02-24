import pandas as pd
import numpy as np
import re


def create_compact_report(input_file: str, output_file: str):
    """Create compact version of the EDA report"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    output_lines = []
    sheets = content.split('Sheet: ')[1:]  # Split by sheets

    for sheet in sheets:
        if not sheet.strip():
            continue

        # Get sheet name and basic info
        sheet_name = sheet.split('\n')[0].strip()
        rows_match = re.search(r'Number of rows: (\d+)', sheet)
        cols_match = re.search(r'Number of columns: (\d+)', sheet)

        output_lines.extend([
            f"Sheet: {sheet_name}",
            f"Number of rows: {rows_match.group(1) if rows_match else 'Unknown'}",
            f"Number of columns: {cols_match.group(1) if cols_match else 'Unknown'}",
            ""
        ])

        # Extract column information
        dtypes = {}
        missing = {}
        stats = {}

        # Get data types
        dtype_section = re.findall(r'(\w+)\s+(\w+)',
                                   sheet[
                                   sheet.find('Column Names and Data Types:'):sheet.find('Missing Values per Column:')])
        for col, dtype in dtype_section:
            dtypes[col] = dtype

        # Get missing values
        missing_section = re.findall(r'(\w+)\s+(\d+)',
                                     sheet[sheet.find('Missing Values per Column:'):sheet.find('Basic Statistics:')])
        for col, count in missing_section:
            missing[col] = count

        # Get statistics
        stats_section = sheet[sheet.find('Basic Statistics:'):sheet.find('Sample Data')]
        numeric_stats = {}
        for line in stats_section.split('\n'):
            if 'mean' in line:
                parts = re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', line)
                if len(parts) >= 4:
                    numeric_stats[line.split()[0]] = {
                        'mean': parts[0],
                        'std': parts[1],
                        'min': parts[2],
                        '25%': parts[3]
                    }

        # Write column summary
        output_lines.append("Column Name,Data Type,Missing Value,Mean,Std,Min,25%")
        for col in dtypes:
            stats_str = ""
            if col in numeric_stats:
                stats_str = f",{numeric_stats[col]['mean']},{numeric_stats[col]['std']},{numeric_stats[col]['min']},{numeric_stats[col]['25%']}"
            output_lines.append(f"{col},{dtypes[col]},{missing.get(col, 0)}{stats_str}")

        # Extract and format sample data
        output_lines.append("")
        sample_data = sheet[sheet.find('Sample Data'):sheet.find('Insights')]
        sample_lines = [line for line in sample_data.split('\n') if line.strip()][:11]  # Header + 10 rows
        output_lines.extend(sample_lines)
        output_lines.extend(["", "=" * 80, ""])

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))


if __name__ == "__main__":
    create_compact_report(
        'EDA_Report_2023_all_results_compiled.txt',
        'EDA_Report_2023_all_results_compiled_compact.txt'
    )