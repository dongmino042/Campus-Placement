# Data Directory

This directory contains the dataset for the Campus Placement prediction project.

## Dataset Information

- **Name**: Factors Affecting Campus Placement
- **Source**: [Kaggle](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)
- **Description**: This dataset contains various factors that may affect campus placement of students, including their academic performance, work experience, specialization, etc.

## Download Instructions

### Option 1: Manual Download
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)
2. Click on the "Download" button
3. Extract the downloaded ZIP file
4. Place the `Placement_Data_Full_Class.csv` file in this `data/` directory

### Option 2: Using Kaggle API
1. Install the Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Set up your Kaggle API credentials:
   - Go to your Kaggle account settings
   - Create a new API token (downloads `kaggle.json`)
   - Place `kaggle.json` in `~/.kaggle/` directory
   - On Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

3. Download the dataset:
   ```bash
   kaggle datasets download -d benroshan/factors-affecting-campus-placement
   unzip factors-affecting-campus-placement.zip -d data/
   ```

## Expected Files

After downloading, this directory should contain:
- `Placement_Data_Full_Class.csv` - The main dataset file

## Dataset Schema

The dataset contains the following columns:
- `sl_no`: Serial Number
- `gender`: Gender (M/F)
- `ssc_p`: Secondary Education percentage (10th Grade)
- `ssc_b`: Board of Education (Central/Others)
- `hsc_p`: Higher Secondary Education percentage (12th Grade)
- `hsc_b`: Board of Education (Central/Others)
- `hsc_s`: Specialization in Higher Secondary Education
- `degree_p`: Degree Percentage
- `degree_t`: Type of undergraduate degree
- `workex`: Work Experience (Yes/No)
- `etest_p`: Employability test percentage
- `specialisation`: MBA Specialization
- `mba_p`: MBA percentage
- `status`: Placement Status (Placed/Not Placed) - **Target Variable**
- `salary`: Salary offered (only for placed students)

## Note

**Important**: The CSV file is not included in this repository due to licensing and size constraints. Please download it using one of the methods above.
