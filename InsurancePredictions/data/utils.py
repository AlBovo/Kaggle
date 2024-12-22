import csv
from datetime import datetime

def load_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data[0], data[1:]

def parse_row(row: list[str]) -> list:
    # TODO: understand what value to put if missing data
    return [
        int(float(row[1])) if row[1] != '' else 40,  # age
        row[2] == 'Female', # gender
        int(float(row[3])) if row[3] != '' else 32*pow(10, 3),  # annual income
        ['Divorced', 'Married', 'Single'].index(row[4]) if row[4] != '' else 2,  # marital status
        int(float(row[5])) if row[5] != '' else 2,  # number of dependents
        ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'].index(row[6]) if row[6] != '' else 0,  # education
        ['Employed', 'Self-Employed', 'Unemployed'].index(row[7]) if row[7] != '' else 2,  # employment status
        float(row[8]) if row[8] != '' else 25.6,  # health score
        ['Suburban', 'Rural', 'Urban'].index(row[9]) if row[9] != '' else 0,  # location
        ['Premium', 'Comprehensive', 'Basic'].index(row[10]) if row[10] != '' else 0,  # insurance plan
        int(float(row[11])) if row[11] != '' else 1,  # previous claims
        int(float(row[12])) if row[12] != '' else 9,  # months since last claim
        int(float(row[13])) if row[13] != '' else 593,  # credit score
        int(float(row[14])) if row[14] != '' else 5,  # insurance duration
        datetime.strptime(row[15], "%Y-%m-%d %H:%M:%S.%f").timestamp(), # policy start date
        ['Average', 'Poor', 'Good'].index(row[16]) if row[16] != '' else 0,  # customer feedbacks
        ['No', 'Yes'].index(row[17]),  # smoking
        ['Weekly', 'Monthly', 'Daily', 'Rarely'].index(row[18]) if row[18] != '' else 0,  # exercise frequency
        ['House', 'Condo', 'Apartment'].index(row[19]),  # property type
    ]