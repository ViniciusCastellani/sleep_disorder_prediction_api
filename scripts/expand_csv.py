import pandas as pd
import numpy as np

def expand_dataset(df_original, total_final=10000):
    df_clean = df_original.copy()
    
    df_clean['Occupation'] = df_clean['Occupation'].fillna('Other')
    
    allowed_occupations = ['Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Nurse', 'Salesperson', 'Teacher']
    
    df_clean['Occupation'] = df_clean['Occupation'].apply(lambda x: x if x in allowed_occupations else 'Other')
    df_clean['Sleep Disorder'] = df_clean['Sleep Disorder'].fillna('None')
    df_clean['BMI Category'] = df_clean['BMI Category'].replace('Normal Weight', 'Normal')
    
    new_data = []
    n_needed = total_final - len(df_clean)
    
    bp_split = df_clean['Blood Pressure'].str.split('/', expand=True).astype(int)
    df_clean['sys_val'] = bp_split[0]
    df_clean['dia_val'] = bp_split[1]
    
    print(f"Generating {n_needed} new records...")
    
    for _ in range(n_needed):
        base = df_clean.sample(n=1).iloc[0]
        new_row = base.copy()
        
        new_row['Age'] = int(np.clip(base['Age'] + np.random.randint(-2, 3), 20, 65))
        new_row['Sleep Duration'] = round(np.clip(base['Sleep Duration'] + np.random.uniform(-0.5, 0.5), 4, 10), 1)
        new_row['Physical Activity Level'] = int(np.clip(base['Physical Activity Level'] + np.random.randint(-10, 11), 30, 120))
        new_row['Heart Rate'] = int(np.clip(base['Heart Rate'] + np.random.randint(-3, 4), 60, 100))
        new_row['Daily Steps'] = int(np.clip(base['Daily Steps'] + np.random.randint(-800, 801), 2000, 15000))
        
        sys = int(base['sys_val'] + np.random.randint(-4, 5))
        dia = int(base['dia_val'] + np.random.randint(-3, 4))
        new_row['Blood Pressure'] = f"{sys}/{dia}"
        
        new_row['sys_val'] = sys
        new_row['dia_val'] = dia
        
        new_data.append(new_row)
    
    df_final = pd.concat([df_clean, pd.DataFrame(new_data)], ignore_index=True)
    df_final = df_final.drop(columns=['sys_val', 'dia_val'])
    
    df_final['Person ID'] = range(1, len(df_final) + 1)
    
    return df_final

df_original = pd.read_csv("Sleep_health_and_lifestyle_dataset_ORIGINAL.csv")

df_massive = expand_dataset(df_original, total_final=10000)

print("\nOccupation Distribution:")
print(df_massive['Occupation'].value_counts())

df_massive.to_csv('Sleep_Health_Massive_Dataset.csv', index=False)
print(f"\nFile saved with {len(df_massive)} records and ZERO NaNs in Occupation.")