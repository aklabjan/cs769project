import pandas as pd
import os
from BackTranslation import BackTranslation
import sys

trans = BackTranslation()

def read_tsv(group, file, delimiter='\t'):
    """
    Reads a .tsv file and returns a Pandas DataFrame.

    Parameters:
    - file_path (str): Path to the .tsv file.
    - delimiter (str): Delimiter used in the file (default is '\t' for tab-separated values).

    Returns:
    - pd.DataFrame: The DataFrame containing the data from the .tsv file.
    """
    try:
        file_path = os.path.join('data', group, file)
        # Read the .tsv file into a DataFrame
        dataframe = pd.read_csv(file_path, delimiter=delimiter, header=None, names=['text', 'emotion', 'labeler'])
        return dataframe
    except Exception as e:
        print(f"Error reading the .tsv file: {e}")
        return None

group = sys.argv[1]
df = read_tsv(group, 'train.tsv')
length = len(read_tsv(group,'augmented_train.tsv'))
df = df[length:]

chunk_size = 50
chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

chunk_number = 0
for chunk in chunks:
    count = 0
    while count < 2:
        try:
            output = []
            for line in chunk['text'].values.tolist():
                result = trans.translate(line, src='en', tmp = 'ru',sleeping = 0.6)
                output.append(result.result_text)
            translated_df = pd.DataFrame({
            'text': output,
            'emotion': chunk['emotion'],
            'labeler': chunk['labeler']
            })
            # Specify the path to the existing .tsv file
            existing_file_path = os.path.join('data',group,'augmented_train.tsv')

            # Append the new_data DataFrame to the existing .tsv file
            translated_df.to_csv(existing_file_path, sep='\t', mode='a', header=False, index=False)
            print(f'completed {chunk_number}') 
            chunk_number += 1
            break
        except Exception as e: 
            print(f'Error: {e}')
            count += 1
