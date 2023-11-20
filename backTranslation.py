import btranslation
import pandas as pd
import os

def read_tsv(group, delimiter='\t'):
    """
    Reads a .tsv file and returns a Pandas DataFrame.

    Parameters:
    - file_path (str): Path to the .tsv file.
    - delimiter (str): Delimiter used in the file (default is '\t' for tab-separated values).

    Returns:
    - pd.DataFrame: The DataFrame containing the data from the .tsv file.
    """
    try:
        file_path = os.path.join('data',group,'train.tsv')
        # Read the .tsv file into a DataFrame
        dataframe = pd.read_csv(file_path, delimiter=delimiter, header=None, names=['text', 'emotion', 'labeler'])
        return dataframe
    except Exception as e:
        print(f"Error reading the .tsv file: {e}")
        return None
    
def createFile(group):
    original_df = read_tsv(group)
    btrans = btranslation.back_translation(original_df['Text'].values, language = 'ru')
    translated_df = pd.concat([btrans['Back Translated'], original_df['emotion'], original_df['labeler']], axis=1)
    translated_df = translated_df.rename(columns={'Back Translated': 'text'})
    combined_df = original_df.append(translated_df, ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    path = os.path.join('data',group,'augmented_train.tsv')
    combined_df.to_csv(path, sep='\t', index=False)

createFile('original')
createFile('group')
createFile('ekman')