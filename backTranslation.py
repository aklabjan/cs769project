import btranslation
import pandas as pd
import os
import time

def read_tsv(group, delimiter='\t'):
    try:
        file_path = os.path.join('data', group, 'train.tsv')
        dataframe = pd.read_csv(file_path, delimiter=delimiter, header=None, names=['text', 'emotion', 'labeler'])
        return dataframe
    except Exception as e:
        print(f"Error reading the .tsv file: {e}")
        return None

def createFile(group, chunk_size=100, max_attempts=2):
    original_df = read_tsv(group)

    # Split the DataFrame into chunks
    chunks = [original_df[i:i + chunk_size] for i in range(0, len(original_df), chunk_size)]

    translated_dfs = []

    for chunk in chunks:
        attempts = 0
        while attempts < max_attempts:
            try:
                # Use the entire chunk for back translation
                btrans = btranslation.back_translation(chunk['text'].values, language='ru')
                
                # Combine the original chunk and the translated text
                translated_df = pd.DataFrame({
                    'text': btrans['Back Translated'],
                    'emotion': chunk['emotion'],
                    'labeler': chunk['labeler']
                })

                translated_dfs.append(translated_df)
                break  # Break the while loop if successful
            except Exception as e:
                print(f"Error in chunk translation: {e}")
                attempts += 1
                if attempts < max_attempts:
                    print(f"Retrying in 2 minutes (attempt {attempts}/{max_attempts})")
                    # Sleep for 2 minutes before retrying
                    time.sleep(120)
                continue

    # Concatenate all translated chunks
    translated_df = pd.concat(translated_dfs, ignore_index=True)

    # Combine the original DataFrame and the translated DataFrame
    combined_df = pd.concat([original_df, translated_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates()

    path = os.path.join('data', group, 'augmented_train.tsv')
    combined_df.to_csv(path, sep='\t',header = False,  index=False)

createFile('original')
createFile('group')
createFile('ekman')

