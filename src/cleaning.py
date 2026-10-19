import pandas as pd

file_1 = "/Users/macairm1/Desktop/3AECLM2/BE/ML & DS/BE_MachineLearning/data/processed/test_features.csv" 
file_2 = "/Users/macairm1/Desktop/3AECLM2/BE/ML & DS/BE_MachineLearning/data/processed/train_features.csv"

def supprimer_session_id_pandas(input_path, output_path=None):
  # Chargement du CSV
  df = pd.read_csv(input_path)

  # Suppression de la colonne session_id
  df = df.drop(columns=['session_id'], errors='ignore')

  # Sauvegarde si un chemin de sortie est spécifié
  if output_path:
    df.to_csv(output_path, index=False)

  return df



df_clean = supprimer_session_id_pandas(
    file_1, 'data/processed/cleaned_test.csv'
)
df_clean = supprimer_session_id_pandas(
    file_2, 'data/processed/cleaned_train.csv'
)