# %%
import pandas as pd
from transformers import ASTModel
import torch
from transformers import ASTFeatureExtractor
import numpy as np

processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# %%
# Load in the csv file
df = pd.read_csv("D:/archive/undersampled_data.csv")

# Data file folder
data_dir = "D:/archive/new_sort"

from scipy.io import wavfile
import scipy.signal as sps
from tqdm import tqdm

df.keys()

# %%
# file_path = data_dir + "/" + df["uuid"][0] + ".wav"
# # Load the audio file
# sr, audio = wavfile.read(file_path)
# # Resample
# number_of_samples = round(len(audio) * float(16000) / sr)
# audio = sps.resample(audio, number_of_samples)/32768
# print(min(audio))
# # Extract the features
# features = processor(audio, sampling_rate=16000, return_tensors="pt")
# # Get the prediction
# with torch.no_grad():
#     output = model(**features)

#     print(output.pooler_output)

# %%
# transformed output
output_list = []


# Loop through each row in the dataframe, and do a feature extraction
for index, row in df.iterrows():
    file_path = data_dir + "/" + row["uuid"] + ".wav"
    print(file_path)

    # Load the audio file
    sr, audio = wavfile.read(file_path)

    # Resample
    number_of_samples = round(len(audio) * float(16000) / sr)
    audio = sps.resample(audio, number_of_samples)/32768

    
    if len(audio.shape) == 1:
        # Extract the features
        features = processor(audio, sampling_rate=16000, return_tensors="pt", padding="max_length")

        # Get the prediction
        with torch.no_grad():
            output_list.append(model(**features).pooler_output)
            print(index)
            del sr, audio, number_of_samples, features
    else:
        # Extract the features Some are dual track ...
        #print(max(np.transpose(audio)[0]),max(np.transpose(audio)[1]))
        features = processor(np.transpose(audio)[0], sampling_rate=16000, return_tensors="pt", padding="max_length")

        # Get the prediction
        with torch.no_grad():
            output_list.append(model(**features).pooler_output)
            print(index)
            del sr, audio, number_of_samples, features

# Add the output_list to the dataframe
df["transformed_data"] = output_list

# Save the dataframe
df.to_csv("D:/archive/undersampled_data_with_transform.csv", index=False)


