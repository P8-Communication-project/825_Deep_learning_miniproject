# %%
import pandas as pd
from transformers import AutoProcessor, ASTModel
import torch

processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

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
import torchaudio
import numpy as np
# transformed output
output_list_pitch_up = []
output_list_pitch_down = []
output_list_noise = []

# Transform
transform_up = torchaudio.transforms.PitchShift(16000, 7)
transform_down = torchaudio.transforms.PitchShift(16000, -7)

SNR = torch.tensor(5)

# Loop through each row in the dataframe, and do a feature extraction
for index, row in df.iterrows():
    file_path = data_dir + "/" + row["uuid"] + ".wav"
    # Load the audio file
    sr, audio = wavfile.read(file_path)
    # Resample
    number_of_samples = round(len(audio) * float(16000) / sr)
    audio = sps.resample(audio, number_of_samples)/32768

    input = []

    if len(audio.shape) == 1:
        # Augment turn it into tensor
        audio = torch.tensor(np.asarray(audio).astype(np.float32))
        # Add noise
        noise = torch.randn(audio.size())
        
        input.append(torchaudio.functional.add_noise(torch.unsqueeze(audio,0), torch.unsqueeze(noise,0), torch.unsqueeze(SNR,0)).detach().numpy()[0])
    
        input.append(transform_up(audio).detach().numpy())
        input.append(transform_down(audio).detach().numpy())
        
    else:
        print(index)
        print(audio.shape)
        print(audio[0].shape)
        # Augment turn it into tensor
        audio = torch.tensor(np.transpose(np.asarray(audio))[0].astype(np.float32))
        # Add noise
        noise = torch.randn(audio.size())
        
        input.append(torchaudio.functional.add_noise(torch.unsqueeze(audio,0), torch.unsqueeze(noise,0), torch.unsqueeze(SNR,0)).detach().numpy()[0])
    
        input.append(transform_up(audio).detach().numpy())
        input.append(transform_down(audio).detach().numpy())

    print(input[0].shape)
    print(input[1].shape)
    print(input[2].shape)
    if len(audio.shape) == 1:
        # Extract the features
        features = []
        for i in range(3):
            features.append(processor(input[i], sampling_rate=16000, return_tensors="pt", padding = "max_length"))
        # Get the prediction
        #print(features)
        with torch.no_grad():
            output_list_noise.append(model(**features[0]).pooler_output)
            output_list_pitch_up.append(model(**features[1]).pooler_output)
            output_list_pitch_down.append(model(**features[2]).pooler_output)
            
            print(index)
            del sr, audio, number_of_samples, features
    else:
        # Extract the features Some are dual track ...
        features = []
        for i in range(3):
            features.append(processor(input[i][0], sampling_rate=16000, return_tensors="pt", padding = "max_length"))


        # Get the prediction
        with torch.no_grad():
            output_list_noise.append(model(**features[0]).pooler_output)
            output_list_pitch_up.append(model(**features[1]).pooler_output)
            output_list_pitch_down.append(model(**features[2]).pooler_output)
            print(index)
            del sr, audio, number_of_samples, features

# Add the output_list to the dataframe
df["pitch_up"] = output_list_pitch_up
df["pitch_down"] = output_list_pitch_down
df["noise"] = output_list_noise


# Save the dataframe
df.to_csv("D:/archive/augmented_undersampled_data_with_transform.csv", index=False)


