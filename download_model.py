import gdown

# Google Drive link for a pretrained emotion detection CNN model (FER2013 dataset)
url = "https://drive.google.com/uc?id=1F5qay4_VoTDcxcncLZmzP1r14EwKw5Cp"
output = "emotion_model.h5"

print("Downloading pretrained emotion detection model...")
gdown.download(url, output, quiet=False)
print("✅ Download complete! Model saved as 'emotion_model.h5'")
