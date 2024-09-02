# # import streamlit as st
# # import torch
# # from torchvision import models
# # import torchvision
# # from torchvision import transforms
# # from torch.utils.data import Dataset
# # import numpy as np
# # import cv2
# # import face_recognition
# # from torch import nn
# # from torch.autograd import Variable
# # import os
# # import warnings

# # warnings.filterwarnings("ignore")

# # # Directory for storing uploads
# # UPLOAD_FOLDER = 'Uploaded_Files'
# # if not os.path.exists(UPLOAD_FOLDER):
# #     os.makedirs(UPLOAD_FOLDER)

# # # Creating Model Architecture
# # class Model(nn.Module):
# #     def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
# #         super(Model, self).__init__()
# #         model = models.resnext50_32x4d(pretrained=True)
# #         self.model = nn.Sequential(*list(model.children())[:-2])
# #         self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
# #         self.relu = nn.LeakyReLU()
# #         self.dp = nn.Dropout(0.4)
# #         self.linear1 = nn.Linear(2048, num_classes)
# #         self.avgpool = nn.AdaptiveAvgPool2d(1)

# #     def forward(self, x):
# #         batch_size, seq_length, c, h, w = x.shape
# #         x = x.view(batch_size * seq_length, c, h, w)
# #         fmap = self.model(x)
# #         x = self.avgpool(fmap)
# #         x = x.view(batch_size, seq_length, 2048)
# #         x_lstm, _ = self.lstm(x, None)
# #         return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# # # Softmax for final output
# # sm = nn.Softmax()

# # # Prediction function
# # def predict(model, img):
# #     fmap, logits = model(img)
# #     weight_softmax = model.linear1.weight.detach().cpu().numpy()
# #     logits = sm(logits)
# #     _, prediction = torch.max(logits, 1)
# #     confidence = logits[:, int(prediction.item())].item() * 100
# #     return [int(prediction.item()), confidence]

# # # Dataset class for processing videos
# # class ValidationDataset(Dataset):
# #     def __init__(self, video_names, sequence_length=60, transform=None):
# #         self.video_names = video_names
# #         self.transform = transform
# #         self.count = sequence_length

# #     def __len__(self):
# #         return len(self.video_names)

# #     def __getitem__(self, idx):
# #         video_path = self.video_names[idx]
# #         frames = []
# #         a = int(100 / self.count)
# #         first_frame = np.random.randint(0, a)
# #         for i, frame in enumerate(self.frame_extract(video_path)):
# #             faces = face_recognition.face_locations(frame)
# #             try:
# #                 top, right, bottom, left = faces[0]
# #                 frame = frame[top:bottom, left:right, :]
# #             except:
# #                 pass
# #             frames.append(self.transform(frame))
# #             if len(frames) == self.count:
# #                 break
# #         frames = torch.stack(frames)
# #         frames = frames[:self.count]
# #         return frames.unsqueeze(0)

# #     def frame_extract(self, path):
# #         vidObj = cv2.VideoCapture(path)
# #         success = 1
# #         while success:
# #             success, image = vidObj.read()
# #             if success:
# #                 yield image

# # # Main function to detect fake video
# # def detect_fake_video(video_path):
# #     im_size = 112
# #     mean = [0.485, 0.456, 0.406]
# #     std = [0.229, 0.224, 0.225]
    
# #     train_transforms = transforms.Compose([
# #         transforms.ToPILImage(),
# #         transforms.Resize((im_size, im_size)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean, std)
# #     ])
    
# #     video_dataset = ValidationDataset([video_path], sequence_length=20, transform=train_transforms)
# #     model = Model(2)
# #     model.load_state_dict(torch.load('model/df_model.pt', map_location=torch.device('cpu')))
# #     model.eval()
    
# #     prediction = predict(model, video_dataset[0])
# #     return prediction

# # # Streamlit UI
# # st.title("Deepfake Video Detection")

# # uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
# # if uploaded_video is not None:
# #     # Save uploaded video to disk
# #     video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
# #     with open(video_path, "wb") as f:
# #         f.write(uploaded_video.getbuffer())
    
# #     # Call the fake video detection model
# #     st.write("Processing video...")
# #     prediction = detect_fake_video(video_path)
    
# #     # Display the results
# #     output = "REAL" if prediction[0] == 1 else "FAKE"
# #     confidence = prediction[1]
# #     st.write(f"Prediction: {output}")
# #     st.write(f"Confidence: {confidence:.2f}%")

# #     # Optionally remove the video after prediction
# #     os.remove(video_path)
# import streamlit as st
# import torch
# from torchvision import models
# import torchvision
# from torchvision import transforms
# from torch.utils.data import Dataset
# import numpy as np
# import cv2
# import face_recognition
# from torch import nn
# from torch.autograd import Variable
# import os
# import warnings

# warnings.filterwarnings("ignore")

# # Directory for storing uploads
# UPLOAD_FOLDER = 'Uploaded_Files'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # Creating Model Architecture
# class Model(nn.Module):
#     def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
#         super(Model, self).__init__()
#         model = models.resnext50_32x4d(pretrained=True)
#         self.model = nn.Sequential(*list(model.children())[:-2])
#         self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
#         self.relu = nn.LeakyReLU()
#         self.dp = nn.Dropout(0.4)
#         self.linear1 = nn.Linear(2048, num_classes)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x):
#         batch_size, seq_length, c, h, w = x.shape
#         x = x.view(batch_size * seq_length, c, h, w)
#         fmap = self.model(x)
#         x = self.avgpool(fmap)
#         x = x.view(batch_size, seq_length, 2048)
#         x_lstm, _ = self.lstm(x, None)
#         return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

#     def get_heatmap(self, fmap, weight):
#         """ Generate heatmap from feature map and class weight """
#         _, nc, h, w = fmap.shape
#         heatmap = np.zeros((h, w), dtype=np.float32)

#         # Compute the weighted combination of class weights and feature maps
#         for i in range(nc):
#             heatmap += weight[i] * fmap[0, i, :, :].detach().numpy()
        
#         # Normalize the heatmap between 0 and 1
#         heatmap = np.maximum(heatmap, 0)
#         heatmap = heatmap / heatmap.max()
#         return heatmap

# # Softmax for final output
# sm = nn.Softmax()

# # Prediction function
# def predict(model, img):
#     fmap, logits = model(img)
#     weight_softmax = model.linear1.weight.detach().cpu().numpy()
#     logits = sm(logits)
#     _, prediction = torch.max(logits, 1)
#     confidence = logits[:, int(prediction.item())].item() * 100
    
#     # Get heatmap for the predicted class
#     heatmap = model.get_heatmap(fmap, weight_softmax[int(prediction.item())])
    
#     return [int(prediction.item()), confidence, heatmap]

# # Dataset class for processing videos
# class ValidationDataset(Dataset):
#     def __init__(self, video_names, sequence_length=60, transform=None):
#         self.video_names = video_names
#         self.transform = transform
#         self.count = sequence_length

#     def __len__(self):
#         return len(self.video_names)

#     def __getitem__(self, idx):
#         video_path = self.video_names[idx]
#         frames = []
#         a = int(100 / self.count)
#         first_frame = np.random.randint(0, a)
#         for i, frame in enumerate(self.frame_extract(video_path)):
#             faces = face_recognition.face_locations(frame)
#             try:
#                 top, right, bottom, left = faces[0]
#                 frame = frame[top:bottom, left:right, :]
#             except:
#                 pass
#             frames.append(self.transform(frame))
#             if len(frames) == self.count:
#                 break
#         frames = torch.stack(frames)
#         frames = frames[:self.count]
#         return frames.unsqueeze(0)

#     def frame_extract(self, path):
#         vidObj = cv2.VideoCapture(path)
#         success = 1
#         while success:
#             success, image = vidObj.read()
#             if success:
#                 yield image

# # Main function to detect fake video
# def detect_fake_video(video_path):
#     im_size = 112
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
    
#     train_transforms = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((im_size, im_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
    
#     video_dataset = ValidationDataset([video_path], sequence_length=20, transform=train_transforms)
#     model = Model(2)
#     model.load_state_dict(torch.load('model/df_model.pt', map_location=torch.device('cpu')))
#     model.eval()
    
#     prediction = predict(model, video_dataset[0])
#     return prediction

# # Streamlit UI
# st.title("Deepfake Video Detection")

# uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
# if uploaded_video is not None:
#     # Save uploaded video to disk
#     video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
#     with open(video_path, "wb") as f:
#         f.write(uploaded_video.getbuffer())
    
#     # Call the fake video detection model
#     st.write("Processing video...")
#     prediction = detect_fake_video(video_path)
    
#     # Display the results
#     output = "REAL" if prediction[0] == 1 else "FAKE"
#     confidence = prediction[1]
#     st.write(f"Prediction: {output}")
#     st.write(f"Confidence: {confidence:.2f}%")
    
#     # Highlight fake regions in the frames
#     st.write("Highlighting fake regions...")
    
#     # Convert heatmap to a color overlay
#     heatmap = prediction[2]
#     original_frames = [frame for frame in ValidationDataset([video_path]).frame_extract(video_path)]
    
#     # Overlay the heatmap on the original frame
#     for i, original_frame in enumerate(original_frames):
#         if i >= 20:  # Limit to sequence length of 20
#             break
        
#         # Resize heatmap to match the frame size
#         heatmap_resized = cv2.resize(heatmap, (original_frame.shape[1], original_frame.shape[0]))
#         heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
#         # Overlay heatmap on the original frame
#         overlay = cv2.addWeighted(original_frame, 0.5, heatmap_colored, 0.5, 0)
        
#         # Display frame with overlay
#         st.image(overlay, channels="BGR", caption=f"Frame {i+1}")
    
#     # Optionally remove the video after prediction
#     os.remove(video_path)
import os
import cv2
import face_recognition
import tempfile
import streamlit as st
from video_processing import detect_fake_video
import matplotlib.pyplot as plt
import numpy as np
import ffmpeg

# Video frame extraction function
def frame_extract(path):
    """Extract frames from a video file."""
    vidObj = cv2.VideoCapture(path)
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            yield image

# Function to create face-cropped videos
def create_face_videos(file_path, out_dir):
    """Process a video file and save cropped face videos."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_path = os.path.join(out_dir, "processed_video.mp4")
    frames = []
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))

    
    for idx, frame in enumerate(frame_extract(file_path)):
        if idx <= 150:
            frames.append(frame)
            if len(frames) == 4:
                all_faces = []
                for frm in frames:
                    rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    all_faces.extend(face_locations)
                
                if not all_faces:
                    print(f"No faces detected in frames")
                
                for (top, right, bottom, left) in all_faces:
                    for i in range(len(frames)):
                        try:
                            face_image = frames[i][top:bottom, left:right]
                            if face_image.size == 0:
                                continue
                            face_image = cv2.resize(face_image, (112, 112))
                            out.write(face_image)
                        except Exception as e:
                            print(f"Error processing frame {i}: {e}")
                frames = []
    
    out.release()
    if os.path.exists(out_path):
        print(f"Finished processing video: {out_path}")
    else:
        print(f"Failed to save processed video: {out_path}")
    
    return out_path

def main():
    st.title("Deepfake Video Detection")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        processed_video_path = create_face_videos(video_path, 'Processed_Videos/')
        st.write(f"Video path: {processed_video_path}")
    

# Display the first frame
        


        
        
        if st.button("Analyze Video"):
            prediction = detect_fake_video(processed_video_path)
            output = "REAL" if prediction[0] == 1 else "FAKE"
            confidence = prediction[1]
            st.write(f"Prediction: {output} with {confidence:.2f}% confidence")
            


      
if __name__ == "__main__":
    main()
