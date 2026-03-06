import matplotlib.pyplot as plt
from sktime.datasets import load_from_tsfile_to_dataframe

# 1. Point this to one of your training files
file_path = "data/mediapipe/BS/103007/TRAIN_default_X.ts"

# 2. Load the data
print(f"Loading data from {file_path}...")
X, y = load_from_tsfile_to_dataframe(file_path)

# 3. Select the very first video in the dataset (Row 0)
video_data = X.iloc[0]
label = y[0]

# 4. Plot the data
plt.figure(figsize=(12, 6))

# Your data has 79 dimensions. Plotting all 79 is a mess,
# so we will just plot the first 6 to mimic the example image.
# (You can change this range to plot different body parts!)
for i in range(24, 28):
    plt.plot(video_data[i], label=f'Feature {i}')

plt.title(f'Pose Estimation Keypoints Over Time (Label: {label})')
plt.xlabel('Frame / Time Index')
plt.ylabel('Coordinate Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Save the image and show it
plt.savefig("timeseries_example.png")
print("Saved graph as timeseries_example.png!")
plt.show()