\# 🚀 For Tanay: How to Add Your Models



\## What You Need to Do (2 Minutes):



\### Step 1: Copy Your Model Files Here

Place all your `.h5` model files in this folder:

C:\\Users\\Akshayaa\\Pnemonia\\ACM\_HACKATHON\\pneumonia-backend\\models\\



text



Expected files:

\- `resnet50\_pneumonia.h5` (or your model name)

\- `densenet121\_pneumonia.h5` (or your model name)

\- Any other `.h5` files

\- `metadata.npy` (optional - if you have model weights)



\### Step 2: Restart the Backend



Open terminal in VS Code (Ctrl + `) and run:

```bash

cd C:\\Users\\Akshayaa\\Pnemonia\\ACM\_HACKATHON\\pneumonia-backend

python main.py

You should see:



text

✅ Loaded: resnet50\_pneumonia.h5

✅ Loaded: densenet121\_pneumonia.h5

🎉 Ensemble ready: 2 models loaded

INFO: Uvicorn running on http://0.0.0.0:8000

Step 3: Test It!

Open frontend: http://localhost:8501



Upload an X-ray image



Click "Analyze with PneumoAI"



Check prediction is from real model (not dummy)



Important Information Needed:

Please confirm these details:



1\. Preprocessing Used During Training:

Image size: 224x224 or \_\_\_?



Normalization: Divide by 255? Or ImageNet mean/std?



Color mode: RGB or Grayscale?



2\. Class Labels (in order):

Class 0 = ?



Class 1 = ?



Class 2 = ?



3\. Model Architecture:

ResNet-50?



DenseNet-121?



Other: \_\_\_?



4\. Output Format:

Returns probabilities \[0.95, 0.03, 0.02]?



Or returns class index?



If You Get Errors:

Error: "Could not load model"

Fix: Models might be from different TensorFlow version. Check version with:



python

import tensorflow as tf

print(tf.\_\_version\_\_)  # Should be 2.20.0

Error: "Wrong input shape"

Fix: Update preprocessing in utils/preprocessing.py to match your training.



Error: "Class names wrong"

Fix: Create metadata.npy file:



python

import numpy as np

metadata = {

&nbsp;   'class\_names': \['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia'],

&nbsp;   'input\_size': (224, 224)

}

np.save('metadata.npy', metadata)

Questions?

Ask Akshayaa or check main SETUP\_COMPLETE.md



Thanks for the models! 🎉



text



\*\*\*



\## Steps to Fix Your Notepad:



\### Option 1: Replace Everything

1\. \*\*Select All in Notepad\*\* (Ctrl + A)

2\. \*\*Delete\*\* (Delete key)

3\. \*\*Paste the corrected version above\*\* (copy from the code block)

4\. \*\*Save\*\* (Ctrl + S)



\### Option 2: Keep What You Have (It's Actually OK!)

\*\*Your current version works fine!\*\* The formatting looks messy to me because you copied it into chat, but in the actual file it's probably fine.



\*\*Just add these missing lines at the end:\*\*



Questions?

Ask Akshayaa or check main SETUP\_COMPLETE.md



Thanks for the models! 🎉



text



\*\*\*



\## Quick Check:



\*\*In your Notepad, you should have these main sections:\*\*

\- \[ ] Step 1: Copy Your Model Files Here

\- \[ ] Step 2: Restart the Backend

\- \[ ] Step 3: Test It!

\- \[ ] Important Information Needed

\- \[ ] If You Get Errors

\- \[ ] Questions? (Add this if missing)



\*\*\*



\## My Recommendation:



\*\*Your current version is 90% correct!\*\* 



\*\*Just add the "Questions?" section at the end, then:\*\*

1\. \*\*Save\*\* (Ctrl + S)

2\. \*\*Close Notepad\*\*

3\. \*\*Move to next step\*\* (creating SETUP\_COMPLETE.md)



\*\*\*



\*\*Is it saved now? Ready to move to the next documentation file?\*\* 📝

