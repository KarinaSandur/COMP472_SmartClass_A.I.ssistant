# COMP472 SmartClass A.I.ssistant
An A.I assistant that analyzes the facial expressions of students in a class!

## Team Members, Group Information & Link to Repo

Group: AK-2

Karina Sanchez-Duran (ID: 40189860), Role: Data Specialist & Developer

Marina Girgis (ID: 40168639), Role: Evaluation Specialist & Developer

Nadia Beauregard (ID: 40128655), Role: Training Specialist & Developer

Link to repo: https://github.com/KarinaSandur/COMP472_SmartClass_A.I.ssistant

## Steps to Run Python Scripts

<ol type="1">
  <li>Install Anaconda</li>
  <li>Create new environment in Anaconda with python version 3.6 or above.</li>
  <li>Install the following in the newly created environment to be able to run all python scripts</li>
    <ul>
      <li>Pillow</li>
      <li>Numpy</li>
      <li>Matplotlib</li>
      <li>Pytorch</li>
      <li>scikit-learn</li>
    </ul>
  <li>Open the Python files in your chosen IDE and run the scripts inside the newly created environment</li>
  If you are using Visual Studio Code, do the following:
    <ol>
      <li>View >> Command Palette >> Python: Select Interpreter >> [click newly created environement] </li>
      <li>Right click and select the following: Run Python >> Run Python File in Terminal</li>
    </ol>
  
</ol>

## File Explanation

<strong>The following folders relate to the data itself:</strong>

<ol type="1">
  <li>Raw Data</li>
  <ul>
    <li>Contains 4 zip folders: angry, happy, neutral and focused</li>
    <li>The raw data folder contains the images for the four classes before any data cleaning has been done.</li>
  </ul>
  
  <li>Clean Data</li>
  <ul>
    <li>Contains 4 zip folders: angry, happy, neutral and focused</li>
    <li>The clean data folder contains the images for the four classes after data cleaning and labelling has been done.</li>
  </ul>
</ol>

<strong>The following folders relate to documentation:</strong>
<ol type="1">
  <li>Representative Images</li>
  <ul>
    <li>Contains a zip file with 4 subfolders: angry, happy, neutral and focused./li>
    <li>Each subfolder contains 25 images and are meant to represent the data of its respective class.</li>
  </ul>
  
  <li>Reports</li>
  <ul>
    <li>Contains a pdf of the report for Part 1 of the project.</li>
  </ul>
</ol>

<strong>The following Python scripts relate to data cleaning and labelling:</strong>

<ol type="1">
  <li>PNGtoJPEGConverter.py</li>
  <ul>
    <li>Python script that takes a folder path as input (note that the folder cannot be in a zip folder for the code to work).</li>
    <li>The script converts all PNG images in the folder to JPEG</li>
  </ul>
  
  <li>resizeImages.py</li>
  <ul>
    <li>Python script that takes a folder path as input (note that the folder cannot be in a zip folder for the code to work).</li>
    <li>The script resizes all JPEG images in the folder to 150 x 150</li>
  </ul>

  <li>renameImages.py</li>
  <ul>
    <li>Python script that takes a name and folder path as input (note that the folder cannot be in a zip folder for the code to work).</li>
    <li>The script renames all JPEG images in the folder with the name given as the prefix and a number as the suffix. The suffix of the images (number) are renamed in increasing order (e.g. happy1, happy2 etc..)</li>
    
  </ul>
</ol>

<strong>The following Python scripts relate to data visualization:</strong>

<ol type="1">
  <li>classDistribution.py</li>
  <ul>
    <li>The script generates a bar graph showing the number of images in each class: angry, happy, neutral and focused using Matplotlib.</li>
  </ul>
  
  <li>pixelIntensityDistribution.py</li>
  <ul>
    <li>Python script that takes folder path as input that expects to find one or more zip files in said folder.</li>
    <li>The script generates a histogram depicting the pixel intensity distribution for each class</li>
  </ul>

  <li>sampleImagesWithHistogram.py</li>
  <ul>
    <li>Python script that takes folder path as input that expects to find one or more zip files in said folder.</li>
    <li>The script generates histograms depicting the pixel intensity distribution for sample images for each class.</li>
  </ul>
  
</ol>

<strong>The following Python scripts relate to training, evaluating and running models:</strong>

<ol type="1">
  <li>cnn_model.py</li>
  <ul>
    <li>Python script that takes folder path as input that expects to find one or more zip files in said folder.</li>
    <li>The script trains and validates three different models.</li>
    <li>The script evaluates the models by producing a confusion matrix for each and a table summarizing metrics for all models.</li>
    <li>The script generates a .pth file of the best performance model of each model based on the validation set (it doesn't merely save the last one from the training process): best_model_MainModel.pth, best_model_Variant1.pth, and best_model_Variant2.pth. Afterwards, the script compares the metrics of all three .pth files and outputs the best one out of the three: best_performing_model.pth.</li>
  </ul>
  
  <li>evaluateModels.py</li>
  <ul>
    <li>Python script that takes no input</li>
    <li>Note that one must run the cnn_model.py file before running evaluateModel.py because it requires the .pth files cnn_model.py produces in order to run.</li>
    <li>The script loads best_model_MainModel.pth, best_model_Variant1.pth, and best_model_Variant2.pth and evaluates each model by producing a confusion matrix for each and a table summarizing metrics for all models.</li>
  </ul>

  <li>smartAIssistant.py</li>
  <ul>
    <li>Python script that takes an image path or a folder path containing images as input</li>
    <li>If one inputs an image path, the script will classify that image according to one of the 4 facial expressions and output the result/prediction.</li>
    <li>If one inputs an folder path, the script will classify each image in the folder path according to one of the 4 facial expressions and output the result/prediction for each.</li>
  </ul>
  
</ol>

