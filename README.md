# COMP472 SmartClass A.I.ssistant
An A.I assistant that analyzes the facial expressions of students in a class!

## Team Members & Group Information

Group: AK-2

Karina Sanchez-Duran (ID: 40189860), Role: Data Specialist & Developer

Marina Girgis (ID: 40168639), Role: Evaluation Specialist & Developer

Nadia Beauregard (ID: 40128655), Role: Training Specialist & Developer

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

<strong>The following Python scripts related to data cleaning and labelling:</strong>

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

<strong>The following Python scripts related to data visualization:</strong>

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

