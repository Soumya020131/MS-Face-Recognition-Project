# MS-Face-Recognition-Project
Face Recognition app for identifying missing people and people with criminal records and for preventing crime.
# Project Overview:
Face detection and recognition is a computer technology which leverages the power of AI to locate the presence of human faces in an image or a video. With the advancement of open-source projects, it is now possible to identify and recognize individuals irrespective of skin colour, tone, the position of face and movements.

This project includes the use of face recognition libraries available in python to implement various features for identifying missing people and people with criminal records via images, video clips and live cctv footage to find missing people and it could also be used for the purpose of preventing crime.
It is a web application based on Flask framework for deploying their ML/DL model and I used HTML and CSS for creating the UI for this app. 

Check out the video demo on YouTube:https://youtu.be/XeQ7_Bnqk7Q

GitHub repo link:https://github.com/Soumya020131/MS-Face-Recognition-Project.git

# Project Walkthrough:

# Developed using:
1.HTML and CSS for frontend development.(I used the bootstrap framework)

2.Flask framework written in python for backend development.

3.Various libraries that include: 
   1. OpenCV(It is an open source computer vision and machine learning software library.)
   2. face_recognition(face_recognition is the simplest face detection library built with the deep learning.)
   3. dlib(It is a toolkit for making real world machine learning and data analysis application)
   4. numPy(It is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.)
   5. Pillow(It is a Python Imaging Library (PIL), which adds support for opening, manipulating, and saving images.)

# Guide:
Make sure you have Python installed in your system. 
Our project requires the dependencies listed above to be installed.(Do check the Requirements.txt file for the dependencies that are not installed in your system)
Clone the following repo on your local system : https://github.com/Soumya020131/MS-Face-Recognition-Project.git
Run app.py and go to the generated server to use the app. :)

# Features:
1.Identify people from image: This feature identifies people from an image who have been missing (if any). It also identifies if any person in that image has ever had any criminal record and seems suspicious.

2.Identify people from Video clip: This feature identifies from a video clip the people who have been missing and people who have ever had any criminal record if they are present in that clip.

3.Identify people from Live CCTV camera: This feature identifies people who have been missing and people who have ever had any criminal record in real time from live footage.It could also be used for various security purposes in day-to-day life.

4.UPDATE THE DATASETS FOR MISSING PEOPLE AND CRIMINALS: This feature allows you to enter and remove records in the existing datasets. There is also a feature here that checks if the uploaded image has more than one face and rejects the records with a single person’s name while having multiple faces in the uploaded record. 
[IMPORTANT DISCLAIMER: On updating the datasets if you want to check on the updated datasets the app needs to be RUN AGAIN]

# Can be used to:
1.Look for a person who is missing or has been missing via images, video clips and live cctv footage that could potentially lead to finding the missing person.

2.Get to know if there was a suspicious person with past criminal records with or near the missing person(via images, video clips or live stream service) so that the process of finding the person becomes easy and the rate of crimes could be prevented.

3.CCTV live face recognition feature makes this app a safety and security app that could be used in day-to-day lives for security purposes. On connecting the app with the CCTV one could easily keep track of people who enter in their houses, shops, etc., and be alerted if a person with a criminal record has entered the vicinity.
Update the existing records of missing people and people with criminal records.


