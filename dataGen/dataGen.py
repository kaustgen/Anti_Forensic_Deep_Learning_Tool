# Author: Kaleb Austgen
# Modify Date: 9-25-25
# Purpose:
"""   Goal of this script is to automate the creation of files modified by anti-forensic techniques
      Once the original file is generated the script will choose some anti-forensic techniques
      (Steganography, Encryption, Time-stomping, Data Wiping) and modify the file in a semi-randomized way
      It will then label the original file as original, and the modifed file with the specific technique used
      Once the data is labeled it will add it to a vector database
      The script will generate 100,000 different files, with half of them being modified through data-wiping
      and the other half being modified with steganography

      Generated Adversarial Networks, Convolutional Neural Networks are highly proficient in hiding images, 
      perhaps we can train a model that can counter these?

      Goal: Generate a series of files made by CNNs and other advanced systems and create a system that
      can detect it

      Must be able to detect -
            Least Significant Bit (LSB)
            

      https://sheriffjbabu.medium.com/python-ai-for-steganography-862e732cd3e0 - source
"""


print("Hello world")