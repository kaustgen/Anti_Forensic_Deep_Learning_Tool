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
from fpdf import FPDF
import random
from steg_gen import Steg_Gen

# pdf = FPDF()
# pdf.add_page()
# pdf.set_font("Arial", size=12)
# pdf.cell(200, 10, txt="Hello, this is a simple sentence in a PDF!", ln=True, align="L")

# pdf.output("hello.pdf")

def generate_pdf(text, output_file):
      pdf = FPDF()
      pdf.add_page()
      pdf.set_font("Arial", size=12)
      pdf.cell(200, 10, txt=text, ln=True, align="L")

      pdf.output(f"{output_file}")

sentence = ['My secret', 'My super duper secret', 'joly beans, holy greens, and billy jeen', 'woah oh ohhhhhh']

cover = ['cover_falls.jpg', 'cover_.jpg', 'cover_.jpg', 'cover_.jpg']

steg = Steg_Gen()

# Generate 100 different items
for i in range(0, 1):

      cover_pick = 

      sentence_pick = sentence[random.randint(0, 3)]

      if n == 0:
      
      elif n == 1:
      
      elif n == 2:
      
      elif n == 3:
      
      else:
            print("Out of bounds error")


