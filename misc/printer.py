# Project:
#   Localized Questions in VQA
# Description:
#   Functions to print stuff in the console
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

def print_section(section_name):
    print(40*"~")
    print(section_name)
    print(40*"~")

def print_line():
    print(40*'-')

def print_event(text):
    print('-> Now doing:', text, '...')