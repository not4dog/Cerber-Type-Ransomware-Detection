import pefile
import numpy as np
import os

PATH = ' '
filename = os.listdir(PATH)

def opcode_Get(file_path):
    try:
        pe = pefile.PE(file_path,fast_load=True)

        for section in pe.sections:
            if '.text' in str(section.Name):
                entry = section.PointerToRawData - 1
                end = section.SizeOfRawData + entry
                raw_data = pe.__data__[entry:end]
                data = np.frombuffer(raw_data, dtype = np.float32)                
        return np.nan_to_num(data)
            
    except: 
        return