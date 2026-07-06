import sys 
sys.path.insert(0, r"C:\Program Files (x86)\DECTRIS\ALBULA\ALBULA_3.3.3\bin") 
sys.path.insert(0, r"C:\Program Files (x86)\DECTRIS\ALBULA\ALBULA_3.3.3\python") 

import dectris.albula 
dectris_image = dectris.albula.readImage(r"C:\Program Files (x86)\DECTRIS\ALBULA\ALBULA_3.3.0\testData\in16c_010001.cbf") 
main_frame, sub_frame = dectris.albula.display(dectris_image)