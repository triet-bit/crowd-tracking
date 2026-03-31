# config.py
import numpy as np

CONF_THRESHOLD = 0.4
THRESHOLD_TIME = 2  
VELOCITY_THRESHOLD = 3.0  

# Polygon definition
polygon = np.array([
    (50,50), #trái trên
    (1500,50), #phải trên
    (1500,900), #phải dưới
    (50,900) ])  #trái dưới
    
