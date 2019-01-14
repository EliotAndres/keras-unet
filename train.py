import sys
from utils import *
if len(sys.argv)>1:
  folder = sys.argv[1]
else:
  folder = 'headshoulderdata'
print('seg',folder)
seg = fingernailseg(folder)
# create U-Net model
print('create_unet')
seg.create_unet()
print('fit');
seg.fit()
seg.load_model()
