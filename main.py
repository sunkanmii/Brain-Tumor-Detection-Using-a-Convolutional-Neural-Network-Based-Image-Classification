import os

print("The number of MRI Images labelled 'yes':",len(os.listdir('yes')))
print("The number of MRI Images labelled 'no':",len(os.listdir('no')))