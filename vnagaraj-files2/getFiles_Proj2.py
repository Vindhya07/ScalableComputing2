import requests
import os

url = 'https://cs7ns1.scss.tcd.ie/index.php/?shortname=vnagaraj&download=resume_speed'
myFileNames = requests.get(url).text

print(myFileNames)

for i in myFileNames.split(',\n'):
  file_url = 'https://cs7ns1.scss.tcd.ie/index.php/?shortname=vnagaraj&download=resume_speed&myfilename=' + i
  myFiles = requests.get(file_url)
  myFile = open(i, "wb")
  myFile.write(myFiles.content)
  myFile.close()