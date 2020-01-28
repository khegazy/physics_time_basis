import matplotlib.pyplot as plt
import numpy as np
import glob



Nbases = 6
experiment = "UED"


bases = {}
for fl in glob.glob("../../basisFxns/{}/*dat".format(experiment)):
  print(fl, fl[fl.find("_L-")+3:fl.find("_M-")])
  L = int(fl[fl.find("_L-")+3:fl.find("_M-")])
  with open(fl, "rb") as file:
    bases[L] = np.fromfile(file, np.double)

for i in range(Nbases):
  basis = bases[2*i]/np.sqrt(np.sum(bases[2*i]**2)) + i
  plt.plot(basis, 'k-')

plt.savefig("../bases.png")
plt.close()

#cos2 = bases[0]*np.sqrt(4*np.pi)/3. + 2.0*bases[2]/3.*np.sqrt(4*np.pi/5.)
#plt.plot(cos2)
#plt.savefig("testc2.png")

