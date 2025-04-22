import numpy as np
from matplotlib import pyplot as plt
import re

ifile=open('unweighted_events.lhe','r').read()

#find all events containing muons and antimuons
#assuming all pairs of muons and antimuons belong to a common event (array lengths are equal)
muons=re.findall(' 13 .+',ifile)[2:]
antimuons=re.findall(' -13 .+',ifile)
muons=[re.split(' +',x)[1:] for x in muons]
antimuons=[re.split(' +',x)[1:] for x in antimuons]
muons=np.array(muons,float)
antimuons=np.array(antimuons,float)

muon_energies=muons[:,9]
antimuon_energies=antimuons[:,9]

muon_momentum=muons[:,6:9]
antimuon_momentum=antimuons[:,6:9]

masses=np.sqrt((muon_energies+antimuon_energies)**2-np.linalg.norm(muon_momentum+antimuon_momentum,axis=1)**2)

muon_z_momentum=np.array([np.zeros(len(muon_momentum),float),np.zeros(len(muon_momentum),float),muon_momentum[:,2]]).T
antimuon_z_momentum=np.array([np.zeros(len(muon_momentum),float),np.zeros(len(muon_momentum),float),antimuon_momentum[:,2]]).T

muonpseudorapids=np.einsum('ij,ij->i',muon_momentum,muon_z_momentum)/(np.linalg.norm(muon_momentum,axis=1)*muon_momentum[:,2])
muonpseudorapids=-np.log(np.tan(muonpseudorapids))

antimuonpseudorapids=np.einsum('ij,ij->i',antimuon_momentum,antimuon_z_momentum)/(np.linalg.norm(antimuon_momentum,axis=1)*antimuon_momentum[:,2])
antimuonpseudorapids=-np.log(np.tan(antimuonpseudorapids))

#plt.hist(muonpseudorapids,100,range=(0,6))
#plt.title('Muon pseudorapidity')

#plt.hist(antimuonpseudorapids,100,range=(0,6))
#plt.title('Antimuon pseudorapidity')

plt.hist(masses,1000,(0,200))
plt.title('Invariant masses')
plt.show()