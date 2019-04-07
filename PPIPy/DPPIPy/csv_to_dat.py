import torch.cuda as cutorch
import pandas as pd

cutorch.set_device(0)

ppiFile = pd.read_csv('/data/human.dimer').to_csv('humanCV-dimer_labels.dat', sep='|')