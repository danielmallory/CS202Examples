import torch
import pandas as pd
import torch.cuda as cutorch


# parse args
parser = argparse.ArgumentParser(description='Options:')

# global options
parser.add_argument('-dataset', default='A', help='suffix to log files')

opt = parser.parse_args()

crop_size = 512
nFeatures = 20
c = 0.8


aminos = ['--A', '--R', '--N', '--D', '--C', '--Q', '--E', '--G', '--H', '--I', '--L', '--K', '--M', '--F', '--P', '--S', '--T', '--W', '--Y', '--V']
bg = [0.0799912015849807, 0.0484482507611578, 0.044293531582512, 0.0578891399707563, 0.0171846021407367, 0.0380578923048682, 0.0638169929675978,
              0.0760659374742852, 0.0223465499452473, 0.0550905793661343, 0.0866897071203864, 0.060458245507428, 0.0215379186368154, 0.0396348024787477,
              0.0465746314476874, 0.0630028230885602, 0.0580394726014824, 0.0144991866213453, 0.03635438623143, 0.0700241481678408]

background = pd.DataFrame(aminos,bg)

proteinFile = pd.read_csv(opt.dataset+'.node')
protString = proteinFile.read()
