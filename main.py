import argparse
import warnings
import faulthandler
from utils.logger import *
from bgt import main_bgt

faulthandler.enable()
warnings.filterwarnings('ignore')

# torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description="args for FedGNN")
parser.add_argument('--embed_size', type = int, default=8)
parser.add_argument('--lr', type = float, default = 0.05)
parser.add_argument('--data', required=True,)        #key arg  'ml-1m' / 'ml-100k' / 'filmtrust'    
parser.add_argument('--user_batch', type = int) # key arg 6040 / 943 / 874
parser.add_argument('--clip', type= float, default = 0.1)
parser.add_argument('--total_budget', type= float, required=True) #key arg 100 /400 /400
parser.add_argument('--negative_sample', type = int, required=True)  #key arg 400 / 50 /50
parser.add_argument('--valid_step', type = int, default = 1)
parser.add_argument('--T', type = int, required=True)               #key arg 200
parser.add_argument('--neighbour', type = str, default = 'wo_neighbour')
parser.add_argument('--selected_item', type = str, default ='full')
parser.add_argument('--group', type = bool, default = False)
parser.add_argument('--group_frac', type = float, default = 0)
parser.add_argument('--group_size', type = int, default = 0)
parser.add_argument('--top_k', type = list, default = [1,3,5,10,20,25,50])
parser.add_argument('--dp_mechnisim', required=True )  #key arg 'Gaussian' / 'laplace'
parser.add_argument('--allocation', type=str, default='BGTplanner', help='allocation strategy', required=True)
parser.add_argument('--min_training_rounds_frac', type= float, default = 4/5)
parser.add_argument('--total_run_times', type=int, default=5)
parser.add_argument('--dp_delta', type = float, default = 1e-5)
parser.add_argument('--rdp_alpha', type = float, default = 2)
args = parser.parse_args()


if args.data =='filmtrust':
    args.user_batch = 874
elif args.data in [ 'ml-100k', 'ml-100k_online']:
    args.user_batch = 943
elif args.data == 'ml-1m':
    args.user_batch = 6040
elif args.data in [ 'ml-1m-605', 'ml-1m-605_online']:
    args.user_batch = 605
else:
    raise ValueError("no such datasets")


if args.allocation == "BGTplanner":
    main_bgt(args)
else:
    raise("no such allocation strategy {}".format(args.allocation))