
import os 
from scipy.io import loadmat, savemat
import torch 
import argparse 
import numpy as np 
from scipy.stats import mode
from pathlib import Path 

# only for testing 
import matplotlib.pyplot as plt 

from configs.default_config import get_default_configs
from src import get_standard_score, get_standard_sde, wrapper_ddim, BaseSampler, LinearisedRecoFenics

# This hardcoded path will change 
path_to_model = {
    1: "/localdata/AlexanderDenker/KTC2023/diffusion_models/level_1/version_01/model_training.pt"
}


# regularisation parameters for initial reconstruction 
level_to_alphas = {
    1 : [[1956315.789, 0.,0.],[0., 656.842 , 0.],[0.,0.1,6.105],[1956315.789/3., 656.842/3,6.105/3.], [1e4, 0.1,5.]], 
    2 : [[1890000, 0.,0.],[0., 505.263, 0.],[0.,0.1,12.4210],[1890000/3., 505.263/3.,12.421/3.], [1e4, 0.1,5.]], 
    3 : [[1890000, 0.,0.],[0., 426.842, 0.],[0.,0.1,22.8421],[2143157/3., 426.842/3.,22.8421/3.], [6e5, 3,14]],
    4 : [[1890000, 0.,0.],[0., 1000., 0.],[0.,0.1,43.052],[1890000/3., 1000./3.,43.052/3.], [6e5, 8,16]], 
    5 : [[1890000, 0.,0.],[0., 843.6842, 0.],[0.,0.1,30.7368],[1890000/3., 843.684/3.,30.7368/3.], [6e5, 10,18]], 
    6 : [[40000, 0.,0.],[0., 895.789, 0.],[0.,0.1,74.947],[40000/3., 895.78/3.,74.947/3.], [6e5, 25,20]], 
    7 : [[40000, 0.,0.],[0., 682.105, 0.],[0.,0.1,18.421],[40000/3., 687.3684/3.,18.421/3.], [6e5, 30,22]], 
}



parser = argparse.ArgumentParser(description='reconstruction using conditional diffusion')
parser.add_argument('input_folder')
parser.add_argument('output_folder')
parser.add_argument('level')

def coordinator(args):
    level = int(args.level)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Input folder: ", args.input_folder)
    print("Output folder: ", args.output_folder)
    print("Level: ", args.level)

    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    ### load conditional diffusion model 
    config = get_default_configs()
        
    sde = get_standard_sde(config=config)
    score = get_standard_score(config=config, sde=sde, use_ema=False, load_model=False)

    score.load_state_dict(torch.load(path_to_model[level]))
    score.to(device)
    score.eval() 

    num_samples = 10 

    sampler = BaseSampler(
        score=score,
        sde=sde,
        predictor=wrapper_ddim, 
        corrector=None,
        init_chain_fn=None,
        sample_kwargs={
            'num_steps': 10, # 100
            'start_time_step': 0,
            'batch_size': num_samples,
            'im_shape': [1,256,256],
            'travel_length': 1, 
            'travel_repeat': 1, 
            'predictor': {'eta' : 0.9}
            },
        device=device)


    ### read files from args.input_folder 
    # there will be ref.mat in the input_folder, dont process this 
    f_list = [f for f in os.listdir(args.input_folder) if f != "ref.mat"]

    y_ref = loadmat(os.path.join(args.input_folder, "ref.mat"))
    Injref = y_ref["Injref"]
    Mpat = y_ref["Mpat"]
    Uelref = y_ref["Uelref"]

    mesh_name = "sparse"
    B = Mpat.T

    Nel = 32
    vincl_level = np.ones(((Nel - 1),76), dtype=bool) 
    rmind = np.arange(0,2 * (level - 1),1) #electrodes whose data is removed

    #remove measurements according to the difficulty level
    for ii in range(0,75):
        for jj in rmind:
            if Injref[jj,ii]:
                vincl_level[:,ii] = 0
            vincl_level[jj,:] = 0

    reconstructor = LinearisedRecoFenics(Uelref, B, vincl_level, mesh_name=mesh_name)

    alphas = level_to_alphas[level]
    for f in f_list: 

        y = np.array(loadmat(os.path.join(args.input_folder, f))["Uel"])

        ## get initial reconstruction 
        delta_sigma_list = reconstructor.reconstruct_list(y, alphas)

        delta_sigma_0 = reconstructor.interpolate_to_image(delta_sigma_list[0])
        delta_sigma_1 = reconstructor.interpolate_to_image(delta_sigma_list[1])
        delta_sigma_2 = reconstructor.interpolate_to_image(delta_sigma_list[2])
        delta_sigma_3 = reconstructor.interpolate_to_image(delta_sigma_list[3])
        delta_sigma_4 = reconstructor.interpolate_to_image(delta_sigma_list[4])

        sigma_reco = np.stack([delta_sigma_0, delta_sigma_1, delta_sigma_2, delta_sigma_3, delta_sigma_4])

        reco = torch.from_numpy(sigma_reco).float().to(device).unsqueeze(0)
        reco = torch.repeat_interleave(reco, repeats=num_samples, dim=0)
        
        x_mean = sampler.sample(reco, logging=False)

        x_round = torch.round(x_mean).cpu().numpy()[:,0,:,:]
        x_round[x_round > 2] = 2.

        u, _ = mode(x_round, keepdims=False)

        mdic = {"reconstruction": u.astype(int)}

        objectno = f.split(".")[0][-1]

        savemat( os.path.join(output_path, str(objectno) + ".mat"),mdic)
        

    ### save reconstructions to args.output_folder 
    # as a .mat file containing a 256x256 pixel array with the name {file_idx}.mat 
    # the pixel array must be named "reconstruction" and is only allowed to have 0, 1 or 2 as values.


if __name__ == '__main__':
    args = parser.parse_args()
    coordinator(args)