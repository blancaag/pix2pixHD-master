import torch
from .ClassDataset import *

def set_dataloaders (opt): 
        
        datasets = {x: ClassDataset() for x in opt.phases} 
        
        for x in opt.phases: 
                datasets[x].initialize(opt, phase=x)

        [print("dataset [%s] was created with %d images" % (x, len(datasets[x]))) for x in opt.phases]
        
        dataloaders = {x: torch.utils.data.DataLoader(
                                                    datasets[x],
                                                    batch_size=opt.batch_size,
                                                    shuffle=True,
                                                    num_workers=int(opt.nworkers)) for x in opt.phases}
        
        return datasets, dataloaders