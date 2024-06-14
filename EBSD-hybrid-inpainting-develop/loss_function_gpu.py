# The goal of this is to code up the loss function for our partial convolution implementation.

import torch
import torchvision
from time import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#SI = torchvision.models.vgg16(pretrained=True)
#SI.eval()
#RETURN_NODES = {'features.4':'Pool1', 'features.9':'Pool2', 'features.16':'Pool3'}
#SI_P = torchvision.models.feature_extraction.create_feature_extractor(SI,return_nodes=RETURN_NODES).cuda(device=device)

def pconf_loss(ground_truth, input, output, initial_mask):

    I_comp = torch.mul(initial_mask, ground_truth) + torch.mul(torch.abs(1 - initial_mask), output)


    N_ight = ground_truth.size()[1] * ground_truth.size()[2] * ground_truth.size()[3]
    l_hole = 1/N_ight * torch.sum(torch.abs(torch.mul(1 - initial_mask, output - ground_truth)))
    l_valid = 1/N_ight * torch.sum(torch.abs(torch.mul(initial_mask, output - ground_truth)))


    #si_pgt = SI_P(ground_truth)
    #si_pout = SI_P(output)
    #si_icomp = SI_P(I_comp)


    l_perceptual = 0
    l_style_out = 0
    l_style_comp = 0

    #for p in 'Pool1', 'Pool2', 'Pool3':
        #N_si_pgt = si_pgt[p].size()[1] * si_pgt[p].size()[2] * si_pgt[p].size()[3]
        #C_p = si_pgt[p].size()[1]
        #si_pout_mod = si_pout[p].view(si_pout[p].size()[0], si_pout[p].size()[1], si_pout[p].size()[2] * si_pout[p].size()[3])
        #si_pgt_mod = si_pgt[p].view(si_pgt[p].size()[0], si_pgt[p].size()[1], si_pgt[p].size()[2] * si_pgt[p].size()[3])
        #si_icomp_mod = si_icomp[p].view(si_icomp[p].size()[0], si_icomp[p].size()[1], si_icomp[p].size()[2] * si_icomp[p].size()[3])

        #l_perceptual += (torch.sum(torch.abs((si_pout[p] - si_pgt[p]))) + torch.sum(torch.abs(si_icomp[p] - si_pgt[p]))) / N_si_pgt

        #l_style_out += 1/(C_p^2) * (torch.sum(torch.abs(1 / N_si_pgt * (torch.bmm(torch.transpose(si_pout_mod, 1, 2), si_pout_mod) - (torch.bmm(torch.transpose(si_pgt_mod, 1, 2), si_pgt_mod))))))

        #l_style_comp += 1/(C_p^2) * (torch.sum(torch.abs(1 / N_si_pgt * (torch.bmm(torch.transpose(si_icomp_mod, 1, 2), si_icomp_mod) - (torch.bmm(torch.transpose(si_pgt_mod, 1, 2), si_pgt_mod))))))


    l_l2 = 1/N_ight * torch.sqrt(torch.sum((output - ground_truth)**2))

    l_tv = (torch.sum(torch.abs(I_comp[:,:,:,1:] - I_comp[:,:,:,:-1])) + torch.sum(torch.abs(I_comp[:,:,1:,:] - I_comp[:,:,:-1,:]))) / N_ight

    return l_valid + l_hole + l_tv + 0*l_l2
                                                                                                   