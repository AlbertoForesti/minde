
import torch
from minde.libs.importance import get_normalizing_constant



def mi_cond_sigma(s_marg ,s_cond, x_t, g, mean ,std,sigma,importance_sampling):
    
   
    M = g.shape[0] 

    s_marg = s_marg.view(M,-1)
    s_cond = s_cond.view(M,-1)
    
    chi_t_x = mean**2 * sigma**2 + std**2
    ref_score_x = -(x_t)/chi_t_x
    ref_score_x = ref_score_x.view(M,-1)
    const = get_normalizing_constant((1,)).to(s_marg.device)
    if importance_sampling:
        e_marg = - const * 0.5 * ((s_marg - std* ref_score_x)**2).sum() / M
        e_cond = - const * 0.5 * ((s_cond - std* ref_score_x)**2).sum() / M
    else:
        try:
            e_marg = -0.5*(g**2* ((s_marg -  ref_score_x)**2) ).sum() / M
            e_cond = -0.5*(g**2* ((s_cond -  ref_score_x)**2 ) ).sum() / M
        except:
            raise ValueError(f"Shapes g={g.shape} s_marg={s_marg.shape} s_cond={s_cond.shape}, ref_score_x={ref_score_x.shape}, x_t={x_t.shape}, chi_t_x={chi_t_x.shape}")
    return (e_marg - e_cond).item()



def mi_cond(s_marg ,s_cond, g, importance_sampling):

    M = g.shape[0] 
    const = get_normalizing_constant((1,)).to(s_marg.device)

    s_marg = s_marg.view(M,-1)
    s_cond = s_cond.view(M,-1)

    # raise UserWarning(f"Some shapes: s_marg={s_marg.shape}, s_cond={s_cond.shape}, g={g.shape}")
    
    if importance_sampling:
        mi = const *0.5* ((s_marg - s_cond  )**2).sum()/ M
    else:
        try:
            mi = 0.5* (g**2*(s_marg - s_cond )**2).sum()/ M
        except:
            raise ValueError(f"Shapes g={g.shape} s_marg={s_marg.shape} s_cond={s_cond.shape}")
        
    return mi.item()



def mi_joint( s_joint ,s_cond_x,s_cond_y, g ,importance_sampling):
  
    
    M = g.shape[0] 
    s_cond = torch.cat([s_cond_x,s_cond_y],dim=1)
    s_cond = s_cond.view(M,-1)
    s_joint = s_joint.view(M,-1)

    if importance_sampling:
        const = get_normalizing_constant((1,)).to(g.device)
        mi = const *0.5* ((s_joint - s_cond  )**2).sum()/ M
    else:
        mi = 0.5 * (g**2*(s_joint - s_cond )**2).sum()/ M
    return mi.item()


def mi_joint_sigma( s_joint ,s_cond_x,s_cond_y,x_t,y_t,mean,std, g ,sigma,importance_sampling):
    
    M = g.shape[0] 

    s_cond_x = s_cond_x.view(M,-1)
    s_cond_y = s_cond_y.view(M,-1)
    s_joint = s_joint.view(M,-1)
    
    chi_t = mean**2 * sigma**2 + std**2
    ref_score_x = -(x_t)/chi_t 
    ref_score_y = -(y_t)/chi_t 

    ref_score_x = ref_score_x.view(M,-1)
    ref_score_y = ref_score_y.view(M,-1)

    ref_score_xy =  torch.cat([ref_score_x,ref_score_y],dim=1)

    ## the same as ref_score_xy = - torch.cat([x_t,y_t],dim=1)/chi_t
    if importance_sampling:
        const = get_normalizing_constant((1,)).to(g.device)
        e_joint = -const *0.5* ((s_joint - std * ref_score_xy  )**2).sum()/ M
        e_cond_x = -const *0.5* ((s_cond_x - std * ref_score_x  )**2).sum()/ M
        e_cond_y = -const *0.5* ((s_cond_y - std * ref_score_y  )**2).sum()/ M
        
    else:
        e_joint = -0.5 * (g**2*(s_joint -  ref_score_xy  )**2).sum()/ M
        e_cond_x = -0.5 * (g**2*(s_cond_x -  ref_score_x  )**2).sum()/ M
        e_cond_y = -0.5 * (g**2*(s_cond_y -  ref_score_y  )**2).sum()/ M
        
    return (e_joint - e_cond_x - e_cond_y  ).item()



