"""Continue frozen-parent adapter trials with an explicit LR-policy factor."""
import math

from par_two_protein_fit import update as parent_update


INITIAL_UPDATES=1024
ADDITIONAL_UPDATES=3072
TOTAL_UPDATES=INITIAL_UPDATES+ADDITIONAL_UPDATES
ARMS={
    'beta_floor':('sequence_beta','floor'),
    'beta_restart':('sequence_beta','restart'),
    'uniform_floor':('sequence_uniform','floor'),
    'uniform_restart':('sequence_uniform','restart'),
}


def learning_rate(step,policy):
    if step<0 or step>=ADDITIONAL_UPDATES:raise ValueError('Outside fixed continuation budget')
    if policy=='floor':return 1e-5
    if policy=='restart':return 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*step/(ADDITIONAL_UPDATES-1)))
    raise ValueError('Declared LR policy required')


def global_step(local_step):
    if local_step<0 or local_step>=ADDITIONAL_UPDATES:raise ValueError('Outside fixed continuation budget')
    return INITIAL_UPDATES+local_step


def update(model,optimizer,cases,bank,arm,seed,step,protocol):
    parent,policy=ARMS[arm];lr=learning_rate(step,policy)
    # Equal endpoints make the unchanged updater use exactly our declared LR,
    # while its global step still advances data/noise/time keys beyond1024.
    settings=dict(lr=lr,min_lr=lr,updates=TOTAL_UPDATES,gradient_clip=protocol['gradient_clip'])
    row=parent_update(model,optimizer,cases,bank,parent,seed,global_step(step),settings)
    if row['lr']!=lr:raise RuntimeError('Unexpected LR inside frozen parent updater')
    row['global_update']=row['update'];row['update']=step+1
    row['lr_policy']=policy
    return row


def common_identity(row):
    return dict(global_update=row['global_update'],
        batch=[{k:v for k,v in event.items() if k!='time'} for event in row['batch']])
