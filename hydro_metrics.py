import numpy as np
import properscoring as ps
import hydrostats as hs
import hydrostats.ens_metrics as hm
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from statsmodels.compat import lzip
import seaborn as sns
#####################################
#deterministic metrics
#KGE
def KGE(sim,obs):
    kge=hs.kge_2009(sim,obs)
    return kge
#NSE

def NSE(sim,obs):
    nse=hs.nse(sim,obs)
    return nse
# R2
def R2(sim,obs):
    r2=hs.r_squared(sim,obs)
    return r2
#Compute the root mean square error between the simulated and observed data.
def RMSE(sim,obs):
    rmse=hs.rmse(sim,obs)
    return rmse

#Compute mean square error error between simulation and observation
def MSE(sim,obs):

    mse=hs.mse(sim,obs)

    return mse

#Compute the mean error of the simulated and observed data.
def ME(sim,obs):
    me=hs.me(sim,obs)
    return me
#Compute the mean absolute error of the simulated and observed data.
def MAE(sim,obs):
    mae=hs.mae(sim,obs)
    return mae
def BIAS(sim,obs):

    bias=np.mean(sim-obs)

    return bias

from scipy.stats import pearsonr
#corr, _ = pearsonr(data1, data2)
def r_coe(sim,obs):
    # cov=np.cov(sim,obs)
    # cov=cov[0,1]
    # out=cov/(np.std(sim)*np.std(obs))
    out,_=pearsonr(sim,obs)
    
    return out
#####################################################################
#precipitation
def categorical_table(sim,obs,threshold):

    A=np.sum((sim>=threshold)&(obs>=threshold))
    B=np.sum((sim>=threshold)&(obs<threshold))
    C=np.sum((sim<threshold)&(obs>=threshold))
    D=np.sum((sim<threshold)&(obs<threshold))

    return A,B,C,D
class categorical_metrics:
    def __init__(self,A,B,C,D):
        self.A=A
        self.B=B
        self.C=C
        self.D=D
        self.n=A+B+C+D

    def percent_correct(self):

        out=(self.A+self.D)/self.n

        return out
    def prob_of_Detection(self):
        out=self.A/(self.A+self.C)

        return out
    def miss_rate(self):
        out=self.C/(self.A+self.C)

        return out

    def false_alarm_ratio(self):
        out=self.B/(self.A+self.B)

        return out
    def success_ratio(self):
        out=self.A/(self.A+self.B)

        return out
    def threat_score(self):
        #also known as Critical Success Index
        out=self.A/(self.A+self.B+self.C)

        return out
    def equitable_threat_score(self):
        CH=(self.A+self.B)*(self.A+self.C)/self.n

        out=(self.A-CH)/(self.A+self.B+self.C-CH)

        return out

    def true_skill_stat(self):
        out=(self.A*self.D-self.B*self.C)/((self.A+self.C)*(self.B+self.D))

        return out
    
    def heidke_skill_score(self):
        E=(self.A+self.B)*(self.A+self.C)+(self.B+self.D)*(self.C+self.D)/self.n

        out=(self.A+self.D-E)/(self.n+E)

        return out
    def bias_score(self):
        out=(self.A+self.B)/(self.A+self.C)

        return out
def cal_category_m(sim,obs,threshold):

    A,B,C,D=categorical_table(sim,obs,threshold)

    CM=categorical_metrics(A,B,C,D)

    pc=CM.percent_correct()
    pod=CM.prob_of_Detection()
    mr=CM.miss_rate()
    far=CM.false_alarm_ratio()
    sr=CM.success_ratio()
    ts=CM.threat_score()   #critical success index
    ets=CM.equitable_threat_score()
    tss=CM.true_skill_stat()
    hss=CM.heidke_skill_score()
    bs=CM.bias_score()

    d={'pc':[pc],
        'pod':[pod],
        'mr':[mr],
        'far':[far],
        'sr':[sr],
        'ts':[ts],
        'ets':[ets],
        'tss':[tss],
        'hss':[hss],
        'bs':[bs],
        }
    df=pd.DataFrame(d)
    df=df.round(2)

    return df


def dm(sim,obs):

    r=r_coe(sim,obs)

    rmse=RMSE(sim,obs)

    mae=MAE(sim,obs)

    me=ME(sim,obs)

    bias=BIAS(sim,obs)

    nse=NSE(sim,obs)

    kge=KGE(sim,obs)

    d={'r':[r],
       'rmse':[rmse],
       'mae':[mae],
       'me':[me],
       'bias':[bias],
       'nse':[nse],
       'kge':[kge]}
    df=pd.DataFrame(d)
    df=df.round(2)
    print('ok')
    return df
    
    
def df_list_save(df_list,writer,centrename):
    for n, df in enumerate(df_list):
        df.to_excel(writer, 'sheet_%s' % centrename[n])
    writer.save()
    print('***save complete***')

    
    
    
    

    

            
        
        
  
    
    
    
    

    




    
    
    

    

    


















