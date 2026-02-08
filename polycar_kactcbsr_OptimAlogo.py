import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mealpy.bio_based.BBO import OriginalBBO
from mealpy.bio_based import BBO
from mealpy.utils.visualize import *
from mealpy.evolutionary_based import DE
from mealpy import  TLO, FFA, SSO, HS, PSO, WOA, GWO, ALO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mealpy.swarm_based import AGTO
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from mealpy.bio_based import BBO, EOA, IWO, SBO, SMA, TPO, VCS, WHO
from mealpy.evolutionary_based import CRO, DE, EP, ES, FPA, GA, MA
from mealpy.human_based import BRO, BSO, CA, CHIO, FBIO, GSKA, ICA, LCO, QSA, SARO, SSDO, TLO
from mealpy.math_based import AOA, CEM, CGO, GBO, HC, PSS, SCA
from mealpy.music_based import HS
from mealpy.physics_based import ArchOA, ASO, EFO, EO, HGSO, MVO, NRO, TWO, WDO
from mealpy.system_based import AEO, GCO, WCA
from mealpy.swarm_based import ABC, ACOR, ALO, AO, BA, BeesA, BES, BFO, BSA, COA, CSA, CSO, DO, EHO, FA, FFA, FOA, GOA, GWO, HGS
from mealpy.swarm_based import HHO, JA, MFO, MRFO, MSA, NMRA, PFA, PSO, SFO, SHO, SLO, SRSR, SSA, SSO, SSpiderA, SSpiderO, WOA
#from mealpy.utils.space import FloatVar
#defintion of fitness function
def fitness_function(solution):  
    s1=pd.DataFrame(np.asarray(solution[0].reshape(-1,1)))
    s2=pd.DataFrame(np.asarray(solution[1].reshape(-1,1)))
    s3=pd.DataFrame(np.asarray(solution[2].reshape(-1,1)))
    s4=pd.DataFrame(np.asarray(solution[3].reshape(-1,1)))
    
    #print("solution[0]=",solution[0])
    """
    s1=pd.DataFrame(np.asarray(solution[0]))
    s2=pd.DataFrame(np.asarray(solution[1]))
    s3=pd.DataFrame(np.asarray(solution[2]))
    s4=pd.DataFrame(np.asarray(solution[3]))
    """
    ds1=pd.concat([s1,s2,s3,s4], axis=1, join='inner')
       
    ds1.columns = range(ds1.shape[1])  # assign temporary unique column names
    
    # Now assign correct feature names
    if len(fea_name) != ds1.shape[1]:
        raise ValueError(f"Length mismatch: {len(fea_name)} feature names vs {ds1.shape[1]} columns in data")
    
    ds1.columns = fea_name  # assign correct feature names
    ds = ds1.copy()
    
    y1 = mdl1.predict(ds)
    y2 = mdl2.predict(ds)
    y3 = mdl3.predict(ds)
    y4 = mdl4.predict(ds)
    return [y1[0], y2[0], y3[0], y4[0]]
    #return [y1[0], y2[0],y3[0], y4[0]]
#normalaised values conversion to multi to single
def normalize_objective_values(df, maximize): 
    normalized_df = df.copy()
    for column in df.columns:
        if maximize==True:
            max_val = df[column].max()
            min_val = df[column].min()
            #normalized_df[column] = df[column] / max_val
            normalized_df[column] = (max_val-df[column]) / (max_val - min_val)
        else:
            min_val = df[column].min()
            max_val = df[column].max()
            normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
    return normalized_df
#non-dominated sorted solution
def dominates(solution1, solution2):
    """
    Check if solution1 dominates solution2.
    """
    return all(sol1 <= sol2 for sol1, sol2 in zip(solution1, solution2)) and any(sol1 < sol2 for sol1, sol2 in zip(solution1, solution2))

def non_dominated_sort(solutions):
    """
    Perform non-dominated sorting on the list of solutions.
    """
    pareto_fronts = []
    dominated_count = {i: 0 for i in range(len(solutions))}
    dominating_solutions = {i: [] for i in range(len(solutions))}
    
    for i, solution1 in enumerate(solutions):
        for j, solution2 in enumerate(solutions):
            if i != j:
                if dominates(solution1, solution2):
                    dominating_solutions[i].append(j)
                elif dominates(solution2, solution1):
                    dominated_count[i] += 1
    
    current_front = [i for i, count in dominated_count.items() if count == 0]
    while current_front:
        pareto_fronts.append(current_front)
        next_front = []
        for solution in current_front:
            for dominated_solution in dominating_solutions[solution]:
                dominated_count[dominated_solution] -= 1
                if dominated_count[dominated_solution] == 0:
                    next_front.append(dominated_solution)
        current_front = next_front
    
    sorted_solutions = [solution for front in pareto_fronts for solution in front]
    return sorted_solutions, pareto_fronts
def get_pareto_front_solutions(solutions):
    """
    Retrieve solutions from the first Pareto front.
    """
    sorted_solutions, pareto_fronts = non_dominated_sort(solutions)
    return [solutions[index] for index in pareto_fronts[0]], pareto_fronts[0]
def remove_duplicates_from_pareto_front_with_indices(solutions, indices):
    """
    Remove duplicate solutions from the non-dominated sorted solutions and return their indices.
    """
    unique_solutions = []
    unique_indices = []
    seen = []
    for solution, index in zip(solutions, indices):
        if solution not in seen:
            unique_solutions.append(solution)
            unique_indices.append(index)
            seen.append(solution)
    return unique_indices, unique_solutions
#dengs function
def dengs(wt, ot, obj):
    nobj = obj / np.sqrt(np.sum(obj*obj, axis =0))
    nobjw = nobj * wt
    pis=[]
    nis=[]
    for i in range(len(wt)):
        if ot[i] == 1:
            pis.append(np.min(obj.iloc[:,i]))
            nis.append(np.max(obj.iloc[:,i]))
        if ot[i] == 2:
            pis.append(np.max(obj.iloc[:,i]))
            nis.append(np.min(obj.iloc[:,i]))
    pis = pd.Series(pis)   
    nis = pd.Series(nis)

    cp=[]
    cm=[]
    sp=[]
    sm=[]
    c=[]
    for i in range(obj.shape[0]):
        cp.append(np.sum(nobjw.iloc[i,:]*pis,axis=0)/np.sqrt((np.sum(obj.iloc[i,:]*nobjw.iloc[i,:], axis = 0)*np.sum(pis*pis,axis=0))))
        cm.append(np.sum(nobjw.iloc[i,:]*nis,axis=0)/np.sqrt((np.sum(obj.iloc[i,:]*nobjw.iloc[i,:], axis = 0)*np.sum(nis*nis,axis=0)))) 
    cp = pd.Series(cp)
    cm = pd.Series(cm)
    for i in range(obj.shape[0]):
        sp.append((cp.iloc[i]*np.sqrt(np.sum(nobjw.iloc[i,:]*nobjw.iloc[i,:], axis =0)))/(np.sqrt(np.sum(pis*pis,axis=0))))
        sm.append((cm.iloc[i]*np.sqrt(np.sum(nobjw.iloc[i,:]*nobjw.iloc[i,:], axis =0)))/(np.sqrt(np.sum(nis*nis,axis=0))))
    sp = pd.Series(sp)
    sm = pd.Series(sm)
    #c=pd.Series(sp/(sp+sm)).sort_values(ascending = False)
    c=pd.Series(sp/(sp+sm))
    print(c)
    return c   
#problem defintion
lower_bounds = [250, 300, 200, 1.5]
upper_bounds = [350, 500, 400, 2.5]
# Create FloatVar objects for each variable
#bounds = [FloatVar(lb=lb, ub=ub) for lb, ub in zip(lower_bounds, upper_bounds)]

problem_dict1 = {
"fit_func": fitness_function,
"lb": lower_bounds,
"ub": upper_bounds,
#"bounds": FloatVar(lb=(225, 250, 10), ub=(375, 400, 40)),
"minmax": "min",
"obj_weights": [0.25, 0.25, 0.25, 0.25],
"save_population": True,
"log_file": "result.log",
}
#data set
ds_fea0=pd.read_csv('d:/msk/lenin/polycarbonate/mldata_ka2_m.csv')
ds_fea1=ds_fea0.iloc[:,0:4]
ds_tar_end1=ds_fea0.iloc[:,4]#4 for end 5 for exd and 6 for erd 7 for tr
xtrain1, xtest1, ytrain1, ytest1 = train_test_split(ds_fea1, ds_tar_end1, test_size = 0.25, random_state = 2)
#mdl1 = RandomForestRegressor(n_estimators=469, max_depth=16, min_samples_split=3, min_samples_leaf=7)
mdl1 = RandomForestRegressor(n_estimators=469, max_depth=16)
mdl1.fit(xtrain1,ytrain1)

ds_fea00=pd.read_csv('d:/msk/lenin/polycarbonate/mldata_ct_m.csv')
ds_fea1=ds_fea0.iloc[:,0:4]
ds_tar_end1=ds_fea0.iloc[:,4]#4 for end 5 for exd and 6 for erd 7 for tr
xtrain1, xtest1, ytrain1, ytest1 = train_test_split(ds_fea1, ds_tar_end1, test_size = 0.25, random_state = 2)
#mdl1 = RandomForestRegressor(n_estimators=469, max_depth=16, min_samples_split=3, min_samples_leaf=7)
mdl2 = RandomForestRegressor(max_depth=12, min_samples_leaf=3, min_samples_split=7,n_estimators=318)
mdl2.fit(xtrain1,ytrain1)


#data set
ds_fea0=pd.read_csv('d:/msk/lenin/polycarbonate/mldata_cb.csv')
ds_fea1=ds_fea0.iloc[:,0:4]
ds_tar_end1=ds_fea0.iloc[:,4]#4 for end 5 for exd and 6 for erd 7 for tr
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(ds_fea1, ds_tar_end1, test_size = 0.25, random_state = 2)
#mdl1 = RandomForestRegressor(n_estimators=469, max_depth=16, min_samples_split=3, min_samples_leaf=7)
mdl3 = RandomForestRegressor(max_depth=14, min_samples_leaf=2, min_samples_split=6,n_estimators=245)
mdl3.fit(xtrain2,ytrain2)

#data set
ds_fea0=pd.read_csv('d:/msk/lenin/polycarbonate/mldata_sr1.csv')
ds_fea1=ds_fea0.iloc[:,0:4]
ds_tar_end1=ds_fea0.iloc[:,4]#4 for end 5 for exd and 6 for erd 7 for tr
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(ds_fea1, ds_tar_end1, test_size = 0.25, random_state = 2)
#mdl1 = RandomForestRegressor(n_estimators=469, max_depth=16, min_samples_split=3, min_samples_leaf=7)
mdl4 = XGBRegressor(colsample_bytree=np.float64(0.8059264473611898),learning_rate=np.float64(0.30214464853521816),gamma=np.float64(0.06974693032602092), n_estimators=100, max_depth=9)
mdl4.fit(xtrain2,ytrain2)
fea_name=xtrain2.columns.tolist()

epoch = 100
p_size = 20
p_m = 0.01
elites = 2
ntrails=1
rno=11 #run number
x=np.asarray(list(range(epoch)))
npa=4 #numnber of paramerters
nr=4 #number of responses
#wt = [0.25, 0.25, 0.25, 0.25]
#ot =[1, 1, 1, 1]
wt = [0.25, 0.25, 0.25, 0.25]
ot =[1, 1, 1, 1]
# List of algorithms
algorithms = [
    #('MVO', MVO.OriginalMVO),
    ('BBO',BBO.OriginalBBO),
    #('FBIO',FBIO.OriginalFBIO),
    ('PSO', PSO.OriginalPSO),
    ('SSO', SSO.OriginalSSO),
    ('TWO', TWO.OriginalTWO),
    #('EHO', EHO.OriginalEHO),
    #('SCA', SCA.OriginalSCA)
    ]
rnol=[7, 77, 78, 79, 80]
for rno in rnol:
    
    #no. of axis is based on number of responses
    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    f3, ax3 = plt.subplots()
    f4, ax4 = plt.subplots()
 #based on number of responses increase ax
    fc =['Red', 'm', 'g', 'c', 'b']#based on number of algorithms k and b
    k1=0
    k2=0
    k3=0
    k4=0
    #model1 = GWO.OriginalGWO(epoch=epoch, pop_size=p_size)
    mlist=[]
    rt=[]
    itno=[]
    obst=[]
    conv_value=[]
    maximize = [False, False]
    fname2='d:/msk/lenin/polycarbonate/polycar_kactcbsr_' + str(rno) + '_rt_itno_convalue.xlsx'
    fname21='d:/msk/lenin/polycarbonate/polycar_kactcbsr_' + str(rno) + '_conv_value.xlsx'
    for mname, model1 in algorithms:
        #fname='d:/msk/lenin/vbalaji/' + mname + '_opf1.xlsx'
        fname='d:/msk/lenin/polycarbonate/polycar_kactcbsr_' + str(rno) + mname + '_opf1.xlsx'
        fname1='d:/msk/lenin/polycarbonate/polycar_kactcbsr_' + str(rno) + mname + '_parsal.xlsx'
        fname3='d:/msk/lenin/polycarbonate/polycar_kactcbsr_' + str(rno) + mname + '_parsal_pareto_front.xlsx'
        model1=model1(epoch=epoch, pop_size=p_size)
        mlist.append(mname)
        bpf=[]
    
        for tr in range(ntrails):
            """
            fn= 'd:/msk/lenin/vbalaji/ML 4by4/rslt_dertr_' + str(tr) + '.xlsx'
            fnt= 'd:/msk/lenin/vbalaji/ML 4by4/rslt__dertr_' + str(tr) + '.xlsx'
            fnf= 'd:/msk/lenin/vbalaji/ML 4by4/rslt__dertr_' + str(tr) + '.jpeg'
            plt.figure(tr) 
            """
            model1.solve(problem_dict1)
    
            best_position1=model1.history.list_global_best
            best_fitness1=model1.history.list_global_best_fit
            popu=model1.history.list_population
            
    
            bpn1=best_position1[0]
            res1=bpn1[1][1]
            #change bpn1 and res1 based on number of parameters and responses
            bp=[bpn1[0][0],bpn1[0][1],bpn1[0][2],bpn1[0][3], res1[0],res1[1],res1[2],res1[3]]#change bpn1 based on no. of parameter and res1 based on no. of responses
            bpf.append(bp)
            print(bp)
            y1=model1.history.list_global_best_fit
            t1=model1.history.list_epoch_time
            t1_ar = sum(np.asarray(t1))    
            
            rt.append(t1_ar)
            """
            y1_ar = np.asarray(y1)
            y1_df = pd.DataFrame({'BBOriginal': y1_ar})
            res_list = []
            for i in range(0, len(y1_ar)):
                if y1_ar[i] == y1_ar.min():
                    res_list.append(i)
            itno1=res_list[0]
            itno.append(itno1)
            """
        
        opf=[]
        opf_res=[]
        for i in list(range(epoch)):
            for j in list(range(p_size)):
                a=popu[i][j]
                b=a[1]
                #change a and b value based on no. of parameters and responses
                opf.append([a[0][0],a[0][1],a[0][2],a[0][3],b[1][0],b[1][1],b[1][2],b[1][3]])#update a value based on no. of parameter and b based on no. of responses
                opf_res.append([b[1][0],b[1][1],b[1][2],b[1][3]])#update responses
        opf_df=pd.DataFrame(opf) 
        opf_df.to_excel(fname)
        #dengs method
        obj_r=pd.DataFrame(opf_res)
        cv_1=normalize_objective_values(obj_r, maximize)
        cv1=cv_1.sum(axis=1)
        #cv1=dengs(wt,ot,u_i.iloc[:,npa:])
        sorted_values = cv1.sort_values(ascending=True).reset_index(drop=True)
        sorted_ind=sorted_values.index[0]
        obst.append(opf_df.iloc[sorted_ind,:])
        
        k=0
        parsol=[]
        for i in list(range(epoch)):
            j = (i+1)*p_size
            para=opf_df.iloc[k:j,:]
            obj=obj_r.iloc[k:j,:]
            cv_1=normalize_objective_values(obj, maximize)
            cv1=cv_1.sum(axis=1)
            #cv1=dengs(wt,ot,obj)
            #sorted_ind = sorted(range(len(cv1)), key=lambda x: cv1[x])
            sorted_values = cv1.sort_values(ascending=True).reset_index(drop=True)
            sorted_ind=sorted_values.index[0]
            p1=para.iloc[sorted_ind,:]
            parsol.append(p1)
            k=j
        
        df_parsol1=pd.DataFrame(parsol)
        df_parsol1.to_excel(fname1)
        df_parsol=df_parsol1.iloc[:,npa:]
        """
        #non-dominated solutions
        pareto_n, pareto_frno=get_pareto_front_solutions(opf_res)
        #p_n, p_frno=get_pareto_front_solutions(opf_res)
        #pareto_n, pareto_frno=remove_duplicates_from_pareto_front_with_indices(p_n, p_frno)
        print(pareto_n,pareto_frno)
        pareto_sol=[]
        for i in pareto_frno:
            pareto_sol.append(opf[i])
        df_pareto_sol=pd.DataFrame(pareto_sol)
        df_pareto_sol.to_excel(fname3)
        #to plot convergence of responses
        df_parsol=df_pareto_sol.iloc[:,npa-1:]
        """
        for i in list(range(nr)): #number of responses is 4
            rv=[]
            for j in list(range(epoch)):
                if j==0:
                    rv.append(df_parsol.iloc[j,i])
                else:
                    if ot[i]==1:
                        if df_parsol.iloc[j,i]<=rv[j-1]:
                            rv.append(df_parsol.iloc[j,i])
                        else:
                            rv.append(rv[j-1])
                    if ot[i]==2:
                        if df_parsol.iloc[j,i]>=rv[j-1]:
                            rv.append(df_parsol.iloc[j,i])
                        else:
                            rv.append(rv[j-1])
            srt_ind=sorted(range(len(rv)), key=lambda x: rv[x])#identify the convergence iteration number
            itno.append(srt_ind[0])
            conv_value.append(rv[srt_ind[0]])
            #based on number of responses increase i values
            if i==0:
                ax1.plot(x, rv, label = mlist, linestyle ='-', color = fc[k1])
                ax1.set_xlabel('Epoch No.')
                ax1.set_ylabel('KA')
                ax1.legend(mlist)
                k1=k1+1
            if i==1:
                ax2.plot(x, rv, label = mlist, linestyle ='-', color = fc[k2])
                ax2.set_xlabel('Epoch No.')
                ax2.set_ylabel('CT')
                ax2.legend(mlist)
                k2=k2+1
            #based on number of response enable i values
            if i==2:
                ax3.plot(x, rv, label = mlist, linestyle ='-', color = fc[k3])
                ax3.set_xlabel('Epoch No.')
                ax3.set_ylabel('CB')
                ax3.legend(mlist)
                k3=k3+1
            #based on number of response enable i values 
            if i==3:
                ax4.plot(x, rv, label = mlist, linestyle ='-', color = fc[k3])
                ax4.set_xlabel('Epoch No.')
                ax4.set_ylabel('SR')
                ax4.legend(mlist)
                k4=k4+1
    # to save the convergence plot into a file
    fign1='d:/msk/lenin/polycarbonate/polycar_kactcbsr_' + str(rno) + '_ConvPlot1.jpg'
    fign2='d:/msk/lenin/polycarbonate/polycar_kactcbsr_' + str(rno) + '_ConvPlot2.jpg'
    fign3='d:/msk/lenin/polycarbonate/polycar_kactcbsr_' + str(rno) + '_ConvPlot3.jpg'
    fign4='d:/msk/lenin/polycarbonate/polycar_kactcbsr_' + str(rno) + '_ConvPlot4.jpg'    

    f1.savefig(fign1)   
    f2.savefig(fign2)  
    f3.savefig(fign3)  
    f4.savefig(fign4)  
      
    #to write the computation time, convergence iteration number into a file fname2
    rt=np.asarray(rt).reshape(1,-1)
    rt=pd.DataFrame(rt)
    itno=np.asarray(itno).reshape(1,-1)
    itno=pd.DataFrame(itno)
    conv_value=np.asarray(conv_value).reshape(1,-1)
    conv_value=pd.DataFrame(conv_value)
    rt_itno=pd.concat([rt,itno,conv_value],axis=1,join='inner')
    rt_itno.to_excel(fname2)  
    conv_value.to_excel(fname21)          
 