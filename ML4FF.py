# Import all packages needed

from datetime import timedelta
import numpy as np
import os
import pandas as pd
import pickle
import psutil
from hpsklearn import all_regressors
import hyperopt
from hyperopt import fmin,tpe,space_eval
from hyperopt.early_stop import no_progress_loss
from scipy.stats import bootstrap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
import time



# Define auxiliary functions to be used in the routine:
# CI_Scipy calculates the 95% confidence BCa bootstrap confidence interval of the mean of the input values "vals"
def CI_Scipy(vals):
    vals = (vals,)
    res = bootstrap(vals, np.mean, n_resamples=1000,confidence_level=0.95,random_state=123).confidence_interval
    return [res[0],np.mean(vals),res[1]]

# nse calculates the Nash–Sutcliffe efficiency coefficient for the predicted values "predictions" relative to the "targets".
def nse(predictions, targets):
    return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2)))

# rmse calculates the RMSE between the predicted values "a1" and the observed ones "a2".
def rmse(a1,a2):
    return (np.mean((a1-a2)**2))**0.5

# min_max_med_var calculates the minimum, maximum, median and variance of dataset "x".
def min_max_med_var(x):
    return [np.min(x),np.max(x),np.median(x),np.var(x)]

# kge calculates the Kling-Gupta efficiency coefficient for the predicted values "simulations" relative to the "evaluation".
def kge(simulations, evaluation):
    """Implementation taken from Hallouin (2021) - https://doi.org/10.5281/zenodo.2591217
    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = np.mean(simulations, axis=0, dtype=np.float64)
    obs_mean = np.mean(evaluation, dtype=np.float64)

    r_num = np.sum((simulations - sim_mean) * (evaluation - obs_mean),
                   axis=0, dtype=np.float64)
    r_den = np.sqrt(np.sum((simulations - sim_mean) ** 2,
                           axis=0, dtype=np.float64)
                    * np.sum((evaluation - obs_mean) ** 2,
                             dtype=np.float64))+10**(-10)
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(simulations, axis=0) / np.std(evaluation, dtype=np.float64)
    # calculate error in volume beta (bias of mean discharge)
    beta = (np.sum(simulations, axis=0, dtype=np.float64)
            / np.sum(evaluation, dtype=np.float64))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_


# Load dataset from repository. The path to the .csv file is "file_path". For simplicity, the csv is directly downloaded from the github repo.

file_path = "https://raw.githubusercontent.com/jaqueline-soares/ML4FF/main/data/data.csv"
ML4FF_dataset = pd.read_csv(file_path).set_index("data_hora")

# Shifting the dataset to predict the level of Conselheiro Paulino with a 2h lag. This is added as the "Outp" column of the ML4FF_dataset dataframe.
outpvals = ML4FF_dataset[["nivel_ConselheiroPaulino"]].shift(-8).dropna().to_numpy().flatten()
ML4FF_dataset["Outp"]=np.pad(outpvals, (0, 8), 'constant', constant_values=(4, np.nan))
ML4FF_dataset = ML4FF_dataset.dropna()

# Split the features and the outputs to be predicted.

features = ML4FF_dataset[ML4FF_dataset.columns[:-1]].to_numpy()
outputs = ML4FF_dataset[ML4FF_dataset.columns[-1]].to_numpy()

# Build the list "all_algs" with all the regressors in hpsklearn, specially their names, class and dictionaries with its input parameters and Bayesian search ranges:

def buil_dict(mdl_base):
    lst = mdl_base.named_args
    dictn={}
    for ll in lst:
        dictn[ll[0]]=ll[1]
    return dictn

all_algs = [[x.name,x,buil_dict(x)] for x in all_regressors("reg").inputs()]

#From the list "all_algs", build the reduced list containing only the methods of interest to ML4FF:

ML4FF_algorithms = ["sklearn_RandomForestRegressor","sklearn_BaggingRegressor","sklearn_GradientBoostingRegressor",
             "sklearn_LinearRegression","sklearn_BayesianRidge","sklearn_ARDRegression","sklearn_LassoLars",
             "sklearn_LassoLarsIC","sklearn_Lasso","sklearn_ElasticNet","sklearn_LassoCV",
             "sklearn_TransformedTargetRegressor","sklearn_ExtraTreeRegressor","sklearn_DecisionTreeRegressor",
             "sklearn_LinearSVR","sklearn_PLSRegression","sklearn_MLPRegressor","sklearn_DummyRegressor",
             "sklearn_TheilSenRegressor","sklearn_OrthogonalMatchingPursuitCV","sklearn_OrthogonalMatchingPursuit",
             "sklearn_RidgeCV","sklearn_Ridge","sklearn_SGDRegressor","sklearn_PoissonRegressor",
             "sklearn_ElasticNetCV",
             "sklearn_KNeighborsRegressor","sklearn_RadiusNeighborsRegressor",
             "sklearn_XGBRegressor","sklearn_GaussianProcessRegressor","sklearn_NuSVR","sklearn_LGBMRegressor"
            ]

all_candidates = [x for x in all_algs if x[0] in ML4FF_algorithms]


# The function RefineNCV_General generates two outputs, namely: resultados and resultados_stack.
# "resultados": is a list containing, in this order:
#  a) The model name
#  b) Identification of the dataset considered (either "CV_outer-X" where X is the outer folder number or "Holdout-" if holdout set)
#  c) Numpy array containing the differences between predictions and observed values for the dataset considered
#  d) Best hyperparameters found in the inner loop for the dataset considered
#  e) Mean Absolute Error between the predictions and observed values for the dataset considered

# "resultados_stack": is a list containing, in this order:
#  a) The model name
#  b) Identification of the dataset considered (either "CV_outer-X" where X is the outer folder number or "Holdout-" if holdout set)
#  c) Numpy array containing two sub-arrays: the predictions and the observed values for the dataset considered
#  d) Best hyperparameters found in the inner loop for the dataset considered
#  e) ETTrain (time in seconds to perform the Nested CV + train model on full Nested-CV dataset)
#  f) ETHoldout (time in seconds to perform the predictions on the holdout dataset)
#  *g) VMS memory difference in MB from the beginning to the end of the Nested CV + train model on full Nested-CV dataset prodecure
#  *h) RSS memory difference in MB from the beginning to the end of the Nested CV + train model on full Nested-CV dataset prodecure
#  *i) VMS memory difference in MB from the beginning to the end of the prediction process on the holdout dataset
#  *j) RSS memory difference in MB from the beginning to the end of the prediction process on the holdout dataset
# *items from g) to j) are not reliable, unless you running the code on a dedicated virtual machine.

# The inputs of the RefineNCV_General function are:
# a) "features": input features
# b) "outputs": output values (values to be predicted)
# c) "hold_out": percentage of values out of the holdout (in the case of the paper, 0.875)
# d) "random_state": random state to make the results reproducible
# e) "inner_folds": number of inner folds in the Nested-CV (in the case of the paper, 10)
# f) "outer_folds": number of outer folds in the Nested-CV (in the case of the paper, 30)
# g) "all_candidates": list of algorithims to be considered in the benchmark.
# h) "root": path where the pickled values of "resultados" and "resultados_stack" will be stored for each method. For simplicity, taken as root= "D:\\ML4FF"

def RefineNCV_General(features,outputs,hold_out,random_state,inner_folds,outer_folds,all_candidates,root):   
    try:
        os.mkdir(root)
    except:
        pass
    
    try:
        os.mkdir(os.path.join(root,"Models"))
    except:
        pass
    
    resultados=[]
    resultados_stack=[]
     
    features_nestedCV = features[:int(46072*hold_out)]
    outputs_nestedCV = outputs[:int(46072*hold_out)]
    features_holdout = features[int(46072*hold_out):]
    outputs_holdout = outputs[int(46072*hold_out):]

    X_A = features_nestedCV
    y_A = outputs_nestedCV

    cv_inner = TimeSeriesSplit(n_splits=inner_folds)
    cv_outer = TimeSeriesSplit(n_splits=outer_folds)   
    process = psutil.Process(os.getpid())
        
    def my_custom_loss_func(targets_s, predictions_s):
        nse = (1-(np.sum((targets_s-predictions_s)**2)/np.sum((targets_s-np.mean(targets_s))**2)))
        return 1/(2-nse)
    
    for mdl_info in all_candidates:
        mdl_name,mdl_type,space = mdl_info
        print(mdl_name)
        try:
            def inner_Search(X_Av,y_Av):
                mdl = hyperopt.pyll.stochastic.sample(mdl_type)
                def objective(params,X_train_i=X_Av,y_train_i=y_Av,tscv=cv_inner):
                    cachedir = mkdtemp()
                    pipeline = Pipeline([('transformer', MinMaxScaler()), ('estimator', mdl.set_params(**params))],memory=cachedir)
                    scoring = make_scorer(my_custom_loss_func, greater_is_better=True)
                    scr = -cross_val_score(pipeline, X_train_i, y_train_i, cv = tscv,n_jobs=-1,scoring=scoring).mean()
                    return scr

                best=fmin(fn=objective, 
                        space=space, 
                        algo=tpe.suggest, 
                        max_evals=100, 
                        early_stop_fn=no_progress_loss(10),
                        rstate=np.random.default_rng(random_state)
                      )
                best_par = space_eval(space, best)
                return mdl.set_params(**best_par)    

            pvi = 0
            start_time_outer = time.monotonic()
            start_vms_outer = process.memory_info().vms/(1024*1024)
            start_rss_outer = process.memory_info().rss/(1024*1024)
            for train_ix, test_ix in cv_outer.split(X_A):
                start_time = time.monotonic()
                start_vms = process.memory_info().vms/(1024*1024)
                start_rss = process.memory_info().rss/(1024*1024)
                stackinp=[]
                resulp=[]

                X_train, X_test = X_A[train_ix, :], X_A[test_ix, :]
                y_train, y_test = y_A[train_ix], y_A[test_ix]
                result = inner_Search(X_train, y_train)

                sc_X_A = MinMaxScaler()

                X_train_s = sc_X_A.fit_transform(X_train)
                X_test_s = sc_X_A.transform(X_test)

                result.fit(X_train_s, y_train.ravel())
                end_time = time.monotonic()
                end_vms = process.memory_info().vms/(1024*1024)
                end_rss = process.memory_info().rss/(1024*1024)
                yhat = result.predict(X_test_s)
                end_time_2 = time.monotonic()
                end_vms_2 = process.memory_info().vms/(1024*1024)
                end_rss_2 = process.memory_info().rss/(1024*1024)
                stackinp.append([yhat.flatten(),y_test.flatten()])
                real_vals_diff = yhat.flatten()-y_test.flatten()
                resulp.append(real_vals_diff)

                resultados.append([mdl_name,"CV_outer-"+str(pvi),np.array(resulp),result.get_params(),np.mean(np.abs(real_vals_diff))])
                resultados_stack.append([mdl_name,"CV_outer-"+str(pvi),np.array(stackinp),result.get_params(),timedelta(seconds=end_time - start_time),
                                           timedelta(seconds=end_time_2 - end_time),end_vms-start_vms,end_rss-start_rss,
                                         end_vms_2-end_vms,end_rss_2-end_rss])
                pvi=pvi+1

            bst_mdl_hyp = inner_Search(X_A,y_A)

            sc_X_A = MinMaxScaler()

            X_A_s = sc_X_A.fit_transform(X_A)

            bst_mdl = bst_mdl_hyp.fit(X_A_s,y_A.ravel())

            stackinp=[]
            resulp=[]
            end_time_outer = time.monotonic()
            end_vms_outer = process.memory_info().vms/(1024*1024)
            end_rss_outer = process.memory_info().rss/(1024*1024)
            self_evals = bst_mdl.predict(sc_X_A.transform(features_holdout))
            end_time_outer_2 = time.monotonic()
            end_vms_outer_2 = process.memory_info().vms/(1024*1024)
            end_rss_outer_2 = process.memory_info().rss/(1024*1024)
            stackinp.append([self_evals.flatten(),outputs_holdout])
            real_vals_diff = self_evals.flatten()-outputs_holdout
            resulp.append(real_vals_diff)

            resultados.append([mdl_name,"HoldOut-",np.array(resulp),bst_mdl.get_params(),np.mean(np.abs(real_vals_diff))])
            resultados_stack.append([mdl_name,"HoldOut-",np.array(stackinp),bst_mdl.get_params(),timedelta(seconds=end_time_outer - start_time_outer),
                                     timedelta(seconds=end_time_outer_2 - end_time_outer),end_vms_outer-start_vms_outer,end_rss_outer-start_rss_outer,
                                     end_vms_outer_2-end_vms_outer,end_rss_outer_2-end_rss_outer])   

            with open(root+"\\Models\\"+mdl_name+"-r", "wb") as f:
                pickle.dump([x for x in resultados if x[0]==mdl_name], f)
            with open(root+"\\Models\\"+mdl_name+"-rs", "wb") as f:
                pickle.dump([x for x in resultados_stack if x[0]==mdl_name], f)
        except:
            pass
    return [resultados,resultados_stack]

# Set the root to save the pickled lists.
root= "D:\\ML4FF"

# Run the Nested-CV using the same inputs as the paper ML4FF.
# run_ML4FF = RefineNCV_General(features,outputs,0.875,10,10,30,all_candidates,root)

# If the models have already been run, their pickled lists would be available. The pickled lists of the methods considered in the ML4FF paper are also in the github repo.
# The following auxiliary function is needed to gather the pickled data located in "path_v". Its inputs are:
# a) "all_candidates": list of all the models run
# b) "path_v": path of the pickled lists.

def buil_results_pkl(all_candidates,path_v):
    resu=[]
    resuls=[]
    for mdl_info in all_candidates:
        mdl_name,mdl_type,space = mdl_info
        try:
            with open(path_v+"\\"+mdl_name+"-r", "rb") as f:
                rpar = pickle.load(f)
            with open(path_v+"\\"+mdl_name+"-rs", "rb") as f:
                    rspar = pickle.load(f)
            for rr1 in rpar:
                resu.append(rr1)
            for rr2 in rspar:
                resuls.append(rr2)
        except:
            pass
    return [resu,resuls]

ML4FF_pickled = buil_results_pkl(all_candidates,root+"\\Models")

# Two new auxiliary functions can be defined to process the results and save them in a Excel spreadsheet for better visualization of the results.
# The firs function is perf_excel, which builds a self-explanatory excel spreadsheet with performance metrics. Its inputs are:
# a) "simu_runs": the results from the simulations (either run_ML4FF or ML4FF_pickled).
# b) "root": root path where to save the Excel spreadsheet.

def perf_excel(simu_runs,root):
    lst_perf=[]
    lst_comp_perf=[]
    methds = list(set([x[0] for x in simu_runs[1]]))
    for mth in methds:
        rrr = [x for x in simu_runs[1] if x[0]==mth]
        dfperf = pd.DataFrame()
        for itv in rrr:
            algo,part,_,_,tcv,tpred,mvms_cv,mrss_cv,mvms_pred,mrss_pred = itv
            dfperf[part]=[tcv.total_seconds(),tpred.total_seconds(),mvms_cv,mrss_cv,mvms_pred,mrss_pred]
            dfperf.index = ["Loop time","Predict time","VMS_loop","RSS_loop","VMS_pred","RSS_pred"]
            lst_perf.append([mth.split("_")[-1],dfperf])
        dfn = pd.DataFrame([dfperf["HoldOut-"].to_numpy()])
        dfn.columns = dfperf.index
        dfn.index = [mth.split("_")[-1]]
        lst_comp_perf.append(dfn)
    with pd.ExcelWriter(root+'\\Summary_Perf.xlsx') as writer:
        final_eval = pd.concat(lst_comp_perf)
        final_eval.to_excel(writer, sheet_name='Compilation')
        for dfi in lst_perf:
            if dfi[0] in list(final_eval.index):
                dfi[1].to_excel(writer, sheet_name=dfi[0])
            else:
                pass
            
perf_excel(ML4FF_pickled,root)

# Finally, the error metrics can be gathered and summarized in an Excel spreadsheet using the function build_excel. Its inputs are:
# a) "ML4FF_dataset": full dataset present in the github repo and used in the paper
# b) "ncvs": number of outer folds in the nested-CV (in the paper, 30)
# c) "hold_out": percentage of values out of the holdout (in the case of the paper, 0.875)
# d) "simu_runs": the results from the simulations (either run_ML4FF or ML4FF_pickled).

def build_excel(ML4FF_dataset,ncvs,hold_out,simu_runs):
    dfs=[]
    dfs2=[]
    dfs3=[]
    cols = ML4FF_dataset.index[int(46072*hold_out):]
    for mth in list(set([x[0] for x in simu_runs[1]])):
        cis_final = []
        rrr = [x for x in simu_runs[1] if x[0]==mth]
        final_comp=[]
        try:
            for x in rrr:
                if x[1][:2]=="CV":
                    a1,a2=x[2][0]
                    final_comp.append([x[0]+x[1],nse(a1, a2),rmse(a1,a2),1/(2-nse(a1, a2)),kge(a1, a2)])
                else:
                    pass
            if len(final_comp)==ncvs:
                cis_final.append(["CV-Best-"+mth,final_comp,CI_Scipy(np.array(final_comp)[:,1].astype(float)),
                                  CI_Scipy(np.array(final_comp)[:,2].astype(float)),
                                  CI_Scipy(np.array(final_comp)[:,3].astype(float)),
                                  CI_Scipy(np.array(final_comp)[:,4].astype(float)),
                        min_max_med_var(np.array(final_comp)[:,1].astype(float)),
                                  min_max_med_var(np.array(final_comp)[:,2].astype(float)),
                                  min_max_med_var(np.array(final_comp)[:,3].astype(float)),
                                 min_max_med_var(np.array(final_comp)[:,4].astype(float))])
            else:
                pass
        except:
            pass
        if len(cis_final)==0:
            pass
        else:
            outpvf = cis_final[0]

            line_ind_cv = outpvf[0].split("_")[-1]
            cv_evals = np.array(outpvf[1])
            ci_nse = outpvf[2]
            ci_rmse = outpvf[3]
            ci_nse_norm = outpvf[4]
            ci_kge = outpvf[5]
            mmm_nse = outpvf[6]
            mmm_rmse = outpvf[7]
            mmm_nse_norm = outpvf[8]
            mmm_kge = outpvf[9]
            df = pd.DataFrame(cv_evals)
            df.columns = ["Partition","nse","rmse","nse_norm","kge"]
            dfs.append([line_ind_cv,df])
            cis_final = []
            try:
                for x in rrr:
                    if x[1][:2]!="CV":
                        a1,a2=x[2][0]
                        cis_final.append(["Holdout-Best-"+mth,a1,a2,nse(a1, a2),rmse(a1,a2),1/(2-nse(a1, a2)),kge(a1, a2)])
                    else:
                        pass
            except:
                pass
            if len(cis_final)==0:
                pass
            else:
                outpvf = cis_final[0]
                line_ind = outpvf[0].split("_")[-1]
                pred_v = np.array(outpvf[1])
                ori_v = np.array(outpvf[2])
                nse_h = outpvf[3]
                rmse_h = outpvf[4]
                nse_norm_h = outpvf[5]
                kge_h = outpvf[6]
                df2 = pd.DataFrame([pred_v,ori_v])
                df2.columns = cols
                df2.index = [line_ind,"Holdout"]
                dfs2.append(df2)
                df3 = pd.DataFrame([[nse_h,rmse_h,nse_norm_h,kge_h]+mmm_nse+ci_nse+mmm_rmse+ci_rmse+mmm_nse_norm+ci_nse_norm+mmm_kge+ci_kge])
                df3.columns = ["NSE_Holdout","RMSE_Holdout","NSE_Norm_Holdout","KGE_Holdout"]+[x+"NSE" for x in ["Min","Max","Median","Var"]]+["CI-NSE"+x for x in ["Lower","Mean",
                              "Upper"]]+[x+"RMSE" for x in ["Min","Max","Median","Var"]]+["CI-RMSE"+x for x in ["Lower","Mean",
                              "Upper"]]+[x+"NSE_Norm" for x in ["Min","Max","Median","Var"]]+["CI-NSE_norm"+x for x in ["Lower",
                            "Mean","Upper"]]+[x+"KGE" for x in ["Min","Max","Median","Var"]]+["CI-KGE"+x for x in ["Lower","Mean","Upper"]]
                df3.index = [line_ind_cv]
                dfs3.append(df3)
        aaa,bbb,ccc = [dfs,dfs2,dfs3]
    with pd.ExcelWriter(root+'\\Summary.xlsx') as writer:
        final_eval = pd.concat(bbb)
        final_eval = final_eval[~final_eval.index.duplicated(keep='last')]
        final_eval.to_excel(writer, sheet_name='Predictions')
        final_stats = pd.concat(ccc).loc[final_eval.index[:-1]]
        final_stats.to_excel(writer, sheet_name='Statistics')
        for dfi in aaa:
            if dfi[0] in list(final_eval.index):
                dfi[1].to_excel(writer, sheet_name=dfi[0])
            else:
                pass
            
build_excel(ML4FF_dataset,30,0.875,ML4FF_pickled)