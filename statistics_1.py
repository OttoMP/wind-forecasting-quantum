import tensorflow as tf
import numpy as np
import pandas as pd

from scipy import stats  
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from tensorflow.keras import backend

def factor_of_2(y_true, y_pred):
    min_ = 0.5
    max_ = 2.0

    tensor_true = tf.constant(y_true)
    tensor_true = tf.cast(tensor_true, tf.float32)
    tensor_pred = tf.constant(y_pred)
    tensor_pred = tf.cast(tensor_pred, tf.float32)

    division = tf.divide(tensor_pred, tensor_true)

    greater_min = tf.greater_equal(division, min_)
    less_max = tf.less_equal(division, max_)

    res = tf.equal(greater_min, less_max)
    res = tf.cast(res, tf.float32)

    return backend.get_value(tf.reduce_mean(res))

def allmetrics(original,predito):
    r_value = 0
    slope, intercept, r_value, p_value, std_err = stats.linregress(original, predito)
    mse = mean_squared_error(original, predito)
    mae = mean_absolute_error(original, predito)
    rr = r2_score(original,predito)
    pea = stats.pearsonr(original, predito)
    fat = factor_of_2(original,predito)
    nmse = mse/stats.tvar(original)
    rmse = np.sqrt(mse)
    nrmse = rmse/stats.tstd(original)
    return mae,mse,nmse,r_value,rr,fat,rmse,nrmse

def get_error_interval(model, X_val, y_val, y_test_pred, p_value):
    y_val_pred = model.predict(X_val)
    y_val_error = np.abs(y_val - y_val_pred)
    
    error_quantile=np.ndarray((1,y_val.shape[1]))
    for i in range(y_val.shape[1]):
        error_quantile[0,i] = np.quantile(y_val_error[:,i], q=p_value, interpolation='higher')
        
    y_test_interval_pred_left=np.ndarray(y_test_pred.shape)
    y_test_interval_pred_right=np.ndarray(y_test_pred.shape)
    
    for i in range(y_test_pred.shape[1]):
        y_test_interval_pred_left[:,i] = y_test_pred[:,i] - error_quantile[0,i]
        y_test_interval_pred_right[:,i] = y_test_pred[:,i] + error_quantile[0,i]
    
    return error_quantile, y_test_interval_pred_left, y_test_interval_pred_right

def get_mean_left_right_error_interval(model, scaler, X_val, y_val, y_test, y_test_pred):
    error, error_left, error_right = get_error_interval(model, X_val, y_val, y_test_pred, 0.95)
    
    #error_left_normal = scaler.inverse_transform(error_left)
    #error_right_normal = scaler.inverse_transform(error_right)
    error_left_normal  = error_left
    error_right_normal = error_right

    mean_error_normal      = np.ndarray((1,y_test.shape[1]))
    mean_error_left_normal = np.ndarray((1,y_test.shape[1]))
    mean_error_right_normal= np.ndarray((1,y_test.shape[1]))
    mean_predictions       = np.ndarray((1,y_test_pred.shape[1]))

    for i in range(y_test.shape[1]):
        mean_error_left_normal[0,i]  = np.mean(error_left_normal[:,i])
        mean_error_right_normal[0,i] = np.mean(error_right_normal[:,i])
        mean_predictions[0,i]        = np.mean(y_test_pred[:,i])

    mean_error_normal = (mean_error_right_normal - mean_error_left_normal)/2
    
    return mean_predictions, mean_error_normal, mean_error_left_normal, mean_error_right_normal

def quantitative_analysis(y_test, y_test_pred):
    valores = []
    for i in range(len(y_test_pred)):
        mae,mse,nmse,r_value,rr,fat,rmse,nrmse = allmetrics(y_test[:,0],y_test_pred[i][:,0])
        #valores.append([str(i+1)+" depth",mae,mse,nmse,rmse,nrmse,r_value,rr,fat,mean_error_normal[0,i],mean_error_left_normal[0,i],mean_predictions[0,i],mean_error_right_normal[0,i]])
        valores.append([str(i+1)+" depth",mae,mse,nmse,rmse,nrmse,r_value,rr,fat])
        print("MAE:",mae)
        print("MSE:",mse)
        print("NMSE:",nmse)
        print("RMSE:",rmse)
        print("NRMSE:",nrmse)
        print("R:",r_value)
        print("R²:",rr)
        print("Fator de 2:",fat)

    erros = pd.DataFrame(valores)
    erros.columns = ['Index','MAE','MSE','NMSE','RMSE','NRMSE','R','R²','Fator de 2']#, 'error interval (+/-)', 'left limit', 'mean', 'right limit']
    erros = erros.set_index('Index')
    erros.loc['Média'] = erros.mean()
    return erros