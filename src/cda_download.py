from cdasws import CdasWs
import pandas as pd
import numpy as np

class sc_cdaWeb():
    '''to obtain magnetic field & plasma data from cdaweb'''
    
    def __init__(self, name, tRange,*args):
        self.name = name
        self.time_period = tRange
        self.raw = {} # raw data, stored in dict
        self.data=pd.DataFrame() # data after interpolation
#         if args !=():
#             self.add_instrument(args)       
    
    def findData(self,sc_ins, interval=[]):
        instrumt = sc_ins[self.name]
#         raw_data = {}
        
        cdas = CdasWs()
        # i: instrument data directory name, e.g., WI_MAG
        # j: instrument data subdirectory name, e.g., WI_MAG_BT, WI_MAG_BN
        for i in instrumt.columns:
            tempData = pd.DataFrame()
            for j in instrumt[i].dropna():
                
                if interval:
                    temp_data = cdas.get_data(i, [j], self.time_period[0],self.time_period[1],
                                          binData={'interpolateMissingValues': True, 'interval':interval[0]})[1]
                else:
                    temp_data = cdas.get_data(i, [j], self.time_period[0],self.time_period[1],
                                          binData={'interpolateMissingValues': False})[1]
                

                # Deal with the naming of different S/C is not uniform
                # Wind instruments' name format change
                if self.name == 'Wind':
                    j = j.replace('.','$')
    
                if len(temp_data[j].shape) == 1:
                    tempData[j] = temp_data[j]
                elif len(temp_data[j].shape) == 2 and temp_data[j].shape[1]==3:
                    tempData[j+'_x'] = np.array(temp_data[j])[:,0]
                    tempData[j+'_y'] = np.array(temp_data[j])[:,1]
                    tempData[j+'_z'] = np.array(temp_data[j])[:,2]
                elif len(temp_data[j].shape) == 2 and self.name == 'Ulysses' and j == 'Density':
                    tempData[j] = np.array(temp_data[j])[:,0]
                else:
                    tempData[j]=np.zeros(temp_data[j].shape[0])
                    print('Failed to get %s_%s'%(i,j))
                
                # recover Wind instruments' name
                if self.name == 'Wind':
                    j = j.replace('$','.')
                
                # Deal with Epoch naming
                if j == instrumt[i].dropna().iloc[-1]:
                    if self.name == 'PSP' and j.find('mag_RTN')>=0: # PSP's MAG_RTN data epoch name 
                        tempData['Epoch'] = temp_data['epoch_mag_RTN']
                    elif self.name == 'SolO' and i == 'SOLO_L2_MAG-RTN-NORMAL':
                        tempData['Epoch'] = temp_data['EPOCH']
                    else: # other SC instruments
                        tempData['Epoch'] = temp_data['Epoch']
                

                print('%s of %s loaded.'%(j, i))
            
            ## remove outliers using the interquartile range
            # raw = tempData.set_index('Epoch')
            # Q1=raw.quantile(0.25)
            # Q3=raw.quantile(0.75)
            # IQR=Q3-Q1
            # lowqe_bound=Q1 - 2 * IQR
            # upper_bound=Q3 + 2 * IQR
            # IQR_raw = raw[~((raw < lowqe_bound) |(raw > upper_bound)).any(axis=1)]
            
            # self.raw[i] = IQR_raw # store each instrument's DataFrame at one dict entry; dict length = number of instruments
            self.raw[i] = tempData.set_index('Epoch')

    def checkSameString(str1, str2):
        return str1.lower() == str2.lower() 
    
    def timeRes_interp(self, t_series):
        '''to interpolate data from diff instruments in the same time index'''
        from scipy import interpolate
        
        tempDF = pd.DataFrame()
        
        for i in self.raw: # i: dict key
            df = self.raw[i]
            df = df.dropna(axis=0,how='any')
            epoch_stamp = [t.timestamp() for t in df.index] # raw data Epoch -> timestamps, stored in <list>
            for j in df: # j: dataframe's columns
#                 tempDF[j] = interpolate.pchip_interpolate(epoch_stamp, df[j], t_series) # data after interpolation
                f = interpolate.PchipInterpolator(epoch_stamp, df[j],extrapolate=True)
                tempDF[j] = f(t_series)
                # f = interpolate.interp1d(told, a1['Bx'],fill_value="extrapolate")extrapolate=True
                # ynew = f(tnew)
        
        from datetime import datetime, timezone
        tempDF['Epoch'] = [datetime.fromtimestamp(t,tz=timezone.utc) for t in t_series]
        self.data = tempDF.set_index('Epoch')
        
    
    def savData(self,path):
        
        self.data.to_csv(path+'%s_%s_%s.csv'%(self.name, self.time_period[0],self.time_period[1]),index=True)
    
    def savObj(self, pklPath):
        import pickle
        with open(pklPath, 'wb') as f:
            pickle.dump(self, f)