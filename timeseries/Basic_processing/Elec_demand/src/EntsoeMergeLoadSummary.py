"""
Merges individual country files into one file 2011-2014
Input file:  country_code.csv
Output file(s): summary_2015-2020-1h.csv
"""

import pandas as pd
import numpy as np
import time
import os
import sys




class EntsoeMergeAreaLoads:
    """
    Class for merging loads of separate areas for years 2015-
    """

    def __init__(self, ADD=""):
        self.ADD=ADD

        self.country_codes = [
        'AT',
        'BE',	
        #'BG',
        'CH',
        #'CY',
        #'CZ',
        'DE',
        #'DE_AT_LU',
        #'DK',
        'DK_1',
        'DK_2',
        'EE',
        'ES',
        'FI',
        'FR',
        'GB',
        #'GR',
        #'HR',
        #'HU',
        #'IE',
        #'IE_SEM',
        'LT',
        #'LU',
        'LV',
        'NL',
        #'NO',
        'NO_1',
        'NO_2',
        'NO_3',
        'NO_4',
        'NO_5',
        'PL',
        #'PT',
        #'RO',
        #'SE',
        'SE_1',
        'SE_2',
        'SE_3',
        'SE_4',
        #'SI',
        #'SK'
        ]
        self.start='2014-12-31 20:00:00'
        self.end='2021-01-01 10:00:00'


    #defining function for x hour time shift for lambda function
    def utcTime(self,orig, shift):
        t = orig - pd.DateOffset(hours=shift)
        return t

    def process_areas(self):
        startTime = time.time()
        print('\n')
        print('merging and curing load data')

        # creating dateFrames from 2014 to 2021 with 15min and 60min frequencies
        dfMerge15min = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='15 min'))
        dfMerge30min = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='30 min'))
        dfMerge1h = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))

        for c in self.country_codes:

                #refresh dataframe where data will be appended
                
                df = pd.DataFrame()

                try:
                        # read csv
                        #csvInput = os.path.normpath(self.ADD+'output/' + c +'.csv')
                        csvInput = os.path.normpath(self.ADD+'downloads/' + c +'.csv')
                        df = pd.read_csv(csvInput)
                        df.columns=['time', c]

                except Exception as e:
                        print(e)


                # processing yearly data if available
                if not df.empty:

                        # separate timezone info from datetime			
                        df['tz'] = df['time'].str[20:22].astype(int)
                        df['time'] = df['time'].str[:19]
                        df['time'] = pd.to_datetime(df['time'])
                
                        # convert time to utc+0 and set to index
                        df['time'] = df.apply(lambda row: self.utcTime(row['time'], row['tz']), axis=1)
                        df.drop(columns='tz', inplace=True)
                        df.set_index('time', inplace=True)



                        
                        ###
                        # Removing single or double odd values
                        # removing outliers helps in following steps

                        # replacing zeroes with null
                        df = df.replace(0, np.nan)

                        # removing values that are below 0.5 x average
                        picks = (df / df.mean() < 0.4) 
                        df.loc[picks[c], [c]] = np.nan
                
                        # removing values that are above 2 x average
                        picks = (df / df.mean() > 2)
                        df.loc[picks[c], [c]] = np.nan

                        # removing 1h and 2h valleys and peaks if above certain threshold
                        # calculating change / average change
                        relChange = df.diff() / df.diff().abs().mean()
                
                        # removing 1h peaks 
                        picks = (relChange > 5) & (relChange.shift(periods=-1) < -5)
                        df.loc[picks[c], [c]] = np.nan
                                        
                        # removing 1h valleys 
                        picks = (relChange < -5) & (relChange.shift(periods=-1) > 5)
                        df.loc[picks[c], [c]] = np.nan
                
                        # removing 2h peaks 
                        # first hour
                        picks = (relChange > 5) & (relChange.shift(periods=-2) < -5)
                        df.loc[picks[c], [c]] = np.nan
                        # second hour
                        picks = (relChange.shift(periods=1) > 5) & (relChange.shift(periods=-1) < -5)
                        df.loc[picks[c], [c]] = np.nan
                
                        # removing 2h valleys 
                        # first hour
                        picks = (relChange < -5) & (relChange.shift(periods=-2) > 5)
                        df.loc[picks[c], [c]] = np.nan
                        # second hour
                        picks = (relChange.shift(periods=1) < -5) & (relChange.shift(periods=-1) > 5)
                        df.loc[picks[c], [c]] = np.nan


                        ###
                        # Removing longer periods of dubious data

                        # removing values that are high above the rolling average and
                        # not precided by empty value
                        dfRollingAvg = df.rolling(window=24*7, min_periods=1).mean()
                        diff = df-dfRollingAvg
                        diff[diff < 0] = 0
                        relDiff = diff / diff.mean()
                        picks = (relDiff > 8) &  (df.shift(1).sum()>0) 
                        df.loc[picks[c], [c]] = np.nan

                        # removing values that are much smaller than the rolling average 
                        dfRollingAvg = df.rolling(window=24*7, min_periods=1).mean()
                        diff = dfRollingAvg-df
                        diff[diff < 0] = 0
                        relDiff = diff / diff.mean()
                        picks = (relDiff > 8) 
                        df.loc[picks[c], [c]] = np.nan




                        ###
                        # Expanding the dataset to cover all timedates

                        # checking ts timesteps and expanding the dataset to cover all timesteps
                        tsStep = np.timedelta64(df.index.values[-1] - df.index.values[-2], "m")
                        tsStep = str(tsStep)[:2]
                        tsStep = int(tsStep)
                        df = df.resample(str(tsStep)+'min').asfreq()


                        ###
                        # Filling the gaps

                        # finding max 3 consecutive empty data points to be interpolated
                        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=4)
                        mask = df.isna().rolling(window=indexer, min_periods=1).sum() * df.isna()
                        mask = (mask.rolling(4, min_periods=1).max()) < 4* df.isna()

                        # copying df to temp, interpolate temp, pick selected values to original
                        temp = df.copy()
                        temp = temp.interpolate(method='linear', limit_area='inside')
                        df.loc[mask[c], [c]] = temp.loc[mask[c], [c]]


                        # fill longer gaps, if total number of missing values less than threshold
                        if df[c].isna().sum() < (15000 / pd.to_numeric(tsStep)):
                                missing = (df.loc[df.isna()[c], [c]].index)

                                for m in missing:
                                        #print(df.loc[m + pd.DateOffset(years=1, hours=-1), [c]])

                                        #print(df.index[df.index == m].tolist())

                                        # calculating relative change from the same hour previous week
                                        # or next week if no data for previous week
                                        if ((m + pd.DateOffset(days=7)) in df.index) and (df[c][m + pd.DateOffset(days=7)].sum()>0):
                                                factor = df.loc[m + pd.DateOffset(days=7), [c]] / df.loc[m + pd.DateOffset(days=7, minutes=-tsStep), [c]]
                                        elif ((m + pd.DateOffset(days=-7)) in df.index) and (df[c][m + pd.DateOffset(days=-7)].sum()>0):
                                                factor = df.loc[m + pd.DateOffset(days=-7), [c]] / df.loc[m + pd.DateOffset(days=-7, minutes=-tsStep), [c]]

                                        #calculating projected value and reference value based on rolling average
                                        projection = (df[c][m + pd.DateOffset(minutes=-tsStep)] * factor).sum()
                                        refValue = df.rolling(window=24*7, min_periods=1).mean()

                                        # using projected value if between 0.8-1.2 * refValue
                                        if projection > 1.2 * refValue.loc[m, c]:
                                                df.loc[m, [c]] = refValue.loc[m, c] * 1.2
                                        elif projection < 0.8 * refValue.loc[m, c]:
                                                df.loc[m, [c]] = refValue.loc[m, c] * 0.8
                                        else:
                                                df.loc[m, [c]] = projection


                        # interpolate as the last measure
                        df.interpolate(method='index', inplace=True, limit_area='inside')				



                        # merge country ts to collection tables. 
                        # preserve original frequency and resample to 1h when needed.
                        if tsStep == 15:
                                dfMerge15min = pd.merge(dfMerge15min, df, how='left', left_index=True, right_index=True, validate='one_to_one')
                                df1h = df.resample('60min').mean()
                                dfMerge1h = pd.merge(dfMerge1h, df1h, how='left', left_index=True, right_index=True, validate='one_to_one')
                        elif tsStep == 30:
                                dfMerge30min = pd.merge(dfMerge30min, df, how='left', left_index=True, right_index=True, validate='one_to_one')
                                df1h = df.resample('60min').mean()
                                dfMerge1h = pd.merge(dfMerge1h, df1h, how='left', left_index=True, right_index=True, validate='one_to_one')
                        else:
                                dfMerge1h = pd.merge(dfMerge1h, df, how='left', left_index=True, right_index=True, validate='one_to_one')



                        #print(dfMerge.head())

                        print(c, " ", round(time.time() - startTime,2), "s  -- done")



        dfMerge15min.dropna(inplace = True, how='all')

        #print('\n')
        #print(dfMerge15min.head())
        #print(dfMerge15min.info())

        #csvOutput15min = os.path.normpath(self.ADD+'output/summary_2015-2020-15min' +'.csv')
        #dfMerge15min.to_csv(csvOutput15min)



        dfMerge30min.dropna(inplace = True, how='all')

        #print('\n')
        #print(dfMerge30min.head())
        #print(dfMerge30min.info())

        #csvOutput30min = os.path.normpath('output/summary_2015-2020-30min' +'.csv')
        #dfMerge30min.to_csv(csvOutput30min)



        dfMerge1h.dropna(inplace = True, how='all')

        noAreas = ['NO_1','NO_2','NO_3','NO_4','NO_5']
        seAreas = ['SE_1','SE_2','SE_3','SE_4']
        dkAreas = ['DK_1','DK_2']

        if pd.Series(noAreas).isin(dfMerge1h.columns).all():
                dfMerge1h['NO'] = dfMerge1h[noAreas].sum(axis=1)
        if pd.Series(seAreas).isin(dfMerge1h.columns).all():
                dfMerge1h['SE'] = dfMerge1h[seAreas].sum(axis=1)
        if pd.Series(dkAreas).isin(dfMerge1h.columns).all():
                dfMerge1h['DK'] = dfMerge1h[dkAreas].sum(axis=1)

        dfMerge1h = dfMerge1h.reindex(sorted(dfMerge1h.columns), axis=1)

        #print('\n')
        #print(dfMerge1h.head())
        #print(dfMerge1h.info())
                
        csvOutput1h = os.path.normpath(self.ADD+'output/summary_2015-2020-1h.csv')
        dfMerge1h.to_csv(csvOutput1h)

        print("load summaries ", round(time.time() - startTime,2), "s  -- done")


    def delete_intermediates(self):
        """
        Removes input files
        """
        for c in self.country_codes:
            csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
            if os.path.exists(csvName):
                os.remove(csvName)

"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        summary = EntsoeMergeAreaLoads(ADD)
        
        summary.process_areas()

        #removes area specific files so that they do not get mixed with files from
        #earlier years
        summary.delete_intermediates()


