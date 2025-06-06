import os,time,shutil
import subprocess
import numpy as np
import pandas as pd
def addinitmacline(macline,templateinitfile,initfile):

    f=open(templateinitfile)
    lines=f.readlines()
    f.close()
    
    if os.path.isfile(initfile):
            os.remove(initfile)
    
    f=open(initfile,"w")
    for line in lines:
        if line.endswith('\n'):
            f.write(line)
        else:
            f.write(f"{line}\n")
    if lines[-1]!=macline:
        f.write(macline)
    f.close()

def runrod(rodexe):
    return subprocess.Popen(rodexe,cwd=os.path.dirname(rodexe),creationflags=subprocess.CREATE_NEW_CONSOLE) 

def checkchidiff(fitlist):
    return fitlist.values[2:].max()-fitlist.values[2:].min()

def domacrofits(donefile,rodexe):
    if os.path.isfile(donefile):
        os.remove(donefile)
    rodproc=runrod(rodexe)
    while not os.path.isfile(donefile):
        time.sleep(5)
    rodproc.kill()
    
def calcfitlist(run,n):
    ranges=[]
    for i in np.arange(0,n):
        #print(i)
        #parse information from .par file for fitting cycle
        fitdf=pd.read_csv('{}_{}.par'.format(run,str(i+1)),header=0)
        
        if i==1:
            fitted=[]
            for n in np.arange(len(fitdf)):
                if ('YES' in fitdf.iloc[n][0]):
                   # print(i)
                    #print('par {}: {}'.format(str(n-3),fitdf.iloc[n][0]))
                    fitted.append(n-4)
                    ranges.append('par {}: {}'.format(str(n-3),fitdf.iloc[n][0]))

        #parse location of dataset file
        datfile=fitdf.iloc[0][0].split(' ')[2]

        #read in dataset from file
        dat=pd.read_csv(r'{}'.format(datfile),sep='\s+')

        #calculate the number of datapoints
        ndat=len(dat)

        #calculate the number of free parameters used in the fitting
        nfree=len(fitdf[fitdf['!results_from_loop_{}'.format(i+1)].str.contains('YES')])

        #get non-normalised chi^2 value
        chi2=float(fitdf.iloc[1][0].split('=')[1])

        #calculate normalised chi^2 value following equation from ROD source code
        normchi2=round(chi2/(ndat-nfree+1e-10),4)

        #save normalised chi^2 value as panda series and add to total list
        fitval=pd.Series(normchi2,index=['fit{}'.format(i)])
        if i==0:
            fitlist=fitval
        else:
            fitlist=pd.concat([fitlist,fitval])
    return fitlist,fitted,ranges,fitdf 



def createmacro(mname,files,startvals,donefile,mtype):
    mactypes={'ASA':'fitlogloop5','RUN':'fitlogloop5run'}
    f=open(fr"{files['fitmac']}",'w')
    f.write(fr"re bul {files['bul']}"+'\n')
    f.write(fr"re dat  {files['dat']}"+'\n')
    f.write('mac sfsetup'+'\n')
    f.write(fr"mac {files['loadmac']} return return"+'\n')
    outname=mname.replace('.sur','')
    f.write(fr"re fit {files['fit']}"+'\n')
    f.write(fr"re par {startvals}"+'\n')
    f.write(f'mac {mactypes[mtype]}'+'\n')
    f.write(fr"li par {donefile} done")
    f.close()
    

def ASAloop(modname,fitlist,files,fit5path,donefile,rodexe,asacount,run):
    while checkchidiff(fitlist)>0.01:
        startvals=fit5path
        createmacroASA(modname,files,startvals,donefile)
        domacrofits(donefile,rodexe)
        asacount+=1
        print(f'asa loop {asacount}')
        fitlist,fitted,ranges,fitdf=calcfitlist(run,5)
    return fitlist[4],asacount

def runloop(modname,fitlist,files,fit5path,donefile,rodexe,runcount,run):
    while checkchidiff(fitlist)>0.005:
        startvals=fit5path
        createmacroRUN(modname,files,startvals,donefile)
        domacrofits(donefile,rodexe)
        runcount+=1
        print(f'run loop {runcount}')
        fitlist,fitted,ranges,fitdf=calcfitlist(run,5)
    return fitlist[4],runcount
    