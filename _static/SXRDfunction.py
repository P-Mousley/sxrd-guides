import os, fnmatch
import pandas as pd
import numpy as np
import IPython
import ctypes
import win32gui
import win32con

def close_gr_window(titlestring):
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    
    def enum_windows_callback(hwnd, results):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if titlestring.lower() in window_title.lower():
                results.append((hwnd, window_title))

    results = []
    win32gui.EnumWindows(enum_windows_callback, results)
    if results:
        for hwnd, title in results:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)


def makesavefit(folder,bestfit):
    mypath="{}\\{}".format(folder,bestfit)
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
        f = open(r'O:\anarod_standard_1-7_mingw\savefit.mac', "w")
        f.write("calc data\n")
        f.write("list data {}\\{}\dat_{} data_file_for_best_fit\n".format(folder,bestfit,bestfit))
        f.write("list Smod {}\\{}\sur_{} surface_model_bestfit\n".format(folder,bestfit,bestfit))
        f.write("list Bmod {}\\{}\\bul_{} bulk_model_bestfit\n".format(folder,bestfit,bestfit))
        f.write("list Mmod {}\\{}\\mol_{} molecule_model_bestfit\n".format(folder,bestfit,bestfit))
        f.write("list para {}\\{}\\par_{} parameters_bestfit\n".format(folder,bestfit,bestfit))
        f.write("list fit  {}\\{}\\fit_{} fit_file_bestfit\n".format(folder,bestfit,bestfit))
        f.write("list comp {}\\{}\comp_{} comparison_file_bestfit\n".format(folder,bestfit,bestfit))
        f.write("plot xyz 2 2 1 {}\\{}\{} return\n".format(folder,bestfit,bestfit))
        f.close()
        print(mypath)

    else:
        print('Folder already exists')

def readerrs(errorfile):
    f=open(fr"{errorfile}")
    lines=f.readlines()
    disps=[]
    occs=[]
    for line in lines:
        if ('Displacement' in line)& ('FIXED' not in line):
            disps.append(line)
        if ('Occupancy' in line)&('FIXED' not in line):
            occs.append(line)
    errors_par,errors_val,errors_err=[],[],[]
    for d in disps:
        errors_par.append(d.split('=')[0].strip(' ').strip('Displacement '))
        errors_val.append(float(d.split('=')[1].split()[0]))
        errors_err.append(float(d.split('=')[1].split()[2]))
    occs_par,occs_val,occs_err=[],[],[]
    for o in occs:
        occs_par.append(o.split('=')[0].strip(' '))
        occs_val.append(float(o.split('=')[1].split()[0]))
        occs_err.append(float(o.split('=')[1].split()[2]))
    derrorinfo=pd.DataFrame({'Parameter':errors_par,'Value':errors_val,'error':errors_err})
    oerrorinfo=pd.DataFrame({'Parameter':occs_par,'Value':occs_val,'error':occs_err})

    return derrorinfo,oerrorinfo

def calcreconab(mat,a,b):
    anew=round((mat[0]*a)+(mat[1]*b),5)
    bnew=round((mat[2]*a)+(mat[3]*b),5)
    return(anew,bnew)

def calcreverseab(mat,anew,bnew):
    arev=((mat[1]*bnew)-(mat[3]*anew))/(mat[1]*mat[2]-mat[0]*mat[3])
    brev=((mat[0]*bnew)-(anew*mat[2]))/(mat[3]*mat[0] - mat[2]*mat[1])
    return(arev,brev)

def matrixconv(mat1,mat2):
    """
    find conversion matrix needed to convert reciprocal position in notation of domain 1 (mat 1) into notation of domain 2 (mat2)
    """
    (n1,n2)=calcreconab(mat1,1,0)
    (n3,n4)=calcreconab(mat1,0,1)
    (k1,k2)=calcreconab(mat2,1,0)
    (k3,k4)=calcreconab(mat2,0,1)
    ac=((n2*k3) -(n4*k1))/((n2*n3) -(n4*n1))
    ad=((n2*k4) -(n4*k2))/((n2*n3) -(n4*n1))
    denom=abs((n2*n3)-(n4*n1))
    bc=(k1-(n1*ac))/n2
    bd=(k2-(n1*ad))/n2
    print(mat1,'convert to ',mat2)
    print('{} {}'.format(ac*denom, ad*denom), denom)
    print('{} {}'.format(bc*denom, bd*denom), denom)
    return(ac,bc,ad,bd)


def setgenfitloop():
    #set a generic fit loop to just save to anarod main directory
    f = open(r"O:\anarod_standard_1-7_mingw\\fitlogloop5.mac","w")
    f.write("fit control open O:\\anarod_standard_1-7_mingw\\fittest.log return return\npl allbo\n")
    for i in np.arange(5):
        f.write("fit asa pl allbo\n")
        f.write("li par O:\\anarod_standard_1-7_mingw\\fitting_{}.par results_from_loop_{}\n".format(str(i+1),str(i+1)))
    f.write("fit control close return return")
    f.close()

def openrod():
    os.chdir(r'O:\anarod_standard_1-7_mingw')
    os.startfile("rod.exe") 

def newpoint(n):
    print('<a class="anchor" id="{}"></a>'.format(n))
def calcfitlist(run,n):
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
            fitlist=fitlist.append(fitval)
    return fitlist,fitted

def loaddfs(folder,pattern,head=1):
    os.chdir(folder)
    listOfFiles = os.listdir('.') 
    totaldfs={}
    for n in range(len(listOfFiles)):
        entry=listOfFiles[n]
        if fnmatch.fnmatch(entry, pattern):
            fitdf=pd.read_csv(r'{}'.format(entry),header=head,sep='\s+')
            newname=entry.split('.')[0]
            totaldfs.update({'{}'.format(newname):fitdf})
            
    return totaldfs
    
def readfit(fitname,workfolder=0,pardata=[0],occdata=0):
    fitcols=['El','X','x1','x2','x3','x4','Y','y1','y2','y3','y4','Z','z1','z2','z3','z4','dw1','dw2','Occ']#'ind',
    fitcols2=['ind','El','X','x1','x2','x3','x4','Y','y1','y2','y3','y4','Z','z1','z2','z3','z4','dw1','dw2','Occ']#
    if len(fitname.split('\\'))==1:
        openf=r"{}\{}\fit_{}.fit".format(workfolder,fitname,fitname)
    else:
        openf=fitname
    try:
        fitfile=pd.read_csv(openf,header=1,sep='\s+',names=fitcols)
    except:
        fitfile=pd.read_csv(openf,header=1,sep='\s+',names=fitcols2)
    fitfile=fitfile.reset_index(drop=True)
    if len(pardata)>1:
        dispdf=pardata[pardata['data'].str.contains('displace')]
        if len(dispdf)>0:
            dispdf[['type','parameter','value','upp','low','fitted']]=dispdf.data.str.split(expand=True)
            fitfile['newZ']=fitfile.apply(lambda x: float(x['Z'])+float(dispdf[dispdf['parameter']==str(x['z2'])]['value'].values[0]) if x['z2']>0 else float(x['Z']), axis=1)
            fitfile['newZ']=fitfile.apply(lambda x: float(x['newZ'])+float(dispdf[dispdf['parameter']==str(x['z4'])]['value'].values[0]) if x['z4']>0 else float(x['newZ']), axis=1)
            fitfile['newX']=fitfile.apply(lambda x: float(x['X'])+float(dispdf[dispdf['parameter']==str(x['x2'])]['value'].values[0]) if x['x2']>0 else float(x['X']), axis=1)
            fitfile['newX']=fitfile.apply(lambda x: float(x['newX'])+float(dispdf[dispdf['parameter']==str(x['x4'])]['value'].values[0]) if x['x4']>0 else float(x['newX']), axis=1)
            fitfile['newY']=fitfile.apply(lambda x: float(x['Y'])+float(dispdf[dispdf['parameter']==str(x['y2'])]['value'].values[0]) if x['y2']>0 else float(x['Y']), axis=1)
            fitfile['newY']=fitfile.apply(lambda x: float(x['newY'])+float(dispdf[dispdf['parameter']==str(x['y4'])]['value'].values[0]) if x['y4']>0 else float(x['newY']), axis=1)
        occdf=pardata[pardata['data'].str.contains('occupancy')]
        if len(occdf)>0:
            occdf[['type','parameter','value','upp','low','fitted']]=occdf.data.str.split(expand=True)
            occdf.reset_index(drop=True)
            fitfile['occstr']=fitfile.apply(lambda x: occdf[occdf['parameter']==str(x['Occ'])]['value'].values[0] if x['Occ']>0 else 1 , axis=1)
            fitfile['occval']=fitfile['occstr'].apply(lambda x: float(x))
    return(fitfile)

def checkbonds(structurefile,highlim):
    xyzfile=pd.read_csv(r"{}".format(structurefile),sep='\s+',header=1,names=['El','X','Y','Z'])
    toplayer=xyzfile[(xyzfile['Z']>5)&(xyzfile['Z']<7)]
    xyzpos=np.array([xyzfile['X'],xyzfile['Y'],xyzfile['Z']])
    separations = xyzpos[ :,np.newaxis, :] - xyzpos[:,:, np.newaxis]
    squared_displacements = separations * separations
    square_distances = np.sum(squared_displacements, 0)
    bond_distances=np.sqrt(square_distances)
    limits=[0,highlim]
    smallbonds_val=[bond_distances[i][j] for i in range(bond_distances.shape[0]) for j in range(bond_distances.shape[1]) if (bond_distances[i][j]>limits[0])&(bond_distances[i][j]<limits[1]) ]
    smallbonds_ind=[[i,j] for i in range(bond_distances.shape[0]) for j in range(bond_distances.shape[1]) if (bond_distances[i][j]>limits[0])&(bond_distances[i][j]<limits[1]) ]

    a=np.array(smallbonds_val)
    minbondlocs=np.where(a==a.min())

    for loc in minbondlocs[0]:
        print(smallbonds_ind[loc],xyzfile.loc[smallbonds_ind[loc][0],'El'],xyzfile.loc[smallbonds_ind[loc][1],'El'], round(a[loc],4),'Å',xyzfile.loc[smallbonds_ind[loc][0],'Z'])
    
def readpar(fitname,workfolder=0):
    if workfolder==0:
        infodf=pd.read_csv(r"{}".format(fitname),header=1)
        pardata=pd.read_csv(r"{}".format(fitname),header=5, index_col=False, names=['data'])
    else:
        infodf=pd.read_csv(r"{}\{}\par_{}.par".format(workfolder,fitname,fitname),header=1)
        pardata=pd.read_csv(r"{}\{}\par_{}.par".format(workfolder,fitname,fitname),header=5, index_col=False, names=['data'])
    occdf=pardata[pardata['data'].str.contains('occupancy')]
    dispdf=pardata[pardata['data'].str.contains('displace')]
    if len(dispdf)>0:
        dispdf[['type','parameter','value','upp','low','fitted']]=dispdf.data.str.split(expand=True)
    if len(occdf)>0:
        occdf[['type','parameter','value','upp','low','fitted']]=occdf.data.str.split(expand=True)
    return(pardata,occdf,dispdf,infodf)
    
def readxyz(workfolder,fitname=0):
    if fitname!=0:
        filen=r"{}\{}\{}.xyz".format(workfolder,fitname,fitname)
        structure=pd.read_csv(r'{}'.format(filen),header=1,names=['El','X','Y','Z'],sep='\s+')
    else:
        filen=r"{}".format(workfolder)
        structure=pd.read_csv(r'{}'.format(filen),header=1,names=['El','X','Y','Z'],sep='\s+')
    return(structure)
    
#create dataset for loading into winrod
def combine_CTR(parts,scales,totscale,limits):
    """Combines individual scans of CTR sections into one dataset
    parts = array of filepaths to individual scan .dat files [filepath1,filepath2...etc]
    scales = array of individual section scale factors 
    totscale = total scale factor for overall dataset usually 1
    limits = array of snipping limit datapoint indexes for each section  either ['allpoints'] or [lowersnip,uppersnip]"""
    for i in np.arange(len(parts)):
        CTR_part=pd.read_csv('{}'.format(parts[i]), sep='\s+') #reads in data from dat file
        CTR_part['I']=CTR_part['I']*scales[i]*totscale #scales intensity according to scale factors
        print(len(CTR_part)) #print out check to see how many datapoints read in
        if limits[i][0]!='allpoints':
            CTR_part=CTR_part[limits[i][0]:limits[i][1]]  #if snipping limits set cut out unwanted points
            print(len(CTR_part))  #print out number of datapoints again to check that length of datapoints has been snipped properly
        else:
            print('no snip') #if no snipping limits set then print no snip message
        #print('max L = {}\n limits= [ {} ]'.format(CTR_part['L'].max(), limits[i])) #optional to print limits used as L range
        if i==0:
            CTR_total=CTR_part  #for first part create total dataframe
        else:
            CTR_total=pd.concat([CTR_total,CTR_part]) #for rest of parts add data to existing dataframe

    return CTR_total
def checkbonds(filestruct,limits):
    import SXRDplot as sxrdplot
    import numpy as np
    splot=sxrdplot.plotter()
    xyzfile=pd.read_csv(r"{}".format(filestruct),sep='\s+',header=1,names=['El','X','Y','Z'])
    xyzpos=np.array([xyzfile['X'],xyzfile['Y'],xyzfile['Z']])
    separations = xyzpos[ :,np.newaxis, :] - xyzpos[:,:, np.newaxis]
    squared_displacements = separations * separations
    square_distances = np.sum(squared_displacements, 0)
    bond_distances=np.sqrt(square_distances)
    smallbonds_val=[bond_distances[i][j] for i in range(bond_distances.shape[0]) for j in range(bond_distances.shape[1]) if (bond_distances[i][j]>limits[0])&(bond_distances[i][j]<limits[1]) ]
    smallbonds_ind=[[i,j] for i in range(bond_distances.shape[0]) for j in range(bond_distances.shape[1]) if (bond_distances[i][j]>limits[0])&(bond_distances[i][j]<limits[1]) ]

    a=np.array(smallbonds_val)
    minbondlocs=np.where(a==a.min())

    for loc in minbondlocs[0]:
        print(smallbonds_ind[loc],xyzfile.loc[smallbonds_ind[loc][0],'El'],xyzfile.loc[smallbonds_ind[loc][1],'El'], round(a[loc],4),'Å','at height',xyzfile.loc[smallbonds_ind[loc][0],'Z'])
    xyzfile['col']=xyzfile.apply(lambda x: splot.getcolour(x['El']),axis=1)
    plotvals=[xyzfile['X'],xyzfile['Y'],xyzfile['Z'],xyzfile['col']]
    boundvals=[xyzfile['X'],xyzfile['Y'],xyzfile['Z']]

    def boundbox(ax):
        """
        draw a boundary box on axis to ensure 3D plot is centred on display
        """
        [x,y,z]=boundvals
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(projection='3d')
    ax.clear()
    ptitle='test title'
    if len(ptitle)>0:
        #self.calcstructure(fit,parfile=parf)
        [x,y,z,cols]=plotvals
        scat1=ax.scatter3D(x, y, z,c=cols,edgecolors='black',alpha=0.2)
        ax.view_init(5,3)
        boundbox(ax)
        lines=[]
        for pos in smallbonds_ind:
            start=xyzfile.loc[pos[0]][['X','Y','Z']]
            end=xyzfile.loc[pos[1]][['X','Y','Z']]
            lines.append(ax.plot([start['X'], end['X']],[start['Y'],end['Y']],[start['Z'], end['Z']],color='red'))
    ax.set_title('{}'.format(filestruct),pad=0)
    plt.show()

        

def Mat2VTK(filename,matrix,formatname='ascii'):
# Writes a 3D matrix as a *.VTK file as input for Paraview.
# Coded by Manuel A. Diaz, NHRI 08.21.2016
# Following the example VTK file:
# # vtk DataFile Version 2.0
# Volume example
# ASCII
# DATASET STRUCTURED_POINTS
# DIMENSIONS 3 4 6
# ASPECT_RATIO 1 1 1
# ORIGIN 0 0 0
# POINT_DATA 72
# SCALARS volume_scalars char 1
# LOOKUP_TABLE default
# 0 0 0 0 0 0 0 0 0 0 0 0
# 0 5 10 15 20 25 25 20 15 10 5 0
# 0 10 20 30 40 50 50 40 30 20 10 0
# 0 10 20 30 40 50 50 40 30 20 10 0
# 0 5 10 15 20 25 25 20 15 10 5 0
# 0 0 0 0 0 0 0 0 0 0 0 0
# Here the example is extended to n-dimensional matrices. e.g:
#
# matrix(:,:,1) = [0 0 0 0 0 0 0 0 0 0 0 0;
# 0 5 10 15 20 25 25 20 15 10 5 0;
# 0 10 20 30 40 50 50 40 30 20 10 0;
# 0 10 20 30 40 50 50 40 30 20 10 0;
# 0 5 10 15 20 25 25 20 15 10 5 0;
# 0 0 0 0 0 0 0 0 0 0 0 0];
# matrix(:,:,2) = [0 0 0 0 0 0 0 0 0 0 0 0;
# 0 5 10 15 20 25 25 20 15 10 5 0;
# 0 10 20 30 40 50 50 40 30 20 10 0;
# 0 10 20 30 40 50 50 40 30 20 10 0;
# 0 5 10 15 20 25 25 20 15 10 5 0;
# 0 0 0 0 0 0 0 0 0 0 0 0];
#
# Usage: Mat2VTK('example.vtk',matrix,'binary');
# Get the matrix dimensions.
    [Nx,Ny,Nz] = np.shape(matrix);
    # Open the file.
    fid = open(filename, 'w');
    if fid == -1:
        error('Cannot open file for writing.');
    elif formatname=='ascii':    
            fid.write('# vtk DataFile Version 2.0\n');
            fid.write('Volume example\n');
            fid.write('ASCII\n');
            fid.write('DATASET STRUCTURED_POINTS\n');
            fid.write('DIMENSIONS {} {} {}\n'.format(Nx,Ny,Nz));
            fid.write('ASPECT_RATIO {} {} {}\n'.format(1,1,1));
            fid.write('ORIGIN {} {} {}\n'.format(0,0,0));
            fid.write('POINT_DATA {}\n'.format(Nx*Ny*Nz));
            fid.write('SCALARS Pressure int 1\n');
            fid.write('LOOKUP_TABLE default\n');
            for l in np.arange(Nz):
                for k in np.arange(Ny):
                    outdata=str(matrix[:,k,l].astype(int).tolist()).replace(',','').replace('\n',' ')
                    outdata=outdata.replace('[','')
                    outdata=outdata.replace(']','')
                    fid.write('{} '.format(outdata));    # Close the file.
                fid.write('\n')
#         case 'binary'
#             fprintf(fid,'# vtk DataFile Version 2.0\n');
#             fprintf(fid,'Volume example\n');
#             fprintf(fid,'BINARY\n');
#             fprintf(fid,'DATASET STRUCTURED_POINTS\n');
#             fprintf(fid,'DIMENSIONS %d %d %d\n',Nx,Ny,Nz);
#             fprintf(fid,'ASPECT_RATIO %d %d %d\n',1,1,1);
#             fprintf(fid,'ORIGIN %d %d %d\n',0,0,0);
#             fprintf(fid,'POINT_DATA %d\n',Nx*Ny*Nz);
#             fprintf(fid,'SCALARS Pressure float 1\n');
#             fprintf(fid,'LOOKUP_TABLE default\n');
#             fwrite(fid, matrix(:),'float','ieee-be');
#         otherwise
#             error('wrong input dummy :P');
    fid.close();


def calchklpoints(alpha,gamma,delta,omega):
    def rotang(ang,x=0,y=0,z=0):
        if x==1:
            rotm=np.matrix([[1,0,0],[0,np.cos(ang),-np.sin(ang)],[0,np.sin(ang),np.cos(ang)]])
        elif y==1:
            rotm=1
        elif z==1:
            rotm= np.matrix([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]])
        return rotm
    RGam =rotang(gamma-alpha,x=1)
    RDel =rotang(delta,z=1)
    ROm = rotang(omega,z=1)
    Ralp= rotang(alpha,x=1)
    energy=12.3
    #k = 2 * (math.pi / 12.398) * energy
    wavelength=1.0162704918 # Angstrom wavelength from experiment setup
    k=2*np.pi/wavelength
    kin=np.array([0,k,0]).reshape(3,1)

    invRO=np.linalg.inv(ROm)
    invRA=np.linalg.inv(Ralp)
    kinmat=np.dot(invRO,np.dot(invRA,kin))
    kfinmat=np.dot(invRO,np.dot(RGam,np.dot(RDel,kin)))
    Hom=kfinmat-kinmat#np.subtract(kfinmat,kinmat)
    
    #kinman=np.dot(ROm,np.dot(Ralp,kin))
    #koutman=np.dot(RDel,np.dot(RGam,kin))
    #Q=np.subtract(koutman,kinman)
    
    # #knew=np.array(np.reshape(knew,(1,3)))[0]    
    # k1=rotang(alpha,x=1)*rotang(omega,z=1)*kin
    # #k1=np.array([k*np.cos(alpha)*np.cos(omega),k*np.cos(alpha)*np.sin(omega),-k*np.sin(alpha)])
    # k2=rotang(-1*alpha,x=1)*rotang(omega,z=1)*kin
    # #k2=[k*np.cos(delta)*np.cos(omega),k*np.cos(delta)*np.sin(omega),k*np.sin(delta)]
    # knew=RGam*RDel*(np.reshape(k2,(3,1)))
    # kfin=np.subtract(knew,k1)
    
   # width=0.25
   # width2=0.25
   # u = np.linspace(1.5*np.pi-delta+omega-width, 1.5*np.pi+omega-delta+width, 100)
   # #u = np.linspace(0, 2*np.pi, 100)
   # v = np.linspace((1.5*np.pi)+gamma-alpha-width2,(1.5*np.pi)+gamma-alpha+width2, 100)
   # rad=np.linalg.norm(kinmat)
   # sphx = rad * np.outer(np.cos(u), np.sin(v))
   # sphy =rad * np.outer(np.sin(u), np.sin(v))
   # sphz = rad * np.outer(np.ones(np.size(u)), np.cos(v))
    return(Hom,kinmat,kfinmat)

def createrounded(filelist,projfold):
    rounded=pd.DataFrame()
    for r in filelist:
        rdf=pd.read_csv(r'{}\{}'.format(projfold,r))
        rounded=rounded.append(rdf)
    return(rounded)


def openrf(rf):
    readfile=open(r'{}'.format(rf))
    lines=readfile.readlines()
    return(lines)
def groupav(rounded,step):
    rounded['rhvals']=round(rounded['hvals']/step[0])*step[0]
    rounded['rkvals']=round(rounded['kvals']/step[1])*step[1]
    rounded['rlvals']=round(rounded['lvals']/step[2])*step[2]
    averaged=rounded.groupby(by=['rhvals','rkvals','rlvals']).sum().reset_index()
    #averaged['hkl']=averaged.index
    #averaged.loc[(0.934, 0.96, -0.002),'intvals']
    averaged['rhvals']=round(averaged['rhvals'],int(np.log10((1/step[0]))//1)+1)
    averaged['rkvals']=round(averaged['rkvals'],int(np.log10((1/step[1]))//1)+1)
    averaged['rlvals']=round(averaged['rlvals'],int(np.log10((1/step[2]))//1)+1)

    upper=(averaged['rhvals'].max(),averaged['rkvals'].max(),averaged['rlvals'].max())

    lower=(averaged['rhvals'].min(),averaged['rkvals'].min(),averaged['rlvals'].min())
    hrange=averaged['rhvals'].max()-averaged['rhvals'].min()
    krange=averaged['rkvals'].max()-averaged['rkvals'].min()
    lrange=averaged['rlvals'].max()-averaged['rlvals'].min()
    hpix=round(hrange/step[0])
    kpix=round(krange/step[1])
    lpix=round(lrange/step[2])
    return(averaged,(hpix,kpix,lpix),lower,upper)
