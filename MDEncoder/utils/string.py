from scipy.interpolate import griddata, interp1d
from scipy.ndimage import map_coordinates
#from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import itertools
from functools import reduce

def find_ZTS_nd(df,refine_dims=[],frame=[],initial_pt=[],mid_pt=[],mid_index=None,n_samples=20,bins=50, step_size=0.01, max_itr=5, tolerance = 0.05 , ends_movement = 0.01, pmf_input=False, plots=False):
    #calculate N-D pmf
    def pmf(data,bins,Temperature=300):
        lims = [ [np.min(data[:,i])-0.02,np.max(data[:,i])+0.02] for i in range(data.shape[1]) ]
        h , edges = np.histogramdd(data,bins=bins,range=lims)

        RT =  0.00198588*Temperature
        F = -RT*np.log((h.T+1e-18)/np.sum(h))
        F = F - np.min(F)

        return F , edges

    def ZTS_ND(data,pts,mid_index,n_samples,bins,step_size=0.01,plots=False):
        from datetime import datetime
        F, edges = pmf(data,bins)
        edges = [edge[:-1] for edge in edges]
        dx = [x[1]-x[0] for x in edges]
        xgrid = np.meshgrid(*edges)
        xgrid = [ xi + dxi/2 for xi , dxi in zip(xgrid,dx)]
        gridpoints = np.vstack([ xi.flatten() for xi in xgrid ]).T
        
        mass = np.ones_like(pts)
        mass[0,:] = mass[0,:] * ends_movement
        mass[-1,:] = mass[-1,:] * ends_movement
        if mid_index>=0:
            mass[mid_index,:] = mass[mid_index,:] * ends_movement
        #Make a figure and a axes
        if plots:
            fig, ax = plt.subplots(1,1)
        
        grad = np.gradient(F,*dx)        
        #Evolve points so that they respond to the gradients. This is the "zero temperature string method"
        stepmax=20
        for i in range(stepmax):
            print("Iteration: {0}".format(i))
            #Find gradient interpolation
            #print(np.shape(grad[0]),np.shape(grad[0].flatten()),np.shape(gridpoints))   
            t0 = datetime.now()
            Dx = [ griddata(gridpoints,gradx.flatten(),pts, method='linear',fill_value=0.0) for gradx in grad]
            #Dx = [ for gradx in grad]
            print('Time taken: {}'.format(datetime.now()-t0))
            h = np.amax(np.sqrt( reduce( (lambda x, y: x+y), map(np.square,Dx)) ))
            print(h,np.shape(gridpoints))
            #Evolve
            if h>0:
                pts -= np.multiply(step_size * np.vstack([ Dxi for Dxi in Dx[::-1] ]).T / h , mass)

                #Reparameterize
                arclength = np.hstack([0,np.cumsum(np.linalg.norm(pts[1:] - pts[:-1],axis=1))])
                arclength /= arclength[-1]
                pts = np.vstack([ interp1d(arclength,pts[:,dim])(np.linspace(0,1,n_samples)) for dim in range(pts.shape[1]) ]).T
            
            if i % 5:
                step_size *= 0.7
            if i % 20 and plots:
                #print(i, np.sum(griddata(gridpoints,z.flatten(),(pts[:,0],pts[:,1]), method='linear')))
                #This draws the intermediate states to visualize how the string evolves.
                #ax.plot(pts[:,0],pts[:,1],color='r',marker='x',linestyle='--')
                ax.plot(pts[:,0],pts[:,1], color=plt.cm.spring(i/float(stepmax)))

        if plots:
            ax.plot(pts[:,0],pts[:,1], color='k', linestyle='-', marker='o')
            heatmap = ax.imshow(F, cmap=plt.cm.rainbow, vmin = np.nanmin(F), vmax=np.nanmin(F)+12, origin='lower', aspect='auto', extent = (xgrid[0][0][0], xgrid[0][-1][-1],xgrid[1][0][0], xgrid[1][-1][-1]), interpolation="bicubic")
            heatmap.cmap.set_over('white')
            ax.autoscale(False)

            bar = fig.colorbar(heatmap)
            bar.set_label("Free Energy (kcal/mol)", rotation=90, fontname="Avenir", fontsize=14)

            fig, ax = plt.subplots(1,1)
            ax.plot(np.linspace(0,1,n_samples),griddata(gridpoints,F.flatten(),(pts[:,0],pts[:,1]), method='linear'))
            ax.set_ylabel("Free Energy (kcal/mol)")
            ax.set_xlabel("Reaction Progress")
            plt.show()
            #fig.savefig("1Dpmf.png")

        return pts
    
    def pairdim_by_distance(pts,n_dims):
        distance_list = []
        pairs = []
        for i in range(n_dims):
            for j in range(i+1,n_dims):
                distance = (pts[0][i] - pts[1][i])**2 + (pts[0][j] - pts[1][j])**2
                distance_list.append(distance)
                pairs.append((i,j))
        return [pairs[x] for x in np.argsort(distance_list)[::-1] ]

    def dim_by_distance(pts,n_dims):
        distance_list = []
        for i in range(0,n_dims):
            distance = (pts[0][i] - pts[1][i])**2 
            distance_list.append(distance)
        sorted_col = np.argsort(distance_list)[::-1]
        return [(sorted_col[i],sorted_col[i+1]) for i in range(0,n_dims,2) ]
    
    #initialize string
    assert len(frame) > 0 or len(initial_pt)>0 , "Must provide initial endpoints for string!"
    
    if len(frame)>0:
        pt1 = df.iloc[frame[0]].to_numpy()
        pt2 = df.iloc[frame[1]].to_numpy()
    else:
        pt1 = initial_pt[0]
        pt2 = initial_pt[1]
    if len(mid_pt)>0:
        if mid_index:
            t1 = np.reshape(np.linspace(0,1,mid_index),(mid_index,1))
            t2 = np.reshape(np.linspace(0,1,n_samples-mid_index),(n_samples-mid_index,1))
            string = np.vstack( [pt1+t1*(mid_pt-pt1) , mid_pt+t2*(pt2-mid_pt)] )
        else:
            t1 = np.reshape(np.linspace(0,1,(n_samples+1)//2),((n_samples+1)//2,1))
            t2 = np.reshape(np.linspace(0,1,(n_samples)//2),((n_samples)//2,1))
            string = np.vstack( [pt1+t1*(mid_pt-pt1) , mid_pt+t2*(pt2-mid_pt)] )
            mid_index = t1.shape[0] - 1
    else:
        t = np.reshape(np.linspace(0,1,n_samples),(n_samples,1))
        string = pt1+t*(pt2-pt1)
        mid_index = -1
    
    delta = 1.0
    itr = 1
    #sorted_col = np.argsort(df.var().to_numpy())[::-1]
    #sorted_pairs = dim_by_distance((string[0,:],string[-1,:]),len(df.columns))

    #initial minimization with small step
    string = ZTS_ND(df.to_numpy(),string,mid_index,n_samples,bins,step_size,plots=plots)
    '''
    while (delta>tolerance and itr<=max_itr) or itr<=2:
        old_string = string.copy()
        string = ZTS_ND(df.to_numpy(),string,mid_index,n_samples,bins,step_size*0.1,plots=plots)
        #delta = np.mean(np.abs(string-old_string))
        delta = np.max(np.abs(string-old_string))
        itr += 1
        step_size *= 0.7
        print("Change in string = {0}".format(delta))
    if itr <= max_itr:    
        print("String converged.")
    else:
        print("Maximum iteration reached, please check the string!")
    '''
    return string

def find_ZTS(df,refine_dims=[],frame=[],initial_pt=[],mid_pt=[],mid_index=None,n_samples=20,bins=50, step_size=0.01, max_itr=5, tolerance = 0.05 , ends_movement = 0.01, all_dims=False, plots=False):
    #calculate 2D pmf
    def pmf(xdata,ydata,bins,Temperature=300):
        x , y = xdata , ydata
        h , xe, ye = np.histogram2d(x,y,bins=bins,range=[[np.min(x)-0.02,np.max(x)+0.02],[np.min(y)-0.02,np.max(y)+0.02]])
        X, Y = np.meshgrid(xe,ye)

        RT =  0.00198588*Temperature
        F = -RT*np.log((h.T+1e-18)/np.sum(h))
        F = F - np.min(F)

        return (X,Y,F)

    def ZTS_2D(input_data,pts,n_samples,bins,step_size=0.01,pmf_input=False,plots=False):
        if not pmf_input:
            data = pmf(input_data[0],input_data[1],bins)
            x = data[0]
            y = data[1]
            z = data[2]
            dx = x[0][1] - x[0][0]
            dy = y[1][0] - y[0][0]
            x = x + dx/2
            y = y + dy/2
            gridpoints = np.vstack([x[:-1,:-1].flatten(),y[:-1,:-1].flatten()]).T
        else:
            #input is assumed of the form (xcoor,ycoor,pmf_value) for each row 
            if type(bins) == int:
                bins = (bins,bins)
            x, y = input_data[:,0].reshape(bins).T , input_data[:,1].reshape(bins).T 
            dx , dy = x[0,1] - x[0,0] , y[1,0] - y[0,0]
            z = input_data[:,2].reshape(bins).T
            gridpoints = np.vstack([x.flatten(),y.flatten()]).T
            
        mass = np.ones_like(pts)
        mass[0,:] = mass[0,:] * ends_movement
        mass[-1,:] = mass[-1,:] * ends_movement
        
        #Make a figure and a axes
        if plots:
            fig, ax = plt.subplots(1,1)

        gradx, grady = np.gradient(z,dx,dy)
        #Evolve points so that they respond to the gradients. This is the "zero temperature string method"
        stepmax=100
        for i in range(stepmax):
            #Find gradient interpolation
            Dx = griddata(gridpoints,gradx.flatten(),(pts[:,0],pts[:,1]), method='linear',fill_value=0.0)
            Dy = griddata(gridpoints,grady.flatten(),(pts[:,0],pts[:,1]), method='linear',fill_value=0.0)
            h = np.amax(np.sqrt(np.square(Dx)+np.square(Dy)))
            
            #Evolve
            pts -= np.multiply(step_size * np.vstack([Dy,Dx]).T / h , mass)

            #Reparameterize
            arclength = np.hstack([0,np.cumsum(np.linalg.norm(pts[1:] - pts[:-1],axis=1))])
            arclength /= arclength[-1]
            #pts = np.vstack([interp1d(arclength,pts[:,0])(np.linspace(0,1,2*npts)), interp1d(arclength,pts[:,1])(np.linspace(0,1,2*npts))]).T
            pts = np.vstack([interp1d(arclength,pts[:,0])(np.linspace(0,1,n_samples)), interp1d(arclength,pts[:,1])(np.linspace(0,1,n_samples))]).T
            if i % 20 and plots:
                #This draws the intermediate states to visualize how the string evolves.
                #ax.plot(pts[:,0],pts[:,1],color='r',marker='x',linestyle='--')
                ax.plot(pts[:,0],pts[:,1], color=plt.cm.spring(i/float(stepmax)))

        if plots:
            ax.plot(pts[:,0],pts[:,1], color='k', linestyle='-', marker='o')
            heatmap = ax.imshow(z, cmap=plt.cm.rainbow, vmin = np.nanmin(z), vmax=np.nanmin(z)+12, origin='lower', aspect='auto', extent = (x[0][0], x[-1][-1],y[0][0], y[-1][-1]), interpolation="bicubic")
            heatmap.cmap.set_over('white')
            ax.autoscale(False)

            bar = fig.colorbar(heatmap)
            bar.set_label("Free Energy (kcal/mol)", rotation=90, fontname="Avenir", fontsize=14)

            fig, ax = plt.subplots(1,1)
            ax.plot(np.linspace(0,1,n_samples),griddata(gridpoints,z.flatten(),(pts[:,0],pts[:,1]), method='linear'))
            ax.set_ylabel("Free Energy (kcal/mol)")
            ax.set_xlabel("Reaction Progress")
            plt.show()
            #fig.savefig("1Dpmf.png")

        return pts
    
    def pairdim_by_distance(pts,n_dims):
        distance_list = []
        pairs = []
        for i in range(n_dims):
            for j in range(i+1,n_dims):
                distance = (pts[0][i] - pts[1][i])**2 + (pts[0][j] - pts[1][j])**2
                distance_list.append(distance)
                pairs.append((i,j))
        return [pairs[x] for x in np.argsort(distance_list)[::-1] ]

    def dim_by_distance(pts,n_dims):
        distance_list = []
        for i in range(0,n_dims):
            distance = (pts[0][i] - pts[1][i])**2 
            distance_list.append(distance)
        sorted_col = np.argsort(distance_list)[::-1]
        return [(sorted_col[i],sorted_col[i+1]) for i in range(0,n_dims,2) ]
    
    #initialize string
    assert len(frame) > 0 or len(initial_pt)>0 , "Must provide initial endpoints for string!"
    if type(df) == np.ndarray:
        pmf_input = True
    else:
        pmf_input = False
    #initialize endpoints with specified frames in the dataset    
    if len(frame)>0:
        pt1 = df.iloc[frame[0]].to_numpy()
        pt2 = df.iloc[frame[1]].to_numpy()
    else:
        pt1 = initial_pt[0]
        pt2 = initial_pt[1]
    #interpolate string between specified points
    if len(mid_pt)>0:
        if len(mid_index)>0:
            pts =  mid_pt + [pt2]
            indices =  mid_index + [n_samples-1]
            lastpt , lastidx = pt1 , 0
            string = []
            for pt, idx in zip(pts,indices):                
                t1 = np.reshape(np.linspace(0,1,idx-lastidx+1),(idx-lastidx+1,1))
                if len(string)>0:
                	string = np.vstack( [string ,  (lastpt+t1*(pt-lastpt))[1:,:] ] )
                else:
                    string = lastpt+t1*(pt-lastpt)
                lastpt, lastidx = pt, idx
        else:
            assert len(mid_pt) == 1 , "More than one mid points requires corresponding indices!"
            t1 = np.reshape(np.linspace(0,1,(n_samples+1)//2),((n_samples+1)//2,1))
            t2 = np.reshape(np.linspace(0,1,(n_samples)//2+1),((n_samples)//2+1,1))
            string = np.vstack( [pt1+t1*(mid_pt-pt1) , (mid_pt+t2*(pt2-mid_pt))[1:,:] ] )
            mid_index = t1.shape[0] - 1
    else:
        t = np.reshape(np.linspace(0,1,n_samples),(n_samples,1))
        string = pt1+t*(pt2-pt1)
        mid_index = -1
    delta = 1.0
    itr = 1
    #sorted_col = np.argsort(df.var().to_numpy())[::-1]
    if not pmf_input:
        if all_dims:
            sorted_pairs = pairdim_by_distance((string[0,:],string[-1,:]),len(df.columns))
        else:
            sorted_pairs = dim_by_distance((string[0,:],string[-1,:]),len(df.columns))
    else:
        sorted_pairs = [(0,1)]

    #initial minimization with small step
    for dim_pair in sorted_pairs:
        pair = list(dim_pair)
        if plots:
                print("Initial warming-up on {0} and {1}".format(dim_pair[0],dim_pair[1]))
        if not pmf_input:
            string[:,pair] = ZTS_2D((df.iloc[:,dim_pair[0]],df.iloc[:,dim_pair[1]]),string[:,pair],n_samples,bins,step_size*0.1,pmf_input=False,plots=plots)
        else:
            string[:,pair] = ZTS_2D(df,string[:,pair],n_samples,bins,step_size*0.1,pmf_input=True,plots=plots)
    
    while (delta>tolerance and itr<=max_itr) or itr<=2:
        old_string = string.copy()
        ## string in every two dimensions
        #for idx in range(0,len(sorted_col)-1,2):
        #    dim_pair = (sorted_col[idx],sorted_col[idx+1])
        ## string in every combination of two dimensions, in order of decreasing Var
        #for dim_pair in itertools.combinations(sorted_col,2):
        ## string in every combination of two dimensions, in order of decreasing distance between endpoints
        for dim_pair in sorted_pairs:
        #for dim_pair in itertools.combinations(range(len(df.columns)),2):
            old_string = string.copy()
            pair = list(dim_pair)
            if plots:
                print("Iteration on {0} and {1}".format(dim_pair[0],dim_pair[1]))
            if not pmf_input:
                string[:,pair] = ZTS_2D((df.iloc[:,dim_pair[0]],df.iloc[:,dim_pair[1]]),string[:,pair],n_samples,bins,step_size,pmf_input=False,plots=plots)
            else:
                string[:,pair] = ZTS_2D(df,string[:,pair],n_samples,bins,step_size,pmf_input=True,plots=plots)
        #delta = np.mean(np.abs(string-old_string))
        delta = np.max(np.abs(string-old_string))
        itr += 1
        step_size *= 0.7
        print("Change in string = {0}".format(delta))
    if len(refine_dims)>0:
        print("Refining string in {0} and {1} dimensions...".format(refine_dims[0],refine_dims[1]))
        string[:,refine_dims] = ZTS_2D((df.iloc[:,refine_dims[0]],df.iloc[:,refine_dims[1]]),string[:,refine_dims],n_samples,bins,step_size,pmf_input=False,plots=plots)
    #reparameterize string in all dimensions
    arclength = np.hstack([0,np.cumsum(np.linalg.norm(string[1:] - string[:-1],axis=1))])
    arclength /= arclength[-1]
    string = np.vstack([interp1d(arclength,string[:,i])(np.linspace(0,1,n_samples)) 
                           for i in range(string.shape[1])]).T
    if itr <= max_itr:    
        print("String converged.")
    else:
        print("Maximum iteration reached, please check the string!")
    return string

def PMF_string(input_data,string,fname="",pmf_input=False,reparam_n=0,yscale=(0,6,1),bins=50, Temperature=300, outPMF=False):
	##calculate PMF (in 2D) values along a given string
	#input_data - raw coordinate data or 2D PMF matrix (pmf_input=True)
    def pmf(xdata,ydata,bins):
        x , y = xdata , ydata
        h , xe, ye = np.histogram2d(x,y,bins=bins,range=[[np.min(x)-0.02,np.max(x)+0.02],[np.min(y)-0.02,np.max(y)+0.02]])
        X, Y = np.meshgrid(xe,ye)

        RT =  0.00198588*Temperature
        F = -RT*np.log((h.T+1e-18)/np.sum(h))
        F = F - np.min(F)

        return (X,Y,F)
    
    def rounding(x):
        return int(x*10)/10
    
    n_samples = string.shape[0]
    if not pmf_input:
        data = pmf(input_data[0],input_data[1],bins)
        x = data[0]
        y = data[1]
        z = data[2]
        dx = x[0][1] - x[0][0]
        dy = y[1][0] - y[0][0]
        x = x + dx/2
        y = y + dy/2
        gridpoints = np.vstack([x[:-1,:-1].flatten(),y[:-1,:-1].flatten()]).T
    else:
        #input is assumed of the form (xcoor,ycoor,pmf_value) for each row 
        if type(bins) == int:
            bins = (bins,bins)
        x, y = input_data[:,0].reshape(bins).T , input_data[:,1].reshape(bins).T 
        dx , dy = x[0,1] - x[0,0] , y[1,0] - y[0,0]
        z = input_data[:,2].reshape(bins).T
        gridpoints = np.vstack([x.flatten(),y.flatten()]).T
    
    fig, ax = plt.subplots(1,1)
    #interpolate between points along the string
    if reparam_n>0:
    	#normalized each dim before reparameterization
    	st_lims = np.max(string,axis=0) - np.min(string,axis=0)
    	string /= st_lims
        arclength = np.hstack([0,np.cumsum(np.linalg.norm(string[1:] - string[:-1],axis=1))])
        arclength /= arclength[-1]
        string = np.vstack([interp1d(arclength,string[:,0])(np.linspace(0,1,reparam_n)), interp1d(arclength,string[:,1])(np.linspace(0,1,reparam_n))]).T
        n_samples = reparam_n
        string *= st_lims
    xpts, ypts = np.linspace(0,1,n_samples) , griddata(gridpoints,z.flatten(),(string[:,0],string[:,1]), method='linear')
    ax.plot(xpts,ypts)
    plt.yticks(np.arange(rounding(yscale[0]),rounding(yscale[1])+yscale[2],step=yscale[2]),fontsize=15)
    plt.ylim([yscale[0],yscale[1]])
    ax.tick_params(axis='both',labelsize=20)
    ax.set_ylabel("Free Energy (kcal/mol)",fontsize=22)
    ax.set_xlabel("Reaction Coordinate",fontsize=22)
    plt.tight_layout(pad=1)
    if outPMF:
        return (xpts,ypts)
    if len(fname)>0:
        fig.savefig(fname,dpi=200)
