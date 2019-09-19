from scipy.interpolate import griddata, interp1d
#from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import itertools

def find_ZTS(df,frame,n_samples=20,bins=50):
    #calculate 2D pmf
    def pmf(xdata,ydata,bins):
        x , y = xdata , ydata
        h , xe, ye = np.histogram2d(x,y,bins=bins,range=[[np.min(x)-0.02,np.max(x)+0.02],[np.min(y)-0.02,np.max(y)+0.02]])
        X, Y = np.meshgrid(xe,ye)

        RT = 0.593
        F = -RT*np.log((h.T+1e-10)/np.sum(h))
        F = F - np.min(F)

        return (X,Y,F)

    def ZTS_2D(xdata,ydata,pts,n_samples,bins,step_size=0.01,plots=False):
        data = pmf(xdata,ydata,bins)
        x = data[0]
        y = data[1]
        z = data[2]
        dx = x[0][1] - x[0][0]
        dy = y[1][0] - y[0][0]
        x = x + dx/2
        y = y + dy/2
        gridpoints = np.vstack([x[:-1,:-1].flatten(),y[:-1,:-1].flatten()]).T

        #Make a figure and a axes
        if plots:
            fig, ax = plt.subplots(1,1)

        gradx, grady = np.gradient(z,dx,dy)
        #Evolve points so that they respond to the gradients. This is the "zero temperature string method"
        stepmax=100
        for i in range(stepmax):
            #Find gradient interpolation
            Dx = griddata(gridpoints,gradx.flatten(),(pts[:,0],pts[:,1]), method='nearest',fill_value=0.0)
            Dy = griddata(gridpoints,grady.flatten(),(pts[:,0],pts[:,1]), method='nearest',fill_value=0.0)
            h = np.amax(np.sqrt(np.square(Dx)+np.square(Dy)))
            
            #Evolve
            pts -= step_size * np.vstack([Dy,Dx]).T / h

            #Reparameterize
            arclength = np.hstack([0,np.cumsum(np.linalg.norm(pts[1:] - pts[:-1],axis=1))])
            arclength /= arclength[-1]
            #pts = np.vstack([interp1d(arclength,pts[:,0])(np.linspace(0,1,2*npts)), interp1d(arclength,pts[:,1])(np.linspace(0,1,2*npts))]).T
            pts = np.vstack([interp1d(arclength,pts[:,0])(np.linspace(0,1,n_samples)), interp1d(arclength,pts[:,1])(np.linspace(0,1,n_samples))]).T
            if i % 20 and plots:
                print(i, np.sum(griddata(gridpoints,z.flatten(),(pts[:,0],pts[:,1]), method='linear')))
                #This draws the intermediate states to visualize how the string evolves.
                #ax.plot(pts[:,0],pts[:,1],color='r',marker='x',linestyle='--')
                ax.plot(pts[:,0],pts[:,1], color=plt.cm.spring(i/float(stepmax)))

        if plots:
            ax.plot(pts[:,0],pts[:,1], color='k', linestyle='-', marker='o')
            heatmap = ax.imshow(z, cmap=plt.cm.rainbow, vmin = np.nanmin(z), vmax=np.nanmin(z)+12, origin='lower', aspect='auto', extent = (y[0][0], y[-1][-1],x[0][0], x[-1][-1]), interpolation="bicubic")
            heatmap.cmap.set_over('white')
            ax.autoscale(False)

            bar = fig.colorbar(heatmap)
            bar.set_label("Free Energy (kcal/mol)", rotation=90, fontname="Avenir", fontsize=14)

            fig, ax = plt.subplots(1,1)
            ax.plot(np.linspace(0,1,n_samples),griddata(gridpoints,z.flatten(),(pts[:,1],pts[:,0]), method='linear'))
            ax.set_ylabel("Free Energy (kcal/mol)")
            ax.set_xlabel("Reaction Progress")
            #fig.savefig("1Dpmf.png")

        return pts
    
    #initialize string
    t = np.reshape(np.linspace(0,1,n_samples),(n_samples,1))
    pt1 = df.iloc[frame[0]].to_numpy()
    pt2 = df.iloc[frame[1]].to_numpy()
    string = pt1+t*(pt2-pt1)
    
    delta = 1.0
    max_itr = 10
    step_size = 0.02
    while delta>0.012 and max_itr>0:
        old_string = string.copy()
        for dim_pair in itertools.combinations(range(len(df.columns)),2):
            old_string = string.copy()
            pair = list(dim_pair)
            string[:,pair] = ZTS_2D(df.iloc[:,dim_pair[0]],df.iloc[:,dim_pair[1]],string[:,pair],n_samples,bins,step_size)
        delta = np.mean(np.abs(string-old_string))
        max_itr -= 1
        step_size *= 0.8
        print("Change in string = {0}".format(delta))
        
    print("String converged.")
    return string

def PMF_string(xdata,ydata,string,fname="",n_samples=20,bins=50):
    def pmf(xdata,ydata,bins):
        x , y = xdata , ydata
        h , xe, ye = np.histogram2d(x,y,bins=bins,range=[[np.min(x)-0.02,np.max(x)+0.02],[np.min(y)-0.02,np.max(y)+0.02]])
        X, Y = np.meshgrid(xe,ye)

        RT = 0.593
        F = -RT*np.log((h.T+1e-10)/np.sum(h))
        F = F - np.min(F)

        return (X,Y,F)
    
    data = pmf(xdata,ydata,bins)
    x = data[1]
    y = data[0]
    z = data[2]
    dx = x[1][0] - x[0][0]
    dy = y[0][1] - y[0][0]
    x = x + dx/2
    y = y + dy/2
    gridpoints = np.vstack([x[:-1,:-1].flatten(),y[:-1,:-1].flatten()]).T
    print(np.shape(gridpoints))
    fig, ax = plt.subplots(1,1)
    ax.plot(np.linspace(0,1,n_samples),griddata(gridpoints,z.flatten(),(string[:,1],string[:,0]), method='linear'))
    ax.tick_params(axis='both',labelsize=20)
    ax.set_ylabel("Free Energy (kcal/mol)",fontsize=22)
    ax.set_xlabel("Reaction Coordinate",fontsize=22)
    plt.tight_layout(pad=1)
    if len(fname)>0:
        fig.savefig(fname,dpi=200)