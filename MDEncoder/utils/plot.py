import matplotlib.pyplot as plt
import numpy as np

def plot_energy(xdata,ydata,frame=[],fname='',xylab=('X','Y'),xscale=(-1,1,0.2),yscale=(-1,1,0.2),cscale=(0,5),bins=60,Temperature=300):
    
    x , y = xdata , ydata
    h , xe, ye = np.histogram2d(x,y,bins=bins,range=[[xscale[0],xscale[1]],[yscale[0],yscale[1]]])
    X, Y = np.meshgrid(xe,ye)
    plt.figure(figsize=(6,5))

    RT = 0.00198588*Temperature
    F = -RT*np.log((h.T)/np.sum(h))
    F = F - np.min(F)
    plt.pcolormesh(X,Y,F, cmap='jet')
    plt.clim(cscale[0],cscale[1])
    c = plt.colorbar()
    c.ax.set_ylabel('Kcal/mol',fontsize=20)
    
    for n, f in enumerate(frame):
        plt.text(x[f],y[f], str(n+1) ,fontsize=26)
    
    def rounding(x):
        return int(x*10)/10
    plt.xlabel(xylab[0],fontsize=22)
    plt.ylabel(xylab[1],fontsize=22)
    plt.xticks(np.arange(rounding(xscale[0]),rounding(xscale[1])+xscale[2],step=xscale[2]),fontsize=15)
    plt.yticks(np.arange(rounding(yscale[0]),rounding(yscale[1])+yscale[2],step=yscale[2]),fontsize=15)
    plt.xlim([xscale[0],xscale[1]])
    plt.ylim([yscale[0],yscale[1]])
    plt.tight_layout()
    fig = plt.gcf()
    if len(fname)>0:
        fig.savefig(fname,dpi=200)
    
    return fig