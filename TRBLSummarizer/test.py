import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

w = 6.5
h = 3

def make_plot(df, dpi):
    cmap = ['Reds', 'Blues', 'Greens']
    row_count = len(cmap)
    fig, axs = plt.subplots(nrows = row_count, ncols = 1,
                            sharex = 'col', dpi=dpi,
                            gridspec_kw={'height_ratios': np.repeat(1,row_count), 
                                            'left':0, 'right':1, 'bottom':0, 'top':1,
                                            'hspace':0},
                            figsize=(w,h))
    r=0
    width = len(df.columns) * (dpi/100)
    for c in cmap:
        axs[r].imshow(df, cmap=c, interpolation='nearest',
                    origin='lower', extent=(0,width,0,width/w))
        axs[r].title.set_visible(False)
        axs[r].axis('off')
        r += 1
    
    fname = f"plot_{dpi}dpi.png"
    plt.savefig(fname)
    plt.show()
    plt.close()

dates = pd.date_range("2022-01-01", "2022-01-30").tolist()
temp_data = np.arange(0,30)
df = pd.DataFrame(data=temp_data, index=dates, columns=["High temp"])
df = df.transpose()

make_plot(df, 100)
make_plot(df, 300)



#        x = np.arange(30)
#        y = 3.*np.sin(x*2.*np.pi/30.)
#        #axs[r].plot(x, y)

#fig.tight_layout()
#plt.subplots_adjust(left=0, bottom=0, right=0.001, top=0.001, wspace=0, hspace=0)

#axs[r].autoscale(enable=False, axis='both')
#axs[r].set_xlim(0,100*scale)
#axs[r].set_ylim(0,10*scale)

#axs[r].tick_params(
#    axis='y',
#    which='both',      # both major and minor ticks are affected
##    left=False, right=False,  # ticks along the sides are off
#   labelleft=False, labelright=False) # labels on the Y are off 
#axs[r].margins(x=0,y=0)
