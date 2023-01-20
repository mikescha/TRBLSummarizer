import matplotlib.pyplot as plt
import pandas as pd

def plot(data, w, h):
    dpi = 300

    fig, ax = plt.subplots(figsize = (w,h), dpi=dpi)
    ax.plot(data)
    fname = "file_{}.png".format(w)
    plt.savefig(fname)
    plt.close()


dates = pd.date_range("2022-01-01", "2022-01-06").tolist()
temp_data = [60,70,80,70,60,80]
precip_data = [0,0,1,0.5,0,.6]

df_temp = pd.DataFrame(data=temp_data, index=dates, columns=["High temp"])
df_precip = pd.DataFrame(data=precip_data, index=dates, columns=["Precip"])

w = 10
h = 5

plot(df_temp, w, h)
plot(df_precip, w + 1, h)
