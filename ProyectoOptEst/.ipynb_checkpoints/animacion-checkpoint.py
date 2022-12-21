import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data=np.load("data/anim/sol600.npz")

sol=data["sol"]
tsol=data["t"]
centroS=data["centroS"]
tiemposDeci=data["tiemposDeci"]
pol=data["pol"]
def darpol(t):
	ind=np.where(tiemposDeci<t)[0][-1]
	return pol[ind]
print(darpol(3.0))

def init():
    im.set_data(sol[0,:,:][:,::-1].T)
    ant.set(text="t=0.0")
    return [im,ant,an1,an2,li]
# animation function.  This is called sequentially
def animate(i):
    print(i)
    im.set_array(sol[i,:,:][:,::-1].T)
    ant.set(text="t="+str(round(tsol[i],1)))
    pol=darpol(tsol[i])
    an1.set(text=pol[0])
    an2.set(text=pol[1])
    li.set_data([50*centroS[i][0]],[50*(1-centroS[i][1])])
    print([centroS[i][1]],[centroS[i][0]])
    return [im,ant,an1,an2,li]
    
fig, ax = plt.subplots()
im=ax.imshow(sol[0,:,:][:,::-1].T,cmap="coolwarm",interpolation="nearest", interpolation_stage='rgba',vmin=-0.5,vmax=6)
fig.colorbar(im,ax=ax)
ant=ax.text(5,5,"t=0.0")
an1=ax.text(11,47,"F")
an2=ax.text(35,47,"F")
li,=ax.plot([50*centroS[0][0]],[50*(1-centroS[0][1])],'-ro')
anim = FuncAnimation(fig, animate, init_func=init,frames=range(2,200,1), interval=10)
plt.show()
