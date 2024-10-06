import matplotlib as mtlpt
from matplotlib import pyplot as plt

def Cuadratic(x): #python doesnt have hoising
    return (2*(x**2)+3*x+10)
def derivative(x):
    return (4*x+3)
def Plot_data(x,y,xLabel='abscisas',yLabel='ordenadas',curve_label='etiqueta de la curva',title='titulo'):
    fig, ax = plt.subplots() #returns a figure, and a axes manipulator

    fig.suptitle(title) #sets the window title

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    ax.plot(x,y,label=curve_label) #small rectangle at the top corner

    plt.show()
    

# [ (any_modification(var) ) for var in iterator] read the other way around,then, fills the list

if __name__ == '__main__': #if this file is being run =>
    Plot_data(range(-20,20),[Cuadratic(var) for var in range(-20,20)],xLabel='X',yLabel='Y',curve_label='2x^2+3x+10',title='Cuadratic')
    Plot_data(range(-20,20),[derivative(var) for var in range(-20,20)],xLabel='X',yLabel='Y',curve_label='4x+3',title='Cuadratic derivatives')
    #otherwise python will execute all the file when we use an imported method from here
    