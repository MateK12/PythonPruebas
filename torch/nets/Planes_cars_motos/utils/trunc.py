import math

def redondear_si_mayor_65(numero):
    entero, decimal = math.modf(numero)
    
    if decimal > 0.65:
        return math.ceil(numero)
    else:
        return int(entero)